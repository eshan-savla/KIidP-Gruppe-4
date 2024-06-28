import torch.nn as nn
import torch
# import torch.atan2 as atan2
import torch.nn.functional as F

from inference.models.grasp_model import GraspModel, ResidualBlock
# from inference.models.rfb import ReceptiveFieldBlock
from inference.models.rfb_padding import ReceptiveFieldBlock

####  Added for the Multi Dimensional Attention Fusion #####
from inference.models.multi_dimensional_attention_fusion import MultiDimensionalAttentionFusion

class GenerativeResnet(GraspModel):

    def __init__(self, input_channels=4, output_channels=1, channel_size=32, dropout=False, prob=0.0):
        super(GenerativeResnet, self).__init__()
        
        # Downsampling Block
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4) 
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1) 
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1) 
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        # Bottleneck
        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4) 
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4) 
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)

        # 1x Receptive Field Block
        self.rfb = ReceptiveFieldBlock(channel_size * 4, channel_size * 4)  

        # Upsampling Blocks
        self.convT4 = nn.ConvTranspose2d(channel_size * 4 * 2, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)
        
        self.convT5 = nn.ConvTranspose2d(channel_size * 2 * 2, channel_size * 1, kernel_size=4, stride=2, padding=1) # channel_size * 1 as final layer
        self.bn5 = nn.BatchNorm2d(channel_size * 1)

        # Multi Dimensional Attention Fusion Blocks      
        self.attention_fusion_1 = MultiDimensionalAttentionFusion(channel_size * 4 * 2) # dimensions changed due to concatenation

        # Multi Dimensional Attention Fusion Blocks      
        self.attention_fusion_2 = MultiDimensionalAttentionFusion(channel_size * 2 * 2) # updated to match concatenation dimensions
       
        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, stride=1, padding=1) #kernel_size=3 to match dimensions
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, stride=1, padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=3, stride=1, padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        
        # Downsampling Block
        x01 = F.relu(self.bn1(self.conv1(x_in)))
        x02 = F.relu(self.bn2(self.conv2(x01)))
        x03 = F.relu(self.bn3(self.conv3(x02)))
        
        # Bottleneck 
        x04 = self.res1(x03)
        x05 = self.res1(x04)
        x06 = self.res1(x05)

        # Receptive Field Block
        x07 = self.rfb(x06)

        
        # Multi Dimensional Attention Fusion 1
        x08 = torch.cat((x07, x03), 1) # Concatenation of shallow and deep features, Dimension = 1
        x09 = self.attention_fusion_1(x08)

        # Upsampling Block
        x10 = F.relu(self.bn4(self.convT4(x09)))


        # Multi Dimensional Attention Fusion 2
        x11 = torch.cat((x10, x02), 1) # Concatenation of shallow and deep features, Dimension = 1
        x12 = self.attention_fusion_2(x11)

        # Upsampling Block
        x13 = F.relu(self.bn5(self.convT5(x12)))

        # Output
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x13))
            cos_output = self.cos_output(self.dropout_cos(x13))
            sin_output = self.sin_output(self.dropout_sin(x13))
            # angle_output = atan2(sin_output, cos_output) / 2
         
            cos_output + sin_output
            width_output = self.width_output(self.dropout_wid(x13))
        else:
            pos_output = self.pos_output(x13)
            cos_output = self.cos_output(x13)
            sin_output = self.sin_output(x13)
            # angle_output = atan2(sin_output, cos_output) / 2
            width_output = self.width_output(x13)

        return pos_output, cos_output, sin_output, width_output #,angle_output    -> other possible output depending on evaluation
