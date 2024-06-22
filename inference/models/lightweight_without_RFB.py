import torch.nn as nn
import torch
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
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size)
        self.conv3 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.conv4 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size) 
        
        self.mxP1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # MaxPooling to half the dimensions 

        self.conv5 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1) 
        self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(channel_size)

        self.mxP2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) # MaxPooling to half the dimensions

        self.conv7 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(channel_size)
        self.conv8 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(channel_size)

        # Bottleneck 
        self.res1 = ResidualBlock(channel_size, channel_size) # with k = 2 (amount of Conv2d layers in ResidualBlock)
        self.res2 = ResidualBlock(channel_size, channel_size) # with k = 2
        self.res3 = ResidualBlock(channel_size, channel_size) # with k = 2

        # self.rfb = ReceptiveFieldBlock(channel_size, channel_size)  

        # Multi Dimensional Attention Fusion Block 1      
        self.attention_fusion_1 = MultiDimensionalAttentionFusion(channel_size * 2) # doubled amount of channels due to concatenation

        # Upsampling Blocks
        self.convT4 = nn.ConvTranspose2d(channel_size * 2 , channel_size, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        # Multi Dimensional Attention Fusion Block 2     
        self.attention_fusion_2 = MultiDimensionalAttentionFusion(channel_size * 2) # doubled amount of channels due to concatenation

        # Upsampling Block 2
        self.convT5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(channel_size)

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
        x04 = F.relu(self.bn4(self.conv4(x03)))

        x05 = self.mxP1(x04)
        
        x06 = F.relu(self.bn2(self.conv5(x05)))
        x07 = F.relu(self.bn3(self.conv6(x06)))

        x08 = self.mxP2(x07)
        
        x09 = F.relu(self.bn3(self.conv7(x08)))
        x10 = F.relu(self.bn4(self.conv8(x09)))
        
        # Bottleneck 
        x11 = self.res1(x10)
        x12 = self.res1(x11)
        x13 = self.res1(x12)

        # Receptive Field Block
        # x14 = self.rfb(x13)
        
        # Multi Dimensional Attention Fusion 1
        x15 = torch.cat((x13, x10), 1) # Concatenation of shallow and deep features, Dimension = 1
        x16 = self.attention_fusion_1(x15)

        # Upsampling Block 1
        x17 = F.relu(self.bn5(self.convT4(x16)))

        # Multi Dimensional Attention Fusion 2
        x18 = torch.cat((x17, x07), 1) # Concatenation of shallow and deep features, Dimension = 1
        x19 = self.attention_fusion_2(x18)

        # Upsampling Block 2
        x20 = F.relu(self.bn6(self.convT5(x19)))

        # Output
        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x20))
            cos_output = self.cos_output(self.dropout_cos(x20))
            sin_output = self.sin_output(self.dropout_sin(x20))
            # angle_output = torch.atan2(sin_output, cos_output) / 2    
            width_output = self.width_output(self.dropout_wid(x20))
        else:
            pos_output = self.pos_output(x20)
            cos_output = self.cos_output(x20)
            sin_output = self.sin_output(x20)
            # angle_output = torch.atan2(sin_output, cos_output) / 2    
            width_output = self.width_output(x20)

        return pos_output, cos_output, sin_output , width_output     #angle_output -> other possible output depending on evaluation
