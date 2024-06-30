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
        
        # Downsampling
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
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)
        
        # Change in Comparision to without rfb
        self.rfb = ReceptiveFieldBlock(channel_size * 4, channel_size * 4)  


        ####  Added for the Multi Dimensional Attention Fusion #####
        self.attention_fusion = MultiDimensionalAttentionFusion(channel_size * 4)


        # Upsampling
        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4, stride=2, padding=1,
                                        output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=2,
                                        output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x01 = F.relu(self.bn1(self.conv1(x_in)))
        x02 = F.relu(self.bn2(self.conv2(x01)))
        x03 = F.relu(self.bn3(self.conv3(x02)))
        x04 = self.res1(x03)
        x05 = self.res2(x04)
        x06 = self.res3(x05)
        x07 = self.res4(x06)
        x08 = self.res5(x07)

        x09 = self.rfb(x08)

        ####  Added for the Multi Dimensional Attention Fusion #####
        x10 = torch.cat((x09, x03), 1) # Concatenation of shallow and deep features, Dimension = 1
        x11 = self.attention_fusion(x10)

        x12 = F.relu(self.bn4(self.conv4(x11)))
        x13 = F.relu(self.bn5(self.conv5(x12)))
        x14 = self.conv6(x13)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x14))
            cos_output = self.cos_output(self.dropout_cos(x14))
            sin_output = self.sin_output(self.dropout_sin(x14))
            width_output = self.width_output(self.dropout_wid(x14))
        else:
            pos_output = self.pos_output(x14)
            cos_output = self.cos_output(x14)
            sin_output = self.sin_output(x14)
            width_output = self.width_output(x14)

        return pos_output, cos_output, sin_output, width_output
