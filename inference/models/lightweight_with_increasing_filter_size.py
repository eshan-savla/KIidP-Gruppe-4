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
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.channel_size = channel_size
        self.dropout = dropout
        self.prob = prob
        
        self._build_layers(channel_size)
        
    def _build_layers(self, channel_size):
        # Define the layers using the provided channel_size
        self.conv1 = nn.Conv2d(self.input_channels, channel_size, kernel_size=3, stride=1, padding=1) 
        self.bn1 = nn.BatchNorm2d(channel_size)
        self.conv2 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size)
        self.conv3 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size)
        self.conv4 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size) 
        
        self.mxP1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1) 
        self.bn5 = nn.BatchNorm2d(channel_size)
        self.conv6 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(channel_size)

        self.mxP2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv7 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(channel_size)
        self.conv8 = nn.Conv2d(channel_size, channel_size, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(channel_size)

        self.res1 = ResidualBlock(channel_size, channel_size)
        self.res2 = ResidualBlock(channel_size, channel_size)
        self.res3 = ResidualBlock(channel_size, channel_size)

        self.rfb = ReceptiveFieldBlock(channel_size, channel_size)
        
        self.attention_fusion_1 = MultiDimensionalAttentionFusion(channel_size * 2)

        self.convT4 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(channel_size)

        self.attention_fusion_2 = MultiDimensionalAttentionFusion(channel_size * 2)

        self.convT5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4, stride=2, padding=1)
        self.bn10 = nn.BatchNorm2d(channel_size)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=self.output_channels, kernel_size=3, stride=1, padding=1)

        self.dropout_pos = nn.Dropout(p=self.prob)
        self.dropout_cos = nn.Dropout(p=self.prob)
        self.dropout_sin = nn.Dropout(p=self.prob)
        self.dropout_wid = nn.Dropout(p=self.prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)
    
    def update_filter_size(self, new_channel_size):
        self._build_layers(new_channel_size)

    def forward(self, x_in):
        # Forward method remains the same
        x01 = F.relu(self.bn1(self.conv1(x_in)))
        x02 = F.relu(self.bn2(self.conv2(x01)))
        x03 = F.relu(self.bn3(self.conv3(x02)))
        x04 = F.relu(self.bn4(self.conv4(x03)))

        x05 = self.mxP1(x04)
        
        x06 = F.relu(self.bn5(self.conv5(x05)))
        x07 = F.relu(self.bn6(self.conv6(x06)))

        x08 = self.mxP2(x07)
        
        x09 = F.relu(self.bn7(self.conv7(x08)))
        x10 = F.relu(self.bn8(self.conv8(x09)))
        
        x11 = self.res1(x10)
        x12 = self.res2(x11)
        x13 = self.res3(x12)

        x14 = self.rfb(x13)
        
        x15 = torch.cat((x14, x10), 1)
        x16 = self.attention_fusion_1(x15)

        x17 = F.relu(self.bn9(self.convT4(x16)))

        x18 = torch.cat((x17, x07), 1)
        x19 = self.attention_fusion_2(x18)

        x20 = F.relu(self.bn10(self.convT5(x19)))

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x20))
            cos_output = self.cos_output(self.dropout_cos(x20))
            sin_output = self.sin_output(self.dropout_sin(x20))
            width_output = self.width_output(self.dropout_wid(x20))
        else:
            pos_output = self.pos_output(x20)
            cos_output = self.cos_output(x20)
            sin_output = self.sin_output(x20)
            width_output = self.width_output(x20)

        return pos_output, cos_output, sin_output, width_output

