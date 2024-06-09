import torch
import torch.nn as nn

class PixelAttention(nn.Module):
    def __init__(self, in_channels):
        super(PixelAttention, self).__init__()
        # Define a 3x3 convolutional layer to generate the attention map
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3)
        # Sigmoid activation function to normalize the attention map values between 0 and 1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate the attention map using the convolutional layer
        attention = self.sigmoid(self.conv(x))
        # Multiply the input by the attention map to emphasize important pixels
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        # Adaptive average pooling to generate a channel descriptor
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Fully connected layers to learn the channel attention weights
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Apply average pooling to get the channel descriptor
        avg_out = self.fc(self.avg_pool(x))
        # Multiply the input by the channel attention weights
        return x * avg_out

class MultiDimensionalAttentionFusion(nn.Module):
    def __init__(self, in_channels):
        super(MultiDimensionalAttentionFusion, self).__init__()
        # Instantiate the pixel attention module
        self.pixel_attention = PixelAttention(in_channels)
        # Instantiate the channel attention module
        self.channel_attention = ChannelAttention(in_channels)

    def forward(self, x):
        # Apply pixel attention to the input
        x = self.pixel_attention(x)
        # Apply channel attention to the result
        x = self.channel_attention(x)
        # Return the attention-enhanced output
        return x
