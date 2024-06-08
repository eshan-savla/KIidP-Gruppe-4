import torch.nn as nn
import torch
import torch.nn.functional as F

class ReceptiveFieldBlock(nn.Module):
    """
    A receptive field block as improvement example --> enhance multi scale grasp detection capabilities (e.g. a conin is detected the same way than a coke can)
    Inspired by human visual context handling (neuroscience research) --> helps humans to emphasize the importance of the area near the visual center and blur the further surroundings

    Advantages/ Chances of RFB:
    - capturing multi scale contextuel information
    - enhancing feature representation and discriminability
    - improving semantic understanding --> should be important to run the grasp detection modell on cluttered senes like the graspnet dataset
    - reduce overfitting, improve generalization and learn more robust features --> skip connections adress vanishing gradint problem (same as normal ResidualBlock)

    --> WITHOUT the need to create a deeper sequential network (vanishing gradient problem)

    Architecture inspired by -------------------------------- TODO

    Explanations:
    - 1x1 conv layers are used at the input of each branch and after all branches are concatenated to reduce dimension by reducing number of channels to learn generalized feature representations and enhance effiency

    """

    def __init__(self, in_channels, out_channels):    # count filters = out_channels --> every filter learns to detect features in every in_channel
        # init the module (network) with a spec. name
        super(ReceptiveFieldBlock, self).__init__()

        inter_channels = in_channels // 8   # integer division, no floats --> channel variable to use for easier internal definitions of filter count per layer
        branch_out_channels = inter_channels*2

        # define the layers of the network (block)
        # --> each branch defined with layers of different kernel sizes to adress other scales of features and connect them during the training, branches ae fused afterwards

        ## branch 0 ##
        self.conv01 = nn.Conv2d(in_channels, 2*inter_channels, kernel_size=1, padding=0) # reduce input channel dimensions to 1/4 of input
        self.bn01 = nn.BatchNorm2d(2*inter_channels) # has to equal out_channels of conv layer the batchnorm should be applied to

        self.conv02 = nn.Conv2d(2*inter_channels, branch_out_channels, kernel_size=3, padding=1)  # 3x3 kernel
        self.bn02 = nn.BatchNorm2d(branch_out_channels)

        ### branch 1 ###
        self.conv11 = nn.Conv2d(in_channels, 2*inter_channels, kernel_size=1, padding=0)
        self.bn11 = nn.BatchNorm2d(2*inter_channels) 

        self.conv12 = nn.Conv2d(2*inter_channels, 2*inter_channels, kernel_size=3, padding=1) 
        self.bn12 = nn.BatchNorm2d(2*inter_channels)

        self.conv13 = nn.Conv2d(2*inter_channels, branch_out_channels, kernel_size=3, dilation=3, padding=1)
        self.bn13 = nn.BatchNorm2d(branch_out_channels)

        ### branch 2 ###
        self.conv21 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, padding=0)
        self.bn21 = nn.BatchNorm2d(inter_channels)

        self.conv22 = nn.Conv2d(inter_channels, 2*inter_channels, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(2*inter_channels)

        self.conv23 = nn.Conv2d(2*inter_channels, 2*inter_channels, kernel_size=3, padding=1)
        self.bn23 = nn.BatchNorm2d(2*inter_channels)

        self.conv23 = nn.Conv2d(2*inter_channels, branch_out_channels, kernel_size=3, dilation=5, padding=1)
        self.bn23 = nn.BatchNorm2d(branch_out_channels)

        ### branch 3 ###
        self.conv31 = nn.Conv2d(in_channels, 4*inter_channels, kernel_size=1, padding=0)
        self.bn31 = nn.BatchNorm2d(4*inter_channels)

        self.conv32 = nn.Conv2d(4*inter_channels, 6*inter_channels, kernel_size=(1,7), padding=1) # not rectangular filters
        self.bn32 = nn.BatchNorm2d(6*inter_channels)

        self.conv33 = nn.Conv2d(6*inter_channels, 6*inter_channels, kernel_size=(7,1), padding=1) # perform filter flipped by 90 deg to generate quadr. output featuremap
        self.bn33 = nn.BatchNorm2d(6*inter_channels)
        
        self.conv34 = nn.Conv2d(6*inter_channels, branch_out_channels, kernel_size=3, dilation=7, padding=1)
        self.bn34 = nn.BatchNorm2d(branch_out_channels)

        ### branch 4 ###
        self.conv41 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn41 = nn.BatchNorm2d(out_channels)

        # concatenation layer to rescale channels - define out channels (feature map depth) without rescale spital dimensions (heiht x width)
        self.conv5 = nn.Conv2d(branch_out_channels*4, out_channels, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm2d(out_channels)



    def forward(self, x_in):
        ## branch 0 ##
        x0 = self.bn01(self.conv01(x_in))
        x0 = F.relu(x0)
        x0 = self.bn02(self.conv02(x0))
        x0 = F.relu(x0)

        ## branch 1 ##
        x1 = self.bn11(self.conv11(x_in))
        x1 = F.relu(x1)  # introduce nonlinearity to model with nonlinear activations
        x1 = self.bn12(self.conv12(x1))
        x1 = F.relu(x1)
        x1 = self.bn13(self.conv13(x1))
        x1 = F.relu(x1)

        ## branch 2 ##
        x2 = self.bn21(self.conv21(x_in))
        x2 = F.relu(x2)
        x2 = self.bn22(self.conv22(x2))
        x2 = F.relu(x2)
        x2 = self.bn23(self.conv23(x2))
        x2 = F.relu(x2)

        ## branch 3 ##
        x3 = self.bn31(self.conv31(x_in))
        x3 = F.relu(x3)
        x3 = self.bn32(self.conv32(x3))
        x3 = F.relu(x3)
        x3 = self.bn33(self.conv33(x3))
        x3 = F.relu(x3)
        x3 = self.bn34(self.conv34(x3))
        x3 = F.relu(x3)

        ## branch 4 ##
        x4 = self.bn41(self.conv41(x_in))
        x4 = F.relu(x4)


        ## concatenation ## --> note: all branches have the same output dimensions [batch_size, channels, height, width] to concatenate
        height = x_in.shape[2]
        width = x_in.shape[3]

        # pad all tensors to desired (SAME!!!) spital dimensions 
        x0_pad = F.pad(x0,(0,width - x0.shape[3], 0, height - x0.shape[2]))
        x1_pad = F.pad(x1,(0,width - x1.shape[3], 0, height - x1.shape[2]))
        x2_pad = F.pad(x2,(0,width - x2.shape[3], 0, height - x2.shape[2]))
        x3_pad = F.pad(x3,(0,width - x3.shape[3], 0, height - x3.shape[2]))
        x4_pad = F.pad(x4,(0,width - x4.shape[3], 0, height - x4.shape[2]))

        # concatenate branches 0 - 3
        x = torch.cat((x0_pad, x1_pad, x2_pad, x3_pad), 1)  # concatenate along the channel dimension (add in channel dimension: x.channels = 4* branch_out_channels)

        # add branch 4 elementwise
        x = x + x4_pad

        ## last dimension rescaling to output channel size ##
        x = self.bn5(self.conv5(x))

        return x
