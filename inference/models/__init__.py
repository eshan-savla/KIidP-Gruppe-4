def get_network(network_name):
    network_name = network_name.lower()
    
    # Original GR-ConvNet
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    
    # Configurable GR-ConvNet with multiple dropouts
    elif network_name == 'grconvnet2':
        from .grconvnet2 import GenerativeResnet
        return GenerativeResnet
    
    # Configurable GR-ConvNet with dropout at the end
    elif network_name == 'grconvnet3':
        from .grconvnet3 import GenerativeResnet
        return GenerativeResnet
    
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet4':
        from .grconvnet4 import GenerativeResnet
        return GenerativeResnet
    
    # GR-Confnet with RFB in bottleneck
    # Inverted GR-ConvNet
    elif network_name == 'grconvnet3_1rfb':
        from .grconvnet3_1rfb import GenerativeResnet
        return GenerativeResnet
     # GR-Confnet with RFB in bottleneck and multi-directional attention fusion
    elif network_name == 'grconvnet3_1mdaf':
        from .grconvnet3_1mdaf import GenerativeResnet
        return GenerativeResnet
    # GR-Confnet with RFB in bottleneck and multi-directional attention fusion
    elif network_name == 'grconvnet3_1rfb_1mdaf_1residual':
        from .grconvnet3_1rfb_1mdaf_1residual import GenerativeResnet
        return GenerativeResnet

    # GR-Confnet with RFB in bottleneck and multi-directional attention fusion
    elif network_name == 'grconvnet3_1rfb_1mdaf_5residual':
        from .grconvnet3_1rfb_1mdaf_5residual import GenerativeResnet
        return GenerativeResnet
    
    # GR-Confnet_3 with MaxPooling
    elif network_name == 'grconvnet3_with_maxpooling':
        from .grconvnet3_with_maxpooling import GenerativeResnet
        return GenerativeResnet

    # GR-Confnet with RFB  in bottleneck and multi-directional attention fusions including concatenation of shallow and deep features in upsampling blocks
    elif network_name == 'grconvnet3_rfb_mdaf_multi_lightweight':
        from .grconvnet3_rfb_mdaf_multi_lightweight import GenerativeResnet
        return GenerativeResnet
    
    # Lightweight CNN with ResBlocks, RFB, MDAF, MaxPooling
    elif network_name == 'lightweight':
        from .lightweight import GenerativeResnet
        return GenerativeResnet

    # Lightweight CNN with ResBlocks, RFB, MDAF, MaxPooling
    elif network_name == 'lightweight_channelsize_128':
        from .lightweight_channelsize_128 import GenerativeResnet
        return GenerativeResnet
    
    # Lightweight CNN with ResBlocks, MDAF, MaxPooling but without RFB
    elif network_name == 'lightweight_without_rfb':
        from .lightweight_without_RFB import GenerativeResnet
        return GenerativeResnet

    # Lightweight CNN with ResBlocks, MDAF but without MaxPooling
    elif network_name == 'lightweight_without_maxpooling':
        from .lightweight_without_MaxPooling import GenerativeResnet
        return GenerativeResnet

    # Lightweight CNN with only 1 MDAF
    elif network_name == 'lightweight_only_1MDAF':
        from .lightweight_only_1MDAF import GenerativeResnet
        return GenerativeResnet
    

    # GRConvNet Basic with only 1 ResBlock
    elif network_name == 'grconvnet2_1resblock':
        from .grconvnet2_1ResBlock import GenerativeResnet
        return GenerativeResnet
    
    # GRConvNet Basic with 2 MaxPooling layers during Downsampling and adjusted upsampling to match dimensions
    elif network_name == 'grconvnet2_maxpooling':
        from .grconvnet2_MaxPooling import GenerativeResnet
        return GenerativeResnet
    
    elif network_name == 'lightweight_with_increasing_filter_size':
        from .lightweight_with_increasing_filter_size import GenerativeResnet
        return GenerativeResnet
    

    
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))



