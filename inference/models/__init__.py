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
    elif network_name == 'grconvnet3_rfb':
        from .grconvnet3_rfb import GenerativeResnet
        return GenerativeResnet
     # GR-Confnet with RFB in bottleneck and multi-directional attention fusion
    elif network_name == 'grconvnet3_rfb_mdaf_single':
        from .grconvnet3_rfb_mdaf_single import GenerativeResnet
        return GenerativeResnet
        # GR-Confnet with RFB  in bottleneck and multi-directional attention fusions including concatenation of shallow and deep features in upsampling blocks
    elif network_name == 'grconvnet3_rfb_mdaf_multi_lightweight':
        from .grconvnet3_rfb_mdaf_multi_lightweight import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))



