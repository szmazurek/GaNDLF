import torch
import torch.nn as nn
import torch.nn.functional as F


class in_conv(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        Conv,
        Dropout,
        InstanceNorm,
        kernel_size=3,
        dropout_p=0.3,
        leakiness=1e-2,
        conv_bias=True,
        inst_norm_affine=True,
        res=False,
        lrelu_inplace=True,
    ):
        """[The initial convolution to enter the network, kind of like encode]

        [This function will create the input convolution]

        Arguments:
            input_channels {[int]} -- [the input number of channels, in our case
                                       the number of modalities]
            output_channels {[int]} -- [the output number of channels, will det-
                                        -ermine the upcoming channels]

        Keyword Arguments:
            kernel_size {number} -- [size of filter] (default: {3})
            dropout_p {number} -- [dropout probability] (default: {0.3})
            leakiness {number} -- [the negative leakiness] (default: {1e-2})
            conv_bias {bool} -- [to use the bias in filters] (default: {True})
            inst_norm_affine {bool} -- [affine use in norm] (default: {True})
            res {bool} -- [to use residual connections] (default: {False})
            lrelu_inplace {bool} -- [To update conv outputs with lrelu outputs]
                                    (default: {True})
        """
        nn.Module.__init__(self)
        self.residual = res
        self.dropout_p = dropout_p
        self.conv_bias = conv_bias
        self.leakiness = leakiness
        self.inst_norm_affine = inst_norm_affine
        self.lrelu_inplace = lrelu_inplace
        self.dropout = Dropout(dropout_p)
        self.in_0 = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.in_1 = InstanceNorm(
            output_channels, affine=self.inst_norm_affine, track_running_stats=True
        )
        self.conv0 = Conv(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )
        self.conv1 = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )
        self.conv2 = Conv(
            output_channels,
            output_channels,
            kernel_size=3,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=self.conv_bias,
        )

    def forward(self, x):
        """The forward function for initial convolution

        [input --> conv0 --> | --> in --> lrelu --> conv1 --> dropout --> in -|
                             |                                                |
                  output <-- + <-------------------------- conv2 <-- lrelu <--|]

        Arguments:
            x {[Tensor]} -- [Takes in a type of torch Tensor]

        Returns:
            [Tensor] -- [Returns a torch Tensor]
        """
        x = self.conv0(x)
        if self.residual == True:
            skip = x
        x = F.leaky_relu(
            self.in_0(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
        )
        x = self.conv1(x)
        if self.dropout_p is not None and self.dropout_p > 0:
            x = self.dropout(x)
        x = F.leaky_relu(
            self.in_1(x), negative_slope=self.leakiness, inplace=self.lrelu_inplace
        )
        x = self.conv2(x)
        if self.residual == True:
            x = x + skip
        # print(x.shape)
        return x
