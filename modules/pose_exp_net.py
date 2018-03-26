import torch
import torch.nn as nn


def conv(ch_in, ch_out, kernel_size = 3):
    return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, stride = 2),
            nn.ReLU(inplace = True)
        )


def transposed_conv(ch_in, ch_out, kernel_size = 3):
    return nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, stride = 2),
            nn.ReLU(inplace = True)
        )


class PoseExpNet(nn.Module):


    def __init__(self):
        super(PoseExpNet, self).__init__()

        # Encoder
        conv_channels = [3*9, 16, 32, 64, 128, 256, 256, 256]
        kernel_size = [7, 5] + [3] * 5
        for idx, (i, j, k) in enumerate(zip(conv_channels[:-1], conv_channels[1:], kernel_size), 1):
            self.__setattr__('conv{}'.format(idx), conv(i,j,k))

        # Decoder: Transposed Conv
        transposed_conv_channels = [512, 512, 256, 128, 64, 32, 16]
        for i in range(len(transposed_conv_channels) - 1):
            self.__setattr__('transposed_conv{}'.format(i+1), transposed_conv(transposed_conv_channels[i], transposed_conv_channels[i + 1]))

        # Decoder: Conv After Concated with Skip Connections
        for i in range(2):
            pass


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):

        # Encoding
        for i in range(2):
            locals()['output_conv{}'.format(i)] = self.__getattr__('conv{}'.format(i))

        # Decoding: Transposed Conv + Merge Skip Connections + Conv
        # TODO


if __name__ == '__main__':
    net = PoseExpNet()
    for i in net.modules():
        print(i)
    