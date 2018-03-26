import torch
import torch.nn as nn


def downsample_conv(ch_in, ch_out, kernel_size = 3):
    return nn.Sequential(
        nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, stride = 2, padding = (kernel_size - 1) // 2),
        nn.ReLU(inplace = True),
        nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, padding = (kernel_size - 1) // 2),
        nn.ReLU(inplace = True)
    )


def conv(ch_in, ch_out, kernel_size = 3):
    return nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, stride = 2),
            nn.ReLU(inplace = True)
        )


def upconv(ch_in, ch_out):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_in, ch_out, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
        nn.ReLU(inplace = True)
    )


def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNet(nn.Module):


    def __init__(self):
        super(DispNet, self).__init__()

        # Encoder
        conv_channels = [3, 32, 64, 128, 256, 512, 512, 512]
        kernel_size = [7, 5] + [3] * 5
        for idx, (i, j, k) in enumerate(zip(conv_channels[:-1], conv_channels[1:], kernel_size), 1):
            self.__setattr__('conv{}'.format(idx), downsample_conv(i,j,k))

        # Decoder: upconv
        upconv_channels = [conv_channels[-1]] + [512, 512, 256, 128, 64, 32, 16]
        for idx, (i, j) in enumerate(zip(upconv_channels[:-1], upconv_channels[1:]), 1):
            self.__setattr__('upconv{}'.format(idx), upconv(i,j))

        # Decoder: conv
        iconv_ch_in = [1024, 1024, 512, 256, 129, 65, 17]
        iconv_ch_out = [512, 512, 256, 128, 64, 32, 16]
        for idx, (i, j) in enumerate(zip(iconv_ch_in, iconv_ch_out), 1):
            self.__setattr__('iconv{}'.format(idx), conv(i,j))

        self.predict_disp4 = predict_disp(upconv_channels[4])
        self.predict_disp3 = predict_disp(upconv_channels[5])
        self.predict_disp2 = predict_disp(upconv_channels[6])
        self.predict_disp1 = predict_disp(upconv_channels[7])
    

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):

        # Encoding
        output_conv1 = self.conv1(x)
        for i in range(2, 8):
            # e.g. output_conv<X> = self.conv<X>(output_conv<X-1>)
            locals()['output_conv{}'.format(i)] = self.__getattr__('conv{}'.format(i))(locals()['output_conv{}'.format(i - 1)])

        # TODO



if __name__ == '__main__':
    net = DispNet()
    print(net)