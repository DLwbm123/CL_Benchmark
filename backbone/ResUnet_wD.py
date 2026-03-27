import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        
        self.conv_skip = nn.Conv2d(input_dim, output_dim, kernel_size=3, bias=False, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(
                input_dim, output_dim, kernel_size=3, bias=False, stride=stride, padding=padding
            )
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, bias=False, padding=1)
        


    def forward(self, x):
        shortcut = self.conv_skip(x)
        x = self.bn1(x)
        out1 = F.relu(x)
        out2 = self.conv1(out1)
        out3 = self.bn2(out2)
        out3 = F.relu(out3)
        out = self.conv2(out3)

        return out1, out3, out + shortcut


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2)

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, inputsize, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.map = []
        self.act = OrderedDict()
        self.ksize = []
        self.in_channel = []
        self.map.append(inputsize)

        self.conv1 = nn.Conv2d(channel, filters[0], kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.ksize.append(3)
        self.in_channel.append(1)
        #self.conv2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, bias=False, padding=1)

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[0], filters[0], filters[1]]
        self.map += [inputsize, inputsize, inputsize // 2]
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[1], filters[1], filters[2]]
        self.map += [inputsize // 2, inputsize // 2, inputsize // 4]

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[2], filters[2], filters[3]]
        self.map += [inputsize // 4, inputsize // 4, inputsize // 8]

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[3] + filters[2], filters[3] + filters[2], filters[2]]
        self.map += [inputsize // 4, inputsize // 4, inputsize // 4]

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[2] + filters[1], filters[2] + filters[1], filters[1]]
        self.map += [inputsize // 2, inputsize // 2, inputsize // 2]

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.ksize += [3, 3, 3]
        self.in_channel += [filters[1] + filters[0], filters[1] + filters[0], filters[0]]
        self.map += [inputsize, inputsize, inputsize]

        self.last = nn.Conv2d(filters[0], 2, 1, 1, bias=False)

    def features(self, x):
        # Encode
        self.act["conv1"] = x.detach()
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        #x1 = self.input_layer(x)
        x1_mid, x1_las, x2 = self.residual_conv_1(x1)
        self.act["conv2"] = x1.detach()
        self.act["conv3"] = x1_mid.detach()
        self.act["conv4"] = x1_las.detach()
        x2_mid, x2_las, x3 = self.residual_conv_2(x2)
        self.act["conv5"] = x2.detach()
        self.act["conv6"] = x2_mid.detach()
        self.act["conv7"] = x2_las.detach()
        # Bridge
        x3_mid, x3_las, x4 = self.bridge(x3)
        self.act["conv8"] = x3.detach()
        self.act["conv9"] = x3_mid.detach()
        self.act["conv10"] = x3_las.detach()
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x4_mid, x4_las, x6 = self.up_residual_conv1(x5)
        self.act["conv11"] = x5.detach()
        self.act["conv12"] = x4_mid.detach()
        self.act["conv13"] = x4_las.detach()

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x5_mid, x5_las, x8 = self.up_residual_conv2(x7)
        self.act["conv14"] = x7.detach()
        self.act["conv15"] = x5_mid.detach()
        self.act["conv16"] = x5_las.detach()

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x6_mid, x6_las, x10 = self.up_residual_conv3(x9)
        self.act["conv17"] = x9.detach()
        self.act["conv18"] = x6_mid.detach()
        self.act["conv19"] = x6_las.detach()

        #output = self.output_layer(x10)

        return x10

    def logits(self, x, k = None):
        x = self.last(x)
        return x

    def forward(self, x, k = None):
        out = self.features(x)
        out = self.logits(out, k)
        return out

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)
    
    def get_params_without_outputlayer(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.named_parameters()):
            if "last" not in pp[0]:
                params.append(pp[1].view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)
    
    def get_grads_without_outputlayer(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.named_parameters()):
            if "last" not in pp[0]:
                if pp[1].grad is None:
                    continue
                grads.append(pp[1].grad.view(-1))
        return torch.cat(grads)
    
    
    
def resunet32_withdict(inputsize, size = 'mid'):
    if size == 'small':
        return ResUnet(inputsize, 1, [16, 32, 64, 128])
    elif size == 'mid':
        return ResUnet(inputsize, 1, [32, 64, 128, 256])
    else:
        return ResUnet(inputsize, 1, [64, 128, 256, 512])


if __name__ == '__main__':
    a = torch.rand(2, 1, 32, 32)
    net = resunet32()
    out = net(a)
    print(out.size())