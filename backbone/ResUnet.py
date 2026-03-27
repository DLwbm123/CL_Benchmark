import torch
import torch.nn as nn
import torch.nn.functional as F
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.bn1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(
                input_dim, output_dim, kernel_size=3, bias=False, stride=stride, padding=padding
            )
        self.bn2 = nn.BatchNorm2d(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, bias=False, padding=1)
        self.conv_skip = nn.Conv2d(input_dim, output_dim, kernel_size=3, bias=False, stride=stride, padding=1)


    def forward(self, x):
        shortcut = self.conv_skip(x)
        x = self.bn1(x)
        out = F.relu(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(F.relu(out))

        return out + shortcut


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2)

    def forward(self, x):
        return self.upsample(x)

class ResUnet(nn.Module):
    def __init__(self, channel, filters=[64, 128, 256, 512]):
        super(ResUnet, self).__init__()

        self.conv1 = nn.Conv2d(channel, filters[0], kernel_size=3, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(filters[0])
        self.conv2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, bias=False, padding=1)

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.last = nn.Conv2d(filters[0], 2, 1, 1, bias=False)

    def features(self, x):
        # Encode
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        #x1 = self.input_layer(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge
        x4 = self.bridge(x3)
        # Decode
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

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
            if pp.grad is None:
                continue
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
    

def resunet32(size = 'mid'):
    if size == 'small':
        return ResUnet(1, [16, 32, 64, 128])
    elif size == 'mid':
        return ResUnet(1, [32, 64, 128, 256])
    else:
        return ResUnet(1, [64, 128, 256, 512])


if __name__ == '__main__':
    a = torch.rand(2, 1, 32, 32)
    net = resunet32()
    out = net(a)
    print(out.size())