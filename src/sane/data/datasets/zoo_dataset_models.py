from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import ResNet as ResNetBase

import logging

from sane.utils.model_utils import get_activation_layer, initialize_weights


class MLP(nn.Module):
    def __init__(
        self,
        i_dim=14,
        h_dim=[30, 15],
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
        use_bias=True,
    ):
        super().__init__()
        self.use_bias = use_bias
        # init module list
        self.module_list = nn.ModuleList()

        # get hidden layer's list
        # wrap h_dim in list of it's not already
        if not isinstance(h_dim, list):
            h_dim = [h_dim]

        # add i_dim to h_dim
        h_dim.insert(0, i_dim)

        # get if bias should be used or not
        for k in range(len(h_dim) - 1):
            # add linear layer
            self.module_list.append(
                nn.Linear(h_dim[k], h_dim[k + 1], bias=self.use_bias)
            )
            # add nonlinearity
            self.module_list.append(get_activation_layer(nlin))
            if dropout > 0:
                self.module_list.append(nn.Dropout(dropout))
        # init output layer
        self.module_list.append(nn.Linear(h_dim[-1], o_dim, bias=self.use_bias))
        # normalize outputs between 0 and 1
        # self.module_list.append(nn.Sigmoid())

        # initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            logging.debug(f"layer {layer}")
            logging.debug(f"input shape:: {x.shape}")
            x = layer(x)
            logging.debug(f"output shape:: {x.shape}")
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            activations.append(x)
        return x, activations


class CNN(nn.Module):
    def __init__(
        self,
        channels_in,
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        # apply functional dropout, don't add to module list
        self.dropout = dropout
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add linear layer 2
        self.module_list.append(nn.Linear(20, o_dim))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for idx, layer in enumerate(self.module_list):
            x = layer(x)
            if self.dropout > 0 and idx in [2, 5, 10]:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for idx, layer in enumerate(self.module_list):
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
            if self.dropout > 0 and idx in [2, 5, 10]:
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x, activations


class CNN2(nn.Module):
    def __init__(
        self,
        channels_in,
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 2
        self.module_list.append(nn.Conv2d(6, 9, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## compose layer 3
        self.module_list.append(nn.Conv2d(9, 6, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 6, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, o_dim))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if isinstance(layer, nn.Tanh):
                activations.append(x)
        return x, activations


class CNN3(nn.Module):
    def __init__(
        self,
        channels_in,
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 32x32 image size
        ## chn_in * 32 * 32
        ## compose layer 0
        self.module_list.append(nn.Conv2d(channels_in, 16, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 16 * 15 * 15
        ## compose layer 1
        self.module_list.append(nn.Conv2d(16, 32, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 32 * 7 * 7 // 32 * 6 * 6
        ## compose layer 2
        self.module_list.append(nn.Conv2d(32, 15, 3))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## 15 * 2 * 2
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(15 * 2 * 2, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if True:  # dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, o_dim))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def forward(self, x):
        # forward prop through module_list
        for layer in self.module_list:
            x = layer(x)
        return x

    def forward_activations(self, x):
        # forward prop through module_list
        activations = []
        for layer in self.module_list:
            x = layer(x)
            if (
                isinstance(layer, nn.Tanh)
                or isinstance(layer, nn.Sigmoid)
                or isinstance(layer, nn.ReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.LeakyReLU)
                or isinstance(layer, nn.SiLU)
                or isinstance(layer, nn.GELU)
            ):
                activations.append(x)
        return x, activations


class BasicBlockCustom(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, nlin="relu"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = get_activation_layer(nlin)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = get_activation_layer(nlin)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act2(out)
        return out


class ResNet18Slim(nn.Module):
    def __init__(self, channels_in=3, o_dim=10, nlin="relu", dropout=0.0, init_type="kaiming_uniform", width_mult=0.5):
        super().__init__()
        c1 = max(1, int(64 * width_mult))
        c2 = max(1, int(128 * width_mult))
        c3 = max(1, int(256 * width_mult))
        c4 = max(1, int(512 * width_mult))
        self.width_mult = width_mult
        self.conv1 = nn.Conv2d(channels_in, c1, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act1 = get_activation_layer(nlin)
        self.layer1 = self._make_layer(c1, c1, 2, stride=1, nlin=nlin)
        self.layer2 = self._make_layer(c1, c2, 2, stride=2, nlin=nlin)
        self.layer3 = self._make_layer(c2, c3, 2, stride=2, nlin=nlin)
        self.layer4 = self._make_layer(c3, c4, 2, stride=2, nlin=nlin)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(c4, o_dim)
        if init_type is not None:
            initialize_weights(self, init_type)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, nlin):
        layers = [BasicBlockCustom(in_channels, out_channels, stride, nlin)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlockCustom(out_channels, out_channels, 1, nlin))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def forward_activations(self, x):
        activations = []
        x = self.act1(self.bn1(self.conv1(x)))
        activations.append(x.clone())
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, activations


class ResCNN(nn.Module):
    """
    extension of the CNN class.
    Added residual connections via max-pooling and 1d convs to the conv layers
    Added a function to load state-dicts from the cnns without res cons
    """

    def __init__(
        self,
        channels_in,
        o_dim=10,
        nlin="leakyrelu",
        dropout=0.2,
        init_type="uniform",
    ):
        super().__init__()

        if dropout > 0.0:
            raise NotImplementedError(
                "dropout is not yet impemented for the residual connections.|"
            )
        # init module list
        self.module_list = nn.ModuleList()
        ### ASSUMES 28x28 image size
        # input shape bx1x28x28

        ## compose layer 1
        self.module_list.append(nn.Conv2d(channels_in, 8, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        # apply dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 8, 12, 12]
        ## residual connection stack 1
        self.res1_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res1_conv = nn.Conv2d(
            in_channels=channels_in, out_channels=8, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 2
        self.module_list.append(nn.Conv2d(8, 6, 5))
        self.module_list.append(nn.MaxPool2d(2, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## output [15, 6, 4, 4]
        self.res2_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=0)
        self.res2_conv = nn.Conv2d(
            in_channels=8, out_channels=6, kernel_size=1, stride=1, padding=0
        )

        ## compose layer 3
        self.module_list.append(nn.Conv2d(6, 4, 2))
        self.module_list.append(get_activation_layer(nlin))
        ## output [15, 4, 3, 3]
        self.res3_pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.res3_conv = nn.Conv2d(
            in_channels=6, out_channels=4, kernel_size=1, stride=1, padding=0
        )

        ## add flatten layer
        self.module_list.append(nn.Flatten())
        ## add linear layer 1
        self.module_list.append(nn.Linear(3 * 3 * 4, 20))
        self.module_list.append(get_activation_layer(nlin))
        ## add dropout
        if dropout > 0:
            self.module_list.append(nn.Dropout(dropout))
        ## add linear layer 1
        self.module_list.append(nn.Linear(20, o_dim))

        ### initialize weights with se methods
        initialize_weights(self, init_type)

    def load_weights_from_cnn_checkpoint(self, check):
        """
        takes weights and biases from CNN architecture without residual connections
        assumes this particular data structure, will not fail gracefully otherwise
        """
        self.module_list[0].weight.data = check["module_list.0.weight"]
        self.module_list[0].bias.data = check["module_list.0.bias"]
        self.module_list[3].weight.data = check["module_list.3.weight"]
        self.module_list[3].bias.data = check["module_list.3.bias"]
        self.module_list[6].weight.data = check["module_list.6.weight"]
        self.module_list[6].bias.data = check["module_list.6.bias"]
        self.module_list[9].weight.data = check["module_list.9.weight"]
        self.module_list[9].bias.data = check["module_list.9.bias"]
        self.module_list[11].weight.data = check["module_list.11.weight"]
        self.module_list[11].bias.data = check["module_list.11.bias"]

    def forward(self, x):
        # layer 1
        x_ = x.clone()
        for idx in range(0, 3):
            x_ = self.module_list[idx](x_)
        x = self.res1_pool(x)
        x = self.res1_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 2
        x_ = x.clone()
        for idx in range(3, 6):
            x_ = self.module_list[idx](x_)
        x = self.res2_pool(x)
        x = self.res2_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # layer 3
        x_ = x.clone()
        for idx in range(6, 8):
            x_ = self.module_list[idx](x_)
        x = self.res3_pool(x)
        x = self.res3_conv(x)
        assert x.shape == x_.shape
        x = x + x_

        # flatten
        # fc1
        # fc2
        for m in self.module_list[8:]:
            x = m(x)

        return x


class CNN_more_layers(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, 8, 5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(8, 4, 5, bias=False),
            nn.Tanh(),
            nn.Conv2d(4, 6, 4, bias=False),
            nn.Tanh(),
            nn.Conv2d(6, 4, 3, bias=False),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(36, 18),
            nn.Tanh(),
            nn.Linear(18, 10),
        )

        initialize_weights(self, init_type)

    def forward(self, x):
        return self.layers(x)


class CNN_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5, bias=False)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 6, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(6, 4, 2, bias=False)
        self.act3 = nn.Tanh()

        self.identity = nn.Conv2d(8, 4, 1, bias=False)

        self.flatten = nn.Flatten()

        self.fc4 = nn.Linear(36, 20, bias=False)
        self.act4 = nn.Tanh()

        self.fc5 = nn.Linear(20, 10)

        initialize_weights(self, init_type)

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x_identity = self.identity(x)
        logging.debug("x_identity", x_identity.shape)

        x = self.act2(self.pool2(self.conv2(x)))
        logging.debug(x.shape)

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, ::4, ::4]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act4(self.fc4(x))
        logging.debug(x.shape)

        x = self.fc5(x)
        logging.debug(x.shape)

        return x


class CNN_more_layers_residual(nn.Module):
    def __init__(self, init_type, channels_in=1):
        super().__init__()

        self.conv1 = nn.Conv2d(channels_in, 8, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(8, 4, 5, bias=False)
        self.act2 = nn.Tanh()

        self.conv3 = nn.Conv2d(4, 6, 4, bias=False)
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(6, 4, 3, bias=False)
        self.act4 = nn.Tanh()

        self.flatten = nn.Flatten()

        self.fc5 = nn.Linear(36, 18)
        self.act5 = nn.Tanh()

        self.fc6 = nn.Linear(18, 10)

        initialize_weights(self, init_type)

    def forward(self, x):
        x = self.conv1(x)
        logging.debug(x.shape)

        x = self.act1(self.pool1(x))
        logging.debug(x.shape)

        x = self.act2(self.conv2(x))
        logging.debug(x.shape)

        x_identity = x.clone()

        x = self.act3(self.conv3(x))
        logging.debug(x.shape)

        x = self.act4(self.conv4(x))
        logging.debug(x.shape)

        x = x + x_identity[:, :, 1:-1, 1:-1][:, :, ::2, ::2]

        logging.debug("skip connection applied")

        x = self.flatten(x)

        x = self.act5(self.fc5(x))
        logging.debug(x.shape)

        x = self.fc6(x)
        logging.debug(x.shape)

        return x


class ResNet(ResNetBase):
    """
    Wrapper for pytroch ResNet class to get access to forward pass.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_activations(self, x):
        """forward pass and return activations"""
        # See note [TorchScript super()]
        activations = []

        # input block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        activations.append(x.clone())

        # residual blocks
        x = self.layer1(x)
        activations.append(x.clone())
        x = self.layer2(x)
        activations.append(x.clone())
        x = self.layer3(x)
        activations.append(x.clone())
        x = self.layer4(x)
        activations.append(x.clone())

        # output block
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, activations


class ResNetBase(ResNet):
    """
    ResNet base class, defaults to ResNet 18, implements init and weight init
    """

    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(block=block, layers=layers, num_classes=out_dim)
        # adpat first layer to fit dimensions
        self.conv1 = nn.Conv2d(
            channels_in,
            64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )
        self.maxpool = nn.Identity()

        if init_type is not None:
            initialize_weights(self, init_type)


class ResNet18(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[2, 2, 2, 2],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet34(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=BasicBlock,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet50(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 6, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet101(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 4, 23, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=block,
            layers=layers,
        )


class ResNet152(ResNetBase):
    def __init__(
        self,
        channels_in=3,
        out_dim=10,
        nlin="relu",  # doesn't yet do anything
        dropout=0.2,  # doesn't yet do anything
        init_type="kaiming_uniform",
        block=Bottleneck,
        layers=[3, 8, 36, 3],
    ):
        # call init from parent class
        super().__init__(
            channels_in=channels_in,
            out_dim=out_dim,
            nlin=nlin,
            dropout=dropout,
            init_type=init_type,
            block=Bottleneck,
            layers=layers,
        )


class MiniAlexNet(nn.Module):
    def __init__(self, channels_in=3, num_classes=10, init_type="kaiming_uniform"):
        super(MiniAlexNet, self).__init__()
        # First Convolutional Layer
        self.conv1 = nn.Conv2d(
            channels_in, 96, kernel_size=3, stride=1
        )  # Use padding to keep size 32x32
        self.maxpool1 = nn.MaxPool2d(
            kernel_size=3, stride=3
        )  # Reduce size from 32x32 to 10x10
        self.batchnorm1 = nn.BatchNorm2d(96)

        # Second Convolutional Layer
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)  # Keep size 10x10
        self.maxpool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Reduce size from 10x10 to 5x5
        self.batchnorm2 = nn.BatchNorm2d(256)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 384)  # Adjusted for 5x5 feature map size
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, num_classes)

        if init_type is not None:
            initialize_weights(self, init_type)

    def forward(self, x):
        # Apply Convolutional Layers
        x = self.batchnorm1(F.relu(self.conv1(x)))
        x = self.maxpool1(x)
        x = self.batchnorm2(F.relu(self.conv2(x)))
        x = self.maxpool2(x)

        x = x.view(-1, 256 * 4 * 4)

        # Apply Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def get_model(config:Dict) -> nn.Module:
    model_type = config["model::type"]
    match model_type:
        case "MLP":
            i_dim = config["model::i_dim"]
            h_dim = config["model::h_dim"]
            o_dim = config["model::o_dim"]
            nlin = config["model::nlin"]
            dropout = config["model::dropout"]
            init_type = config["model::init_type"]
            use_bias = config["model::use_bias"]

            return MLP(i_dim, h_dim, o_dim, nlin, dropout, init_type, use_bias)

        case "CNN":
            channels_in=config["model::channels_in"]
            o_dim=config.get("model::o_dim", 10)
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN(channels_in, o_dim, nlin, dropout, init_type)

        case "CNN2":
            channels_in=config["model::channels_in"]
            o_dim=config.get("model::o_dim", 10)
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN2(channels_in, o_dim, nlin, dropout, init_type)

        case "CNN3":
            channels_in=config["model::channels_in"]
            o_dim=config.get("model::o_dim", 10)
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return CNN3(channels_in, o_dim, nlin, dropout, init_type)

        case "ResCNN":
            channels_in=config["model::channels_in"]
            o_dim=config.get("model::o_dim", 10)
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResCNN(channels_in, o_dim, nlin, dropout, init_type)

        case "CNN_more_layers":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_more_layers(init_type, channels_in)

        case "CNN_residual":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_residual(init_type, channels_in)

        case "CNN_more_layers_residual":
            init_type=config["model::init_type"]
            channels_in=config["model::channels_in"]

            return CNN_more_layers_residual(init_type, channels_in)

        case "Resnet18":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet18(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet34":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet34(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet50":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet50(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet101":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet101(channels_in, out_dim, nlin, dropout, init_type)

        case "Resnet152":
            channels_in=config["model::channels_in"]
            out_dim=config["model::o_dim"]
            nlin=config["model::nlin"]
            dropout=config["model::dropout"]
            init_type=config["model::init_type"]

            return ResNet152(channels_in, out_dim, nlin, dropout, init_type)

        case "ResNet18Slim":
            channels_in=config["model::channels_in"]
            o_dim=config["model::o_dim"]
            nlin=config.get("model::nlin", "relu")
            dropout=config.get("model::dropout", 0.0)
            init_type=config.get("model::init_type", "kaiming_uniform")
            width_mult=config.get("model::width_mult", 0.5)

            return ResNet18Slim(channels_in, o_dim, nlin, dropout, init_type, width_mult)

        case "MiniAlexNet":
            channels_in=config["model::channels_in"]
            num_classes=config["model::o_dim"]
            init_type=config["model::init_type"]

            return MiniAlexNet(channels_in, num_classes, init_type)

        case "efficientnet_v2_s":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.efficientnet_v2_s(num_classes, dropout)

        case "efficientnet_v2_m":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.efficientnet_v2_m(num_classes, dropout)

        case "densenet121":
            num_classes=config["model::o_dim"]

            return torchvision.models.densenet121(num_classes)

        case "densenet161":
            num_classes=config["model::o_dim"]

            return torchvision.models.densenet161(num_classes)

        case "vit_b_16":
            num_classes=config["model::o_dim"]
            dropout=config["model::dropout"]

            return torchvision.models.vit_b_16(num_classes, dropout)

        case _:
            raise NotImplementedError("error: model type unkown")