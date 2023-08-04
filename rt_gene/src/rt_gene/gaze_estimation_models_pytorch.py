import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class GazeEstimationAbstractModel(nn.Module):

    def __init__(self):
        super(GazeEstimationAbstractModel, self).__init__()

    @staticmethod
    def _create_fc_layers_uncertainty(in_features, out_features):
        x_l = nn.Sequential(
            nn.LayerNorm(in_features + 2),
            nn.GELU(),
            nn.Linear(in_features + 2, 1024),
            nn.GroupNorm(8, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.GroupNorm(8, 256),
            nn.ELU(),
            nn.Linear(256, out_features)
        )
        x_r = nn.Sequential(
            nn.LayerNorm(in_features + 2),
            nn.GELU(),
            nn.Linear(in_features + 2, 1024),
            nn.GroupNorm(8, 1024),
            nn.ELU(),
            nn.Linear(1024, 256),
            nn.GroupNorm(8, 256),
            nn.ELU(),
            nn.Linear(256, out_features)
        )

        return x_r, x_l

    @staticmethod
    def _create_fc_layers(in_features, out_features):
        x_l = nn.Sequential(
            nn.GroupNorm(8, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, 1024),
        )
        x_r = nn.Sequential(
            nn.GroupNorm(8, in_features),
            nn.LeakyReLU(),
            nn.Linear(in_features, 1024),
        )

        concat = nn.Sequential(
            nn.GroupNorm(8, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 512)
        )

        fc = nn.Sequential(
            nn.GroupNorm(1, 514),
            nn.LeakyReLU(),
            nn.Linear(514, 256),
            nn.Tanh(),
            nn.Linear(256, out_features)
        )

        return x_l, x_r, concat, fc

    def forward(self, left_eye, right_eye, headpose):
        left_x = self.left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = self.xr(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = self.concat(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc_output = self.fc(eyes_headpose)

        return fc_output


class GazeEstimationModelResnet18(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelResnet18, self).__init__()
        left_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        right_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            left_model.conv1,
            left_model.bn1,
            left_model.relu,
            left_model.maxpool,
            left_model.layer1,
            left_model.layer2,
            left_model.layer3,
            left_model.layer4,
            left_model.avgpool
        )

        self.right_features = nn.Sequential(
            right_model.conv1,
            right_model.bn1,
            right_model.relu,
            right_model.maxpool,
            right_model.layer1,
            right_model.layer2,
            right_model.layer3,
            right_model.layer4,
            right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=left_model.fc.in_features, out_features=num_out)


class GazeEstimationModelResnet18Uncertainty(GazeEstimationAbstractModel):

    def __init__(self, num_out=3):
        super(GazeEstimationModelResnet18Uncertainty, self).__init__()
        left_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        right_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # remove the last ConvBRelu layer
        self.left_features = nn.Sequential(
            left_model.conv1,
            left_model.bn1,
            left_model.relu,
            left_model.maxpool,
            left_model.layer1,
            left_model.layer2,
            left_model.layer3,
            left_model.layer4,
            left_model.avgpool
        )

        self.right_features = nn.Sequential(
            right_model.conv1,
            right_model.bn1,
            right_model.relu,
            right_model.maxpool,
            right_model.layer1,
            right_model.layer2,
            right_model.layer3,
            right_model.layer4,
            right_model.avgpool
        )

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr = GazeEstimationAbstractModel._create_fc_layers_uncertainty(in_features=left_model.fc.in_features, out_features=num_out)

    def forward(self, left_patch, right_patch, headpose):
        left_x = self.left_features(left_patch)
        left_x = torch.flatten(left_x, 1)
        left_x = torch.cat((left_x, headpose), dim=1)
        left_x = self.xl(left_x)

        right_x = self.right_features(right_patch)
        right_x = torch.flatten(right_x, 1)
        right_x = torch.cat((right_x, headpose), dim=1)
        right_x = self.xl(right_x)

        left_x[..., 2] = torch.exp(left_x[..., 2])
        right_x[..., 2] = torch.exp(right_x[..., 2])

        # combine the results of the left & right outputs under an IVW scheme
        sum_variances = (1.0/(1.0 / left_x[..., 2] + right_x[..., 2])).view(-1, 1)
        estimate = ((left_x[..., :2] / left_x[..., 2].view(-1, 1)) + (right_x[..., :2] / right_x[..., 2].view(-1, 1))) * sum_variances
        estimate = torch.concat((estimate, torch.log(sum_variances)), dim=-1)
        return estimate


class GazeEstimationModelPreactResnet(GazeEstimationAbstractModel):
    class PreactResnet(nn.Module):
        class BasicBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride):
                super().__init__()

                self.bn1 = nn.BatchNorm2d(in_channels)
                self.conv1 = nn.Conv2d(in_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=stride,
                                       padding=1,
                                       bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels,
                                       out_channels,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1,
                                       bias=False)

                self.shortcut = nn.Sequential()
                if in_channels != out_channels:
                    self.shortcut.add_module(
                        'conv',
                        nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=1,
                                  stride=stride,
                                  padding=0,
                                  bias=False))

            def forward(self, x):
                x = F.relu(self.bn1(x), inplace=True)
                y = self.conv1(x)
                y = F.relu(self.bn2(y), inplace=True)
                y = self.conv2(y)
                y += self.shortcut(x)
                return y

        def __init__(self, depth=30, base_channels=16, input_shape=(1, 3, 224, 224)):
            super().__init__()

            n_blocks_per_stage = (depth - 2) // 6
            n_channels = [base_channels, base_channels * 2, base_channels * 4]

            self.conv = nn.Conv2d(input_shape[1],
                                  n_channels[0],
                                  kernel_size=(3, 3),
                                  stride=1,
                                  padding=1,
                                  bias=False)

            self.stage1 = self._make_stage(n_channels[0],
                                           n_channels[0],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=1)
            self.stage2 = self._make_stage(n_channels[0],
                                           n_channels[1],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.stage3 = self._make_stage(n_channels[1],
                                           n_channels[2],
                                           n_blocks_per_stage,
                                           GazeEstimationModelPreactResnet.PreactResnet.BasicBlock,
                                           stride=2)
            self.bn = nn.BatchNorm2d(n_channels[2])

            self._init_weights(self.modules())

        @staticmethod
        def _init_weights(module):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

        @staticmethod
        def _make_stage(in_channels, out_channels, n_blocks, block, stride):
            stage = nn.Sequential()
            for index in range(n_blocks):
                block_name = "block{}".format(index + 1)
                if index == 0:
                    stage.add_module(block_name, block(in_channels, out_channels, stride=stride))
                else:
                    stage.add_module(block_name, block(out_channels, out_channels, stride=1))
            return stage

        def forward(self, x):
            x = self.conv(x)
            x = self.stage1(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = F.relu(self.bn(x), inplace=True)
            x = F.adaptive_avg_pool2d(x, output_size=1)
            return x

    def __init__(self, num_out=2):
        super(GazeEstimationModelPreactResnet, self).__init__()
        self.left_features = GazeEstimationModelPreactResnet.PreactResnet()
        self.right_features = GazeEstimationModelPreactResnet.PreactResnet()

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=64, out_features=num_out)


class GazeEstimationModelVGG(GazeEstimationAbstractModel):

    def __init__(self, num_out=2):
        super(GazeEstimationModelVGG, self).__init__()
        left_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        right_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # remove the last ConvBRelu layer
        left_modules = [module for module in left_model.features]
        left_modules.append(left_model.avgpool)
        self.left_features = nn.Sequential(*left_modules)

        right_modules = [module for module in right_model.features]
        right_modules.append(right_model.avgpool)
        self.right_features = nn.Sequential(*right_modules)

        for param in self.left_features.parameters():
            param.requires_grad = True
        for param in self.right_features.parameters():
            param.requires_grad = True

        self.xl, self.xr, self.concat, self.fc = GazeEstimationAbstractModel._create_fc_layers(in_features=left_model.classifier[0].in_features, out_features=num_out)