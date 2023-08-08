import torch
import torch.nn as nn
from torchvision import models
import timm
from enum import Enum
from torch import Tensor
import torch.nn.functional as F


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

    @staticmethod
    def forward_without_uncertainty(left_eye, right_eye, headpose, left_features, left_linear, right_features, right_linear, linear_headpose, fc):
        left_x = left_features(left_eye)
        left_x = torch.flatten(left_x, 1)
        left_x = left_linear(left_x)

        right_x = right_features(right_eye)
        right_x = torch.flatten(right_x, 1)
        right_x = right_linear(right_x)

        eyes_x = torch.cat((left_x, right_x), dim=1)
        eyes_x = linear_headpose(eyes_x)

        eyes_headpose = torch.cat((eyes_x, headpose), dim=1)

        fc_output = fc(eyes_headpose)

        return fc_output

    @staticmethod
    def forward_with_uncertainty(left_patch, right_patch, headpose, left_features, left_linear, right_features, right_linear):
        left_x = left_features(left_patch)
        left_x = torch.flatten(left_x, 1)
        left_x = torch.cat((left_x, headpose), dim=1)
        left_x = left_linear(left_x)

        right_x = right_features(right_patch)
        right_x = torch.flatten(right_x, 1)
        right_x = torch.cat((right_x, headpose), dim=1)
        right_x = right_linear(right_x)

        left_x[..., 2] = torch.exp(left_x[..., 2])
        right_x[..., 2] = torch.exp(right_x[..., 2])

        # combine the results of the left & right outputs under an IVW scheme
        sum_variances = (1.0 / (1.0 / left_x[..., 2] + right_x[..., 2])).view(-1, 1)
        estimate = ((left_x[..., :2] / left_x[..., 2].view(-1, 1)) + (right_x[..., :2] / right_x[..., 2].view(-1, 1))) * sum_variances
        estimate = torch.concat((estimate, torch.log(sum_variances)), dim=-1)
        return estimate


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

    def forward(self, left, right, headpose):
        return self.forward_without_uncertainty(left, right, headpose, self.left_features, self.xl, self.right_features, self.xr, self.concat, self.fc)


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
        return self.forward_with_uncertainty(left_patch, right_patch, headpose, self.left_features, self.xl, self.right_features, self.xr)


class GazeEstimationModelResnet50Uncertainty(GazeEstimationAbstractModel):

    def __init__(self, num_out=3):
        super(GazeEstimationModelResnet50Uncertainty, self).__init__()
        left_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        right_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

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
        return self.forward_with_uncertainty(left_patch, right_patch, headpose, self.left_features, self.xl, self.right_features, self.xr)


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

    def forward(self, left, right, headpose):
        return self.forward_without_uncertainty(left, right, headpose, self.left_features, self.xl, self.right_features, self.xr, self.concat, self.fc)


class GazeEstimationModelVGGUncertainty(GazeEstimationAbstractModel):

    def __init__(self, num_out=3):
        super(GazeEstimationModelVGGUncertainty, self).__init__()
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

        self.xl, self.xr = GazeEstimationAbstractModel._create_fc_layers_uncertainty(in_features=left_model.classifier[0].in_features, out_features=num_out)

    def forward(self, left, right, headpose):
        return self.forward_with_uncertainty(left, right, headpose, self.left_features, self.xl, self.right_features, self.xr)


class GazeEstimationModelResnet18SingleEye(nn.Module):

    class ResNetBackbone(Enum):
        Resnet18 = "resnet18"
        Resnet34 = "resnet34"
        Resnet50 = "resnet50"
        Resnet101 = "resnet101"

    def __init__(self, backbone=ResNetBackbone.Resnet18, num_out: int = 2):
        super().__init__()
        backbone = timm.create_model(backbone.value, pretrained=True, num_classes=0)

        self.model = nn.Sequential(backbone,
                                   nn.GroupNorm(8, backbone.num_features),
                                   nn.GELU(),
                                   nn.Linear(backbone.num_features, backbone.num_features),
                                   nn.GroupNorm(8, backbone.num_features),
                                   nn.Tanh(),
                                   nn.Linear(backbone.num_features, num_out),
                                   )

    def forward(self, imgs: Tensor) -> Tensor:
        x = self.model(imgs)
        return x
