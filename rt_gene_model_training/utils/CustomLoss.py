import torch
from torch import nn
import math


class PinballLoss(nn.Module):

    def __init__(self, reduction="mean"):
        super(PinballLoss, self).__init__()
        self.q1 = 0.45
        self.q9 = 1 - self.q1

        _reduction_strategies = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x
        }
        assert reduction in _reduction_strategies.keys(), "Reduction method unknown, possibilities include 'mean', 'sum' and 'none'"

        self._reduction_strategy = _reduction_strategies.get(reduction)

    def forward(self, output, target, var):
        angle_o = output[:, :2]
        var_o = var.view(-1, 1).expand(var.size(0), 2)

        q_10 = target - (angle_o - var_o)
        q_90 = target - (angle_o + var_o)

        loss_10 = torch.max(self.q1 * q_10, (self.q1 - 1) * q_10)
        loss_90 = torch.max(self.q9 * q_90, (self.q9 - 1) * q_90)

        loss_10 = self._reduction_strategy(loss_10)
        loss_90 = self._reduction_strategy(loss_90)

        return loss_10 + loss_90


class LaplacianNLL(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()

        _reduction_strategies = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x
        }
        assert reduction in _reduction_strategies.keys(), "Reduction method unknown, possibilities include 'mean', 'sum' and 'none'"

        self._reduction_strategy = _reduction_strategies.get(reduction)

    def forward(self, y_pred, y_true, scale):
        log_s = torch.log(2 * scale)
        mae = torch.abs(y_true - y_pred)
        loss = log_s + (mae/scale)

        return self._reduction_strategy(loss)


class WingLoss(nn.Module):

    def __init__(self, reduction="mean"):
        super().__init__()
        self._w = 10.0
        self._e = 2.0

        self._c = self._w * (1.0 - math.log(1.0 + self._w / self._e))
        _reduction_strategies = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x
        }
        assert reduction in _reduction_strategies.keys(), "Reduction method unknown, possibilities include 'mean', 'sum' and 'none'"

        self._reduction_strategy = _reduction_strategies.get(reduction)

    def forward(self, y_pred, y_true):
        x = y_true - y_pred
        absolute_x = torch.abs(x)
        losses = torch.where(self._w > absolute_x,
                             self._w * torch.log(1.0 + absolute_x / self._e),
                             absolute_x - self._c)

        self._reduction_strategy(losses)


class EnhancedLogLoss(nn.Module):

    def __init__(self, alpha=4, reduction="mean"):
        super().__init__()
        self._alpha = alpha
        self._reduction = reduction
        _reduction_strategies = {
            "mean": torch.mean,
            "sum": torch.sum,
            "none": lambda x: x
        }
        assert reduction in _reduction_strategies.keys(), "Reduction method unknown, possibilities include 'mean', 'sum' and 'none'"

        self._reduction_strategy = _reduction_strategies.get(reduction)

    def forward(self, y_pred, y_true):
        x = y_true - y_pred
        absolute_x = torch.abs(x)
        left = (self._alpha + 1) * absolute_x
        right = -self._alpha * torch.log((torch.exp(absolute_x) + torch.exp(-absolute_x)) / 2.0)
        losses = left + right

        self._reduction_strategy(losses)


if __name__ == "__main__":
    t1 = torch.rand((16, 3))
    t2 = t1.clone()[:, :2]
    loss = LaplacianNLL()
    print(loss(t1[:, :2], t2, t1[:, 2:]))
