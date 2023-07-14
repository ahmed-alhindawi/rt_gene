import torch
from torchmetrics import Metric


class GazeAngleMetric(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("angle", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, y_pred, y_true):
        batch_y_pred = y_pred.detach()
        batch_y_true = y_true.detach()

        pred_x = -1 * torch.cos(batch_y_pred[:, 0]) * torch.sin(batch_y_pred[:, 1])
        pred_y = -1 * torch.sin(batch_y_pred[:, 0])
        pred_z = -1 * torch.cos(batch_y_pred[:, 0]) * torch.cos(batch_y_pred[:, 1])
        pred_l = torch.sqrt(pred_x ** 2 + pred_y ** 2 + pred_z ** 2)

        true_x = -1 * torch.cos(batch_y_true[:, 0]) * torch.sin(batch_y_true[:, 1])
        true_y = -1 * torch.sin(batch_y_true[:, 0])
        true_z = -1 * torch.cos(batch_y_true[:, 0]) * torch.cos(batch_y_true[:, 1])
        true_l = torch.sqrt(true_x ** 2 + true_y ** 2 + true_z ** 2)

        angle = ((pred_x * true_x + pred_y * true_y + pred_z * true_z) / (pred_l * true_l)).sum()
        self.angle += angle
        self.total += y_pred.shape[0]

    def compute(self):
        return torch.rad2deg(torch.arccos(self.angle / self.total))
