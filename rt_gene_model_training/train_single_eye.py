import os
from argparse import ArgumentParser
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnetSingleEye
from datasets import RTGENEWithinSubjectDataset, MPIIWithinSubjectDataset, TrainingPhase
from utils.CustomLoss import LaplacianNLL, CharbonnierNLL
from utils.GazeAngleAccuracy import GazeAngleMetric

LOSS_FN = {
    "mse": (torch.nn.MSELoss, 4),
    "mae": (torch.nn.L1Loss, 4),
    "gnll": (torch.nn.GaussianNLLLoss, 4),
    "lnll": (LaplacianNLL, 4),
    "cnll": (partial(CharbonnierNLL, transition=1e-3, slope=1), 5)
}

MODELS = {
    "resnet18": partial(GazeEstimationModelResnetSingleEye, backbone=GazeEstimationModelResnetSingleEye.ResNetBackbone.Resnet18),
    "resnet50": partial(GazeEstimationModelResnetSingleEye, backbone=GazeEstimationModelResnetSingleEye.ResNetBackbone.Resnet50),
    "wresnet50_2": partial(GazeEstimationModelResnetSingleEye, backbone=GazeEstimationModelResnetSingleEye.ResNetBackbone.WResnet50_2),
    "wresnet101_2": partial(GazeEstimationModelResnetSingleEye, backbone=GazeEstimationModelResnetSingleEye.ResNetBackbone.WResnet101_2),
}

DATASETS = {
    "rt-gene": RTGENEWithinSubjectDataset,
    "MPIIGaze": MPIIWithinSubjectDataset
}


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams):
        super(TrainRTGENE, self).__init__()
        self.save_hyperparameters(hparams)

        loss_fn, num_out = LOSS_FN.get(hparams.loss_fn)

        self.model = MODELS.get(hparams.model_base)(num_out=num_out)
        self.loss_num_out = num_out
        self._criterion = loss_fn()
        self._train_metrics = MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError(),
            GazeAngleMetric()
        ])
        self._val_metrics = MetricCollection([
            MeanSquaredError(),
            MeanAbsoluteError(),
            GazeAngleMetric()
        ])

    def forward(self, inputs):
        return self.model(*inputs)

    def shared_step(self, batch, metric):
        left_patch, right_patch, _, y_true = batch

        left_x = self.forward([left_patch])
        right_x = self.forward([right_patch])

        left_x_angle = torch.concat((torch.arctan2(left_x[..., 0], left_x[..., 1]).view(-1, 1), torch.arctan2(left_x[..., 2], left_x[..., 3]).view(-1, 1)),
                                    dim=1)
        right_x_angle = torch.concat((torch.arctan2(right_x[..., 0], right_x[..., 1]).view(-1, 1), torch.arctan2(right_x[..., 2], right_x[..., 3]).view(-1, 1)),
                                     dim=1)

        if self.loss_num_out == 4:
            # the longer the radius is, the more precise we are, therefore we can use precision = 1/var
            left_x_var = torch.sqrt(left_x[..., 0] ** 2 + left_x[..., 1] ** 2).view(-1, 1)
            right_x_var = torch.sqrt(right_x[..., 0] ** 2 + right_x[..., 1] ** 2).view(-1, 1)

            # combine the results of the left & right outputs under an IVW scheme
            sum_variances = (1.0 / (left_x_var + right_x_var))
            angle_out = ((left_x_angle * left_x_var) + (right_x_angle * right_x_var)) / sum_variances

            angle_acc = metric(angle_out, y_true)
            angle_acc["variance"] = sum_variances.mean()

            loss = self._criterion(angle_out, y_true, sum_variances)
        else:
            angle_out = (left_x_angle + right_x_angle) / 2.0
            angle_acc = metric(angle_out, y_true)
            loss = self._criterion(angle_out, y_true)

        angle_acc["loss"] = loss
        return angle_acc

    def training_step(self, batch, batch_idx):
        results = self.shared_step(batch, self._train_metrics)
        self.log_dict({f"train_{k}": v for k, v in results.items()})
        return results["loss"]

    def validation_step(self, batch, batch_idx):
        results = self.shared_step(batch, self._val_metrics)
        self.log_dict({f"val_{k}": v for k, v in results.items()})
        return results["loss"]

    def train_dataloader(self):
        datasets = []
        for ds_name in self.hparams.dataset:
            ds = DATASETS[ds_name](root_path=os.path.join(self.hparams.dataset_root, ds_name), phase=TrainingPhase.Training, fraction=0.95)
            datasets.append(ds)

        concat_ds = torch.utils.data.ConcatDataset(datasets)  # this will introduce a slight overhead with 1 dataset, but it is cleaner
        return DataLoader(concat_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        datasets = []
        for ds_name in self.hparams.dataset:
            ds = DATASETS[ds_name](root_path=os.path.join(self.hparams.dataset_root, ds_name), phase=TrainingPhase.Validation, fraction=0.05)
            datasets.append(ds)

        concat_ds = torch.utils.data.ConcatDataset(datasets)  # this will introduce a slight overhead with 1 dataset, but it is cleaner
        return DataLoader(concat_ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        params_to_update = [param for name, param in self.model.named_parameters()]
        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate, weight_decay=1e-4)

        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_epochs - self.hparams.warmup_epochs)
        constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, 1.0, self.hparams.warmup_epochs)

        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [constant_scheduler, train_scheduler], [self.hparams.warmup_epochs])

        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--loss_fn', choices=list(LOSS_FN.keys()), default=list(LOSS_FN.keys())[0])
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--learning_rate', type=float, default=0.01)
        parser.add_argument('--model_base', choices=list(MODELS.keys()), default=list(MODELS.keys())[0])
        return parser


if __name__ == "__main__":
    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--dataset_root', type=str, required=True)
    root_parser.add_argument('--dataset', required=True, action="append", default=list())
    root_parser.add_argument('--num_io_workers', default=0, type=int)
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--warmup_epochs', type=int, default=10)
    root_parser.add_argument('--max_epochs', type=int, default=200, help="Maximum number of epochs to perform; the trainer will Exit after.")

    model_parser = TrainRTGENE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()
    hyperparams.dataset = list(set(hyperparams.dataset))  # remove duplicates

    pl.seed_everything(hyperparams.seed)

    wandb_logger = WandbLogger(project='gaze_regression', log_model=True)
    wandb_logger.experiment.config.update(vars(hyperparams))  # extend the wandb logger config with the hyperparameters

    model = TrainRTGENE(hparams=hyperparams)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=False, save_top_k=5)
    lr_callback = LearningRateMonitor()

    try:
        torch.set_float32_matmul_precision("medium")
    finally:
        pass

    # start training
    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         precision="32",
                         val_check_interval=0.2,
                         callbacks=[checkpoint_callback, lr_callback],
                         max_epochs=hyperparams.max_epochs,
                         logger=wandb_logger)

    trainer.fit(model)
