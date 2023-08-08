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

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18SingleEye
from datasets import RTGENEWithinSubjectDataset, MPIIWithinSubjectDataset, TrainingPhase
from utils.CustomLoss import LaplacianNLL, CharbonnierNLL
from utils.GazeAngleAccuracy import GazeAngleMetric

LOSS_FN = {
    "mse": (torch.nn.MSELoss, 4),
    "mae": (torch.nn.L1Loss, 4),
    "gnll": (torch.nn.GaussianNLLLoss, 5),
    "lnll": (LaplacianNLL, 5),
    "cnll": (partial(CharbonnierNLL, transition=1e-3, slope=1), 5)
}

MODELS = {
    "resnet18": GazeEstimationModelResnet18SingleEye
}


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams):
        super(TrainRTGENE, self).__init__()
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
        self.save_hyperparameters(hparams)

    def forward(self, inputs):
        return self.model(*inputs)

    def shared_step(self, batch, metric):
        left_patch, right_patch, _, y_true = batch

        left_x = self.forward([left_patch])
        right_x = self.forward([right_patch])

        left_x_angle = torch.concat((torch.arctan2(left_x[..., 0], left_x[..., 1]).view(-1, 1), torch.arctan2(left_x[..., 2], left_x[..., 3]).view(-1, 1)), dim=1)
        right_x_angle = torch.concat((torch.arctan2(right_x[..., 0], right_x[..., 1]).view(-1, 1), torch.arctan2(right_x[..., 2], right_x[..., 3]).view(-1, 1)), dim=1)

        angle_acc = dict()
        if self.loss_num_out == 5:
            # take the mean of those two
            left_x_var = torch.exp(left_x[..., 4]).view(-1, 1)
            right_x_var = torch.exp(right_x[..., 4]).view(-1, 1)

            # combine the results of the left & right outputs under an IVW scheme
            sum_variances = (1.0 / (1.0 / left_x_var + right_x_var))
            angle_out = ((left_x_angle / left_x_var) + (right_x_angle / right_x_var)) * sum_variances

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
        rt_gene_ds = RTGENEWithinSubjectDataset(root_path=os.path.join(self.hparams.dataset_path, "rt_gene/"), phase=TrainingPhase.Training, fraction=0.95)
        mpii_ds = MPIIWithinSubjectDataset(root_path=os.path.join(self.hparams.dataset_path, "MPIIGaze/"), phase=TrainingPhase.Training, fraction=0.95)
        ds = torch.utils.data.ConcatDataset([rt_gene_ds, mpii_ds])
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        rt_gene_ds = RTGENEWithinSubjectDataset(root_path=os.path.join(self.hparams.dataset_path, "rt_gene/"), phase=TrainingPhase.Validation, fraction=0.05)
        mpii_ds = MPIIWithinSubjectDataset(root_path=os.path.join(self.hparams.dataset_path, "MPIIGaze/"), phase=TrainingPhase.Validation, fraction=0.05)
        ds = torch.utils.data.ConcatDataset([rt_gene_ds, mpii_ds])
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        params_to_update = [param for name, param in self.model.named_parameters()]
        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate, betas=(0.9, 0.95))

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
    root_parser.add_argument('--dataset_path', type=str, required=True)
    root_parser.add_argument('--num_io_workers', default=0, type=int)
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--warmup_epochs', type=int, default=10)
    root_parser.add_argument('--max_epochs', type=int, default=200, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.add_argument('--tune_lr', action="store_true", dest="tune_lr", default=False)

    model_parser = TrainRTGENE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()

    pl.seed_everything(hyperparams.seed)

    wandb_logger = WandbLogger(project='gaze_regression', log_model=True)
    wandb_logger.experiment.config.update(vars(hyperparams))  # extend the wandb logger config with the hyperparameters

    model = TrainRTGENE(hparams=hyperparams)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=False, save_top_k=5)
    lr_callback = LearningRateMonitor()

    # start training
    trainer = pl.Trainer(accelerator="gpu",
                         devices="auto",
                         precision="16-mixed",
                         val_check_interval=0.2,
                         callbacks=[checkpoint_callback, lr_callback],
                         max_epochs=hyperparams.max_epochs,
                         logger=wandb_logger)
    if hyperparams.tune_lr:
        from pytorch_lightning.tuner import Tuner

        tuner = Tuner(trainer)
        tuner.lr_find(model)

    trainer.fit(model)
