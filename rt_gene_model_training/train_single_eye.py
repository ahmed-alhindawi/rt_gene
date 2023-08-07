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
    "gnll": (torch.nn.GaussianNLLLoss, 3),
    "lnll": (LaplacianNLL, 3),
    "cnll": (partial(CharbonnierNLL, transition=1e-3, slope=1e-2), 3)
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

        # take the mean of those two
        left_x[..., 2] = torch.exp(left_x[..., 2])
        right_x[..., 2] = torch.exp(right_x[..., 2])

        # combine the results of the left & right outputs under an IVW scheme
        sum_variances = (1.0 / (1.0 / left_x[..., 2] + right_x[..., 2])).view(-1, 1)
        y_pred = ((left_x[..., :2] / left_x[..., 2].view(-1, 1)) + (right_x[..., :2] / right_x[..., 2].view(-1, 1))) * sum_variances
        y_pred = torch.concat((y_pred, torch.log(sum_variances)), dim=-1)

        angle_out = y_pred[:, :2]
        angle_acc = metric(angle_out, y_true)

        if self.loss_num_out == 2:
            loss = self._criterion(angle_out, y_true)
        elif self.loss_num_out == 3:
            variance = y_pred[:, 2]
            variance = torch.exp(variance)
            loss = self._criterion(angle_out, y_true, variance)
            angle_acc["variance"] = variance.mean()
        else:
            raise ValueError(f"Number out isn't right ({self.loss_num_out}) or unknown loss function {self.hparams.loss_fn}")

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
        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate, weight_decay=1e-4)

        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.max_epochs - self.hparams.warmup_epochs)
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: 1 / (10 ** (float(self.hparams.warmup_epochs - current_step))))
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
    root_parser.add_argument('--num_io_workers', default=8, type=int)
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--warmup_epochs', type=int, default=10)
    root_parser.add_argument('--max_epochs', type=int, default=200, help="Maximum number of epochs to perform; the trainer will Exit after.")

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
                         callbacks=[checkpoint_callback, lr_callback],
                         max_epochs=hyperparams.max_epochs,
                         logger=wandb_logger)
    trainer.fit(model)
