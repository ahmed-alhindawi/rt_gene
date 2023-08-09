from argparse import ArgumentParser
from functools import partial

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18, GazeEstimationModelVGG, GazeEstimationModelResnet18Uncertainty, GazeEstimationModelVGGUncertainty, \
    GazeEstimationModelResnet50Uncertainty
from datasets.RTGENEDataset import RTGENEWithinSubjectDataset
from utils.CustomLoss import PinballLoss, LaplacianNLL, CharbonnierNLL
from utils.GazeAngleAccuracy import GazeAngleMetric

LOSS_FN = {
    "mse": (torch.nn.MSELoss, 2),
    "mae": (torch.nn.L1Loss, 2),
    "pinball": (PinballLoss, 3),
    "gnll": (torch.nn.GaussianNLLLoss, 3),
    "lnll": (LaplacianNLL, 3),
    "cnll": (partial(CharbonnierNLL, transition=1e-3, slope=1e-2), 3)
}

MODELS = {
    "resnet18": GazeEstimationModelResnet18,
    "resnet18_uncertainty": GazeEstimationModelResnet18Uncertainty,
    "vgg16": GazeEstimationModelVGG,
    "vgg16_uncertainty": GazeEstimationModelVGGUncertainty,
    "resnet50_uncertainty": GazeEstimationModelResnet50Uncertainty,
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
        left_patch, right_patch, headpose_label, y_true = batch

        y_pred = self.forward((left_patch, right_patch, headpose_label))
        angle_out = y_pred[:, :2]
        angle_acc = metric(angle_out, y_true)

        if self.loss_num_out == 2:
            loss = self._criterion(y_pred, y_true)
        elif self.loss_num_out == 3:
            variance = y_pred[:, 2]

            if self.hparams.loss_fn != "pinball":
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
        dataset = RTGENEWithinSubjectDataset(root_path=self.hparams.dataset_path, phase=RTGENEWithinSubjectDataset.TrainingPhase.Training, fraction=0.90)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        dataset = RTGENEWithinSubjectDataset(root_path=self.hparams.dataset_path, phase=RTGENEWithinSubjectDataset.TrainingPhase.Validation, fraction=0.10)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        params_to_update = [param for name, param in self.model.named_parameters()]
        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 180, 270], gamma=0.5)
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
