import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torchvision.transforms import transforms

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18, GazeEstimationModelVGG
from rtgene_dataset import RTGENEFileDataset
from utils.GazeAngleAccuracy import GazeAngleMetric
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanSquaredError, MeanAbsoluteError
from utils.CustomLoss import PinballLoss, LaplacianNLL, EnhancedLogLoss
from pytorch_lightning.loggers import WandbLogger

LOSS_FN = {
    "mse": (torch.nn.MSELoss, 2),
    "mae": (torch.nn.L1Loss, 2),
    "pinball": (PinballLoss, 3),
    "gnll": (torch.nn.GaussianNLLLoss, 3),
    "lnll": (LaplacianNLL, 3),
    "ell": (EnhancedLogLoss, 2)
}

MODELS = {
    "vgg16": GazeEstimationModelVGG,
    "resnet18": GazeEstimationModelResnet18,
}


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects):
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
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self.save_hyperparameters(hparams)

    def forward(self, left_patch, right_patch, head_pose):
        return self.model(left_patch, right_patch, head_pose)

    def shared_step(self, batch, metric):
        left_patch, right_patch, headpose_label, y_true = batch

        y_pred = self.forward(left_patch, right_patch, headpose_label)
        angle_acc = metric(y_pred[:, :2], y_true)

        if self.loss_num_out == 2:
            loss = self._criterion(y_pred, y_true)
        elif self.loss_num_out == 3:
            angular_out = y_pred[:, :2]
            confidence = y_pred[:, 2:]

            if self.hparams.loss_fn == "gnll" or self.hparams.loss_fn == "lnll":
                confidence = torch.exp(confidence)

            loss = self._criterion(angular_out, y_true, confidence)
            angle_acc["confidence"] = confidence.mean()
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
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(36, 120), scale=(0.8, 1.2), interpolation=transforms.InterpolationMode.NEAREST, antialias=False),
                                               transforms.RandomGrayscale(p=0.1),
                                               transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5, saturation=0.5),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        dataset = RTGENEFileDataset(root_path=self.hparams.dataset_path, subject_list=self._train_subjects, transform=train_transforms)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        dataset = RTGENEFileDataset(root_path=self.hparams.dataset_path, subject_list=self._validate_subjects, transform=None)
        return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def configure_optimizers(self):
        params_to_update = [param for name, param in self.model.named_parameters()]
        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=90, gamma=0.5)
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--loss_fn', choices=list(LOSS_FN.keys()), default=list(LOSS_FN.keys())[0])
        parser.add_argument('--batch_size', default=128, type=int)
        parser.add_argument('--learning_rate', type=float, default=0.0003)
        parser.add_argument('--model_base', choices=list(MODELS.keys()), default=list(MODELS.keys())[0])
        return parser


if __name__ == "__main__":
    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--dataset_path', type=str, required=True)
    root_parser.add_argument('--num_io_workers', default=8, type=int)
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    root_parser.add_argument('--max_epochs', type=int, default=20, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.set_defaults(k_fold_validation=False)

    model_parser = TrainRTGENE.add_model_specific_args(root_parser)
    hyperparams = model_parser.parse_args()

    pl.seed_everything(hyperparams.seed)

    train_subjects = []
    valid_subjects = []
    test_subjects = []
    if hyperparams.k_fold_validation:
        train_subjects.append([1, 2, 8, 10, 3, 4, 7, 9])
        train_subjects.append([1, 2, 8, 10, 5, 6, 11, 12, 13])
        train_subjects.append([3, 4, 7, 9, 5, 6, 11, 12, 13])
        # validation set is always subjects 14, 15 and 16
        valid_subjects.append([0, 14, 15, 16])
        valid_subjects.append([0, 14, 15, 16])
        valid_subjects.append([0, 14, 15, 16])
        # test subjects
        test_subjects.append([5, 6, 11, 12, 13])
        test_subjects.append([3, 4, 7, 9])
        test_subjects.append([1, 2, 8, 10])
    else:
        train_subjects.append([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        valid_subjects.append([0])  # Note that this is a hack and should not be used to get results for papers
        test_subjects.append([0])

    wandb_logger = WandbLogger(project='gaze_regression')
    wandb_logger.experiment.config.update(vars(hyperparams))  # extend the wandb logger config with the hyperparameters

    for fold, (train_s, valid_s, _) in enumerate(zip(train_subjects, valid_subjects, test_subjects)):
        model = TrainRTGENE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s)
        checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', verbose=False, save_top_k=5)
        lr_callback = LearningRateMonitor()

        # start training
        trainer = pl.Trainer(accelerator="gpu",
                             devices="auto",
                             log_every_n_steps=10,
                             precision=32,
                             callbacks=[checkpoint_callback, lr_callback],
                             min_epochs=hyperparams.min_epochs,
                             max_epochs=hyperparams.max_epochs,
                             logger=wandb_logger)
        trainer.fit(model)
