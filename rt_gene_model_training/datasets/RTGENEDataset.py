import glob
import os

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.io as tio
import albumentations as albu
from albumentations import pytorch
import cv2
from datasets.TrainingPhase import TrainingPhase


class RTGENEWithinSubjectDataset(data.Dataset):
    def __init__(self, root_path, phase=TrainingPhase.Training, fraction=0.95):
        self._root_path = root_path

        labels = []

        for subject_data in glob.glob(os.path.join(root_path, "s*_glasses/")):
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
                for line in _lines:
                    split = line.split(",")
                    left_img_path = os.path.join(subject_data, "inpainted/left/", "left_{:0=6d}_rgb.png".format(int(split[0])))
                    right_img_path = os.path.join(subject_data, "inpainted/right/", "right_{:0=6d}_rgb.png".format(int(split[0])))
                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        head_phi = float(split[1].strip()[1:])
                        head_theta = float(split[2].strip()[:-1])
                        gaze_phi = float(split[3].strip()[1:])
                        gaze_theta = float(split[4].strip()[:-1])
                        labels.append([left_img_path, right_img_path, head_phi, head_theta, gaze_phi, gaze_theta])
        assert len(labels) > 0, f"No data found in {root_path}"

        match phase:
            case TrainingPhase.Training:
                self._transform = albu.Compose([
                    albu.RandomResizedCrop(36, 60, scale=(0.8, 1.2), always_apply=True),
                    albu.RandomBrightnessContrast(p=0.1),
                    albu.OneOf([
                        albu.GaussianBlur(),
                        albu.ISONoise()
                        ], p=0.1),
                    albu.ColorJitter(p=0.1),
                    albu.Equalize(p=0.1),
                    albu.HueSaturationValue(p=0.1),
                    albu.Normalize(),
                    albu.pytorch.ToTensorV2(),
                ])
                end_num = int(len(labels) * fraction)
                self._data = labels[:end_num]
            case TrainingPhase.Validation:
                self._transform = albu.Compose([
                    albu.Resize(36, 60),
                    albu.Normalize(),
                    albu.pytorch.ToTensorV2(),
                ])
                end_num = int(len(labels) * fraction)
                self._data = labels[-end_num:]
            case TrainingPhase.Testing:
                self._transform = albu.Compose([
                    albu.Resize(36, 60),
                    albu.Normalize(),
                    albu.pytorch.ToTensorV2(),
                ])
                end_num = int(len(labels) * fraction)
                self._data = labels[-end_num:]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        sample = self._data[index]
        ground_truth_headpose = torch.Tensor([sample[2], sample[3]]).to(torch.float32)
        ground_truth_gaze = torch.Tensor([sample[4], sample[5]]).to(torch.float32)

        # Load data and get label
        left_img = cv2.imread(sample[0])
        right_img = cv2.imread(sample[1])

        transformed_lt = self._transform(image=left_img)["image"]
        transformed_rt = self._transform(image=right_img)["image"]

        return transformed_lt, transformed_rt, ground_truth_headpose, ground_truth_gaze


class RTGENECrossSubjectDataset(data.Dataset):

    def __init__(self, root_path, subject_list=None, transform=None):
        self._root_path = root_path
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.NEAREST, antialias=False),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        subject_path = [os.path.join(root_path, "s{:03d}_glasses/".format(i)) for i in subject_list]

        for subject_data in subject_path:
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
                for line in _lines:
                    split = line.split(",")
                    left_img_path = os.path.join(subject_data, "inpainted/left/", "left_{:0=6d}_rgb.png".format(int(split[0])))
                    right_img_path = os.path.join(subject_data, "inpainted/right/", "right_{:0=6d}_rgb.png".format(int(split[0])))
                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        head_phi = float(split[1].strip()[1:])
                        head_theta = float(split[2].strip()[:-1])
                        gaze_phi = float(split[3].strip()[1:])
                        gaze_theta = float(split[4].strip()[:-1])
                        self._subject_labels.append([left_img_path, right_img_path, head_phi, head_theta, gaze_phi, gaze_theta])

        assert len(self._subject_labels) > 0, f"No data found in {root_path}"

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        sample = self._subject_labels[index]
        ground_truth_headpose = torch.Tensor([sample[2], sample[3]]).to(torch.float32)
        ground_truth_gaze = torch.Tensor([sample[4], sample[5]]).to(torch.float32)

        # Load data and get label
        right_img = tio.read_image(os.path.join(self._root_path, sample[1]), tio.ImageReadMode.RGB).to(torch.float32)
        left_img = tio.read_image(os.path.join(self._root_path, sample[0]), tio.ImageReadMode.RGB).to(torch.float32)

        transformed_lt = self._transform(left_img)
        transformed_rt = self._transform(right_img)

        return transformed_lt, transformed_rt, ground_truth_headpose, ground_truth_gaze
