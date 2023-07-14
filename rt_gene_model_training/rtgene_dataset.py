import os

import torch
from torch.utils import data
from torchvision import transforms
import torchvision.io as tio


class RTGENEFileDataset(data.Dataset):

    def __init__(self, root_path, subject_list=None, transform=None):
        self._root_path = root_path
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((36, 120), transforms.InterpolationMode.NEAREST, antialias=False),
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
        left_img = tio.read_image(os.path.join(self._root_path, sample[0]), tio.ImageReadMode.RGB).to(torch.float32)
        right_img = tio.read_image(os.path.join(self._root_path, sample[1]), tio.ImageReadMode.RGB).to(torch.float32)

        combined = torch.concat((left_img, right_img), dim=2)
        transformed_combined = self._transform(combined)
        transformed_lt = transformed_combined[:, :, :60].clone().detach()
        transformed_rt = transformed_combined[:, :, 60:].clone().detach()

        return transformed_lt, transformed_rt, ground_truth_headpose, ground_truth_gaze
