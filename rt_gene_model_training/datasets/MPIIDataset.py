import glob
import os

import torch
from torch.utils import data
import albumentations as albu
from albumentations import pytorch
import cv2
from datasets.TrainingPhase import TrainingPhase
import numpy as np
import scipy.io as sio


class MPIIWithinSubjectDataset(data.Dataset):

    def __init__(self, root_path, phase=TrainingPhase.Training, fraction=0.95):
        self._data = []
        labels = []
        self._face_model = sio.loadmat(os.path.join(root_path, '6 points-based face model.mat'))["model"]
        subjects = glob.glob(os.path.join(root_path, "Data", "Original", "p*"))

        for subject_path in subjects:
            data_store_idx = 0
            camera_calibration = os.path.join(subject_path, "Calibration/Camera.mat")
            camera_matrix = sio.loadmat(camera_calibration)["cameraMatrix"]
            days = sorted(list(glob.glob(os.path.join(subject_path, "day*"))))
            for day in days:
                with open(os.path.join(day, "annotation.txt"), "r") as reader:
                    image_annotations = reader.readlines()

                for idx, annotation in enumerate(image_annotations):
                    annotations = [float(num) for num in annotation.split(" ")]
                    data_store_idx += 1
                    img_path = os.path.join(day, str("{:04d}.jpg".format(idx + 1)))

                    img, hr, ht, gt = img_path, annotations[29:32], annotations[32:35], annotations[26:29]
                    labels.append((img, hr, ht, gt, camera_matrix.flatten().tolist()))

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

    @staticmethod
    def visualize_eye_result(eye_image, est_gaze):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(eye_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

        endpoint_x = -50.0 * np.cos(est_gaze[0]) * np.sin(est_gaze[1]) + center_x
        endpoint_y = -50.0 * np.sin(est_gaze[0]) + center_y

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
        return output_image

    @staticmethod
    def normalize_img(img, target_3d, head_rotation, gc, roi_size, cam_matrix, focal_new=960, distance_new=600):
        if roi_size is None:
            roi_size = (60, 36)

        distance = np.linalg.norm(target_3d)
        z_scale = distance_new / distance
        cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                            [0.0, focal_new, roi_size[1] / 2],
                            [0, 0, 1.0]])
        scale_mat = np.array([[1.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, z_scale]])
        h_rx = head_rotation[:, 0]
        forward = (target_3d / distance)
        down = np.cross(forward, h_rx)
        down = down / np.linalg.norm(down)
        right = np.cross(down, forward)
        right = right / np.linalg.norm(right)

        rot_mat = np.array([right.T, down.T, forward.T])
        warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
        img_warped = cv2.warpPerspective(img, warp_mat, roi_size)

        # rotation normalization
        cnv_mat = scale_mat @ rot_mat
        h_rnew = cnv_mat @ head_rotation
        hrnew = cv2.Rodrigues(h_rnew)[0].reshape((3,))
        htnew = cnv_mat @ target_3d

        # gaze vector normalization
        gcnew = cnv_mat @ gc
        gvnew = gcnew - htnew
        gvnew = gvnew / np.linalg.norm(gvnew)

        return img_warped, hrnew, gvnew

    def __len__(self):
        return len(self._data)

    def __getitem__(self, item):
        img_path, headpose_hr, headpose_ht, gaze_target, camera_matrix = self._data[item]
        img = cv2.imread(img_path)
        camera_matrix = np.array(camera_matrix).reshape(3, 3)
        h_r = cv2.Rodrigues(np.array(headpose_hr))[0]
        f_c = np.dot(h_r, self._face_model)
        f_c = np.array(headpose_ht).T[:, np.newaxis] + f_c

        right_eye_center = 0.5 * (f_c[:, 0] + f_c[:, 1])
        left_eye_center = 0.5 * (f_c[:, 2] + f_c[:, 3])

        gaze_target = np.array(gaze_target).T

        right_image, right_headpose, right_gaze = self.normalize_img(img, right_eye_center, h_r, gaze_target, (60, 36), camera_matrix)
        left_image, left_headpose, left_gaze = self.normalize_img(img, left_eye_center, h_r, gaze_target, (60, 36), camera_matrix)

        left_eye_theta = np.arcsin(-1 * left_gaze[1])
        left_eye_phi = np.arctan2(-1 * left_gaze[0], -1 * left_gaze[2])

        right_eye_theta = np.arcsin(-1 * right_gaze[1])
        right_eye_phi = np.arctan2(-1 * right_gaze[0], -1 * right_gaze[2])

        gaze_theta = (left_eye_theta + right_eye_theta) / 2.0
        gaze_phi = (left_eye_phi + right_eye_phi) / 2.0

        left_rotation_matrix = cv2.Rodrigues(left_headpose)[0]  # ignore the Jackobian matrix
        left_zv = left_rotation_matrix[:, 2]
        left_head_theta = np.arcsin(left_zv[1])
        left_head_phi = np.arctan2(left_zv[0], left_zv[2])

        right_rotation_matrix = cv2.Rodrigues(right_headpose)[0]  # ignore the Jackobian matrix
        right_zv = right_rotation_matrix[:, 2]
        right_head_theta = np.arcsin(right_zv[1])
        right_head_phi = np.arctan2(right_zv[0], right_zv[2])

        head_theta = (left_head_theta + right_head_theta) / 2.0
        head_phi = (left_head_phi + right_head_phi) / 2.0

        headpose = torch.Tensor([head_theta, head_phi]).to(torch.float32)
        gaze = torch.Tensor([gaze_theta, gaze_phi]).to(torch.float32)

        transformed_lt = self._transform(image=left_image)["image"]
        transformed_rt = self._transform(image=right_image)["image"]

        return transformed_lt, transformed_rt, headpose, gaze


if __name__ == "__main__":
    dataset = MPIIWithinSubjectDataset(root_path="/datasets/MPIIGaze/")
    print(len(dataset))