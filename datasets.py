import glob
import itertools
import os
from abc import ABCMeta, abstractmethod
import json
import numpy as np
import nibabel as nib
import random
from typing import Union, Optional, Tuple

import eval_metrics
import torchio as tio


class Dataset(metaclass=ABCMeta):
    data_path = None
    images_pair = None
    labels_pair = None
    keypoints_pair = None
    normalize_mode = False
    mode = None
    return_mode = None
    imgshape = None
    spacing = None
    classes = None

    def __init__(self, data_path: str, mode: int, normalize_mode: Optional[bool]) -> None:
        assert os.path.isdir(data_path), 'Please provide a valid data file. \n (not valid {})'.format(data_path)
        self.data_path = data_path
        assert (mode in ["train", "test", "val", "val2", "val4", "train4"]), 'Please provide a valid return mode.'
        self.mode = mode
        self.normalize_mode = normalize_mode

    def __len__(self) -> int:
        """

        @return: number of samples/image pairs in dataset
        """
        return len(self.images_pair)

    @abstractmethod
    def __getitem__(self, item: int) -> Tuple[np.array, ...]:
        pass


    def get_initial_dice(self):  # TODO test
        assert self.return_mode > 2
        score = 0
        elements = 0
        for _, _, moving_labels, fixed_labels, _, _ in iter(self):
            score += eval_metrics.dice(moving_labels.squeeze(), fixed_labels.squeeze())
            elements += 1
        return score / elements


class Learn2RegLungCTDataset(Dataset):
    """
    Lung CT-CT dataset from from https://learn2reg.grand-challenge.org/Datasets/
    Train, test, val: 20,3,6

    original : origin/spacing/shape
    (0.0, 0.0, 0.0)
    (1.75, 1.25, 1.75)
    (192, 192, 208)
    """

    def __init__(self, data_path: str, mode: str, normalize_mode: Optional[bool] = True,
                 clip_mode: Optional[bool] = True, seg_mode:Optional[str]="LungRibsLiver") -> None:
        super(Learn2RegLungCTDataset, self).__init__(data_path, mode, normalize_mode)

        def _read_filenames():
            x_ls = list()
            y_ls = list()
            keypoints_x_ls = list()
            keypoints_y_ls = list()
            masks_x_ls = list()
            masks_y_ls = list()

            if self.mode == "train"or self.mode == "train4":
                # load train paths. 0=fixed, 1=moving
                for i in range(1, 21):
                    file_str = "LungCT_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_y_ls.append(self.data_path + "/masksTr/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksTr/" + file_str + "_0001.nii.gz")
            elif self.mode == "val" or self.mode == "val2":
                for i in range(21, 24):
                    file_str = "LungCT_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTs/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTs/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0001.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_y_ls.append(self.data_path + "/masksTs/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksTs/" + file_str + "_0001.nii.gz")
            else:
                for i in range(24, 30):
                    file_str = "LungCT_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTs/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTs/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTs/" + file_str + "_0001.csv")
                    if self.seg_mode == "LungRibsLiver":
                        masks_y_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "LungRibs":
                        masks_y_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksLungRibs/" + file_str + "_0001.nii.gz")
                    elif self.seg_mode == "Lung":
                        masks_y_ls.append(self.data_path + "/masksTs/" + file_str + "_0000.nii.gz")
                        masks_x_ls.append(self.data_path + "/masksTs/" + file_str + "_0001.nii.gz")

            return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), list(zip(keypoints_x_ls, keypoints_y_ls))

        assert (mode in ['train', 'val', 'test', 'val2', 'val4', 'train4']), 'Please provide a valid mode.'
        self.spacing = (2, 2, 2)  # 1.75x1.25x1.75mm
        self.imgshape = (192, 128, 192 )
        self.seg_mode=seg_mode

        if seg_mode=="LungRibsLiver":
            self.classes = {
                0: "background",
                1: "lung",
                2: "bones",
                3: "liver"
            }
        elif seg_mode=="LungRibs":
            self.classes = {
                0: "background",
                1: "lung",
                2: "bones"
            }
        elif seg_mode=="Lung":
            self.classes = {
                0: "background",
                1: "lung"
            }


        if self.mode == "train" or self.mode == "val2":
            self.return_mode = 2
        elif self.mode == "val" or self.mode == "test":
            self.return_mode = 6
        elif self.mode == "train4" or self.mode=='val4':
            self.return_mode = 4
        else:
            print("Wrong mode given")

        self.clip_mode = clip_mode

        self.images_pair, self.labels_pair, self.keypoints_pair = _read_filenames()

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        '''
        @param idx: Index of the item to return
        @return: numpy arrays
        '''

        x_file, y_file = self.images_pair[idx]
        if self.return_mode == 6:
            keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
            keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')

            keypoints_x = keypoints_x[:, [2, 1, 0]]
            keypoints_y = keypoints_y[:, [2, 1, 0]]

            keypoints_x[:, 0] = keypoints_x[:, 0] * -1
            keypoints_y[:, 0] = keypoints_y[:, 0] * -1

            keypoints_x[:, 0] = keypoints_x[:, 0] +208
            keypoints_y[:, 0] = keypoints_y[:, 0] +208

            keypoints_x[:,0] = keypoints_x[:,0]*1.75/2
            keypoints_x[:,1] = keypoints_x[:,1]*1.25/2
            keypoints_x[:,2] = keypoints_x[:,2]*1.75/2
            keypoints_y[:,0] = keypoints_y[:,0]*1.75/2
            keypoints_y[:,1] = keypoints_y[:,1]*1.25/2
            keypoints_y[:,2] = keypoints_y[:,2]*1.75/2

            keypoints_x[:, 0] = keypoints_x[:, 0] + 5
            keypoints_x[:, 1] = keypoints_x[:, 1] + 4
            keypoints_x[:, 2] = keypoints_x[:, 2] + 12
            keypoints_y[:, 0] = keypoints_y[:, 0] + 5
            keypoints_y[:, 1] = keypoints_y[:, 1] + 4
            keypoints_y[:, 2] = keypoints_y[:, 2] + 12

        if self.return_mode >= 4:
            labels_x_file, labels_y_file = self.labels_pair[idx]
            subject_dict = {
                "image_x": tio.ScalarImage(x_file),
                "labels_x": tio.LabelMap(labels_x_file),
                "image_y": tio.ScalarImage(y_file),
                "labels_y": tio.LabelMap(labels_y_file)
            }
        else:
            subject_dict = {
                "image_x": tio.ScalarImage(x_file),
                "image_y": tio.ScalarImage(y_file),
            }

        subject = tio.Subject(subject_dict)

        resample_uniform = tio.Resample(2)
        subject = resample_uniform(subject)

        if self.clip_mode:
            clamp = tio.Clamp(out_min=-1100, out_max=1518)  # tony mok github comment
            subject = clamp(subject)

        if self.normalize_mode:
            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_x"].numpy().min(), subject["image_x"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_y"].numpy().min(), subject["image_y"].numpy().max()))
            subject["image_x"] = rescale_x(subject["image_x"])
            subject["image_y"] = rescale_y(subject["image_y"])

        crop_or_pad = tio.CropOrPad((192, 128, 192),
                                    padding_mode=0.0)  # before: (182, 120, 168), after: (192, 128, 192 ) - added (10,8, 24)
        subject = crop_or_pad(subject)

        x = np.flipud(subject["image_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
        y = np.flipud(subject["image_y"].numpy().transpose((0, 3, 2, 1)).squeeze())

        if self.return_mode == 6:
            labels_x = np.flipud(subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
            labels_y = np.flipud(subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze())
            return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                float).squeeze(), labels_y.astype(float).squeeze(), keypoints_x, keypoints_y
        elif self.return_mode == 4:
            labels_x = np.flipud(subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze())
            labels_y = np.flipud(subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze())
            return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                float).squeeze(), labels_y.astype(float).squeeze()
        else:
            return x.astype(float).squeeze(), y.astype(float).squeeze()



class NLST2023Dataset(Dataset):
    """
    NLST 2023 dataset from https://learn2reg.grand-challenge.org/Datasets/
    209 annotated images in total
    Train, test, val:  169, 30 (first 30 pairs), 10 (as in json file)
    """

    def __init__(self, data_path: str, mode: str, normalize_mode: Optional[bool] = True,
                 clip_mode: Optional[bool] = True) -> None:
        super(NLST2023Dataset, self).__init__(data_path, mode, normalize_mode)

        def _read_filenames():
            x_ls = list()
            y_ls = list()
            keypoints_x_ls = list()
            keypoints_y_ls = list()
            masks_x_ls = list()
            masks_y_ls = list()

            if self.mode == "train" or self.mode == "train4":
                # load train paths. 0=fixed, 1=moving
                for i in range(30, 101):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    # keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    # keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")

                for i in range(200, 300):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    # keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    # keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
            elif self.mode == "val" or self.mode == "val2" or self.mode == "val4":
                for i in range(101, 111):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    # masks_y_ls.append(self.data_path + "/masksTr/" + file_str + "_0000.nii.gz")
                    masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    # masks_x_ls.append(self.data_path + "/masksTr/" + file_str + "_0001.nii.gz")
                    masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")
            else:
                for i in range(1, 31):
                    file_str = "NLST_" + str(i).zfill(4)
                    y_ls.append(self.data_path + "/imagesTr/" + file_str + "_0000.nii.gz")
                    x_ls.append(self.data_path + "/imagesTr/" + file_str + "_0001.nii.gz")
                    keypoints_y_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0000.csv")
                    keypoints_x_ls.append(self.data_path + "/keypointsTr/" + file_str + "_0001.csv")
                    # masks_y_ls.append(self.data_path + "/masksTr/" + file_str + "_0000.nii.gz")
                    masks_y_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0000.nii.gz")
                    # masks_x_ls.append(self.data_path + "/masksTr/" + file_str + "_0001.nii.gz")
                    masks_x_ls.append(self.data_path[:-5] + "/masksLungRibsLiver/" + file_str + "_0001.nii.gz")

            return list(zip(x_ls, y_ls)), list(zip(masks_x_ls, masks_y_ls)), list(zip(keypoints_x_ls, keypoints_y_ls))

        assert (mode in ['train', 'train4', 'val', 'test', 'test4', 'val4', 'val2']), 'Please provide a valid mode.'
        self.spacing = (1.5, 1.5, 1.5)  # 1.75x1.25x1.75mm
        self.imgshape = (224, 192, 224)
        # self.classes = {
        #     0: "background",
        #     1: "lung"
        # }

        self.classes = {
            0: "background",
            1: "lung",
            2: "ribs",
            3: "liver"
        }

        if self.mode == "train" or self.mode == "val2":
            self.return_mode = 2
        elif self.mode == "train4" or self.mode == "val4" :
            self.return_mode = 4
        elif self.mode == "val" or self.mode == "test":
            self.return_mode = 6
        else:
            print("Wrong mode given")

        self.clip_mode = clip_mode
        self.images_pair, self.labels_pair, self.keypoints_pair = _read_filenames()

    def __getitem__(self, idx: int) -> Tuple[np.array, ...]:
        '''
        @param idx: Index of the item to return
        @return: numpy arrays
        '''

        x_file, y_file = self.images_pair[idx]
        if self.return_mode == 6:
            keypoints_x = np.genfromtxt(self.keypoints_pair[idx][0], delimiter=',')
            keypoints_y = np.genfromtxt(self.keypoints_pair[idx][1], delimiter=',')
            keypoints_x = keypoints_x[:, [2, 1, 0]]
            keypoints_y = keypoints_y[:, [2, 1, 0]]
            # keypoints_x = keypoints_x*1/self.spacing
            # keypoints_y = keypoints_y*1/self.spacing

        if self.return_mode >= 4:
            # labels_x_file, labels_y_file = self.labels_pair[idx]
            # labels_x = self.load_data(labels_x_file)
            # labels_y = self.load_data(labels_y_file)
            # labels_y = np.flipud(labels_y.transpose((0, 3, 2, 1)).squeeze())
            # labels_x = np.flipud(labels_x.transpose((0, 3, 2, 1)).squeeze())
            labels_x_file, labels_y_file = self.labels_pair[idx]
            subject_dict = {
                "image_x": tio.ScalarImage(x_file),
                "labels_x": tio.LabelMap(labels_x_file),
                "image_y": tio.ScalarImage(y_file),
                "labels_y": tio.LabelMap(labels_y_file)
            }
        else:
            subject_dict = {
                "image_x": tio.ScalarImage(x_file),
                "image_y": tio.ScalarImage(y_file),
            }

        subject = tio.Subject(subject_dict)

        if self.clip_mode:
            clamp = tio.Clamp(out_min=-1100, out_max=1518)  # tony mok github comment
            subject = clamp(subject)

        if self.normalize_mode:
            rescale_x = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_x"].numpy().min(), subject["image_x"].numpy().max()))
            rescale_y = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
                subject["image_y"].numpy().min(), subject["image_y"].numpy().max()))
            subject["image_x"] = rescale_x(subject["image_x"])
            subject["image_y"] = rescale_y(subject["image_y"])

        x = subject["image_x"].numpy().transpose((0, 3, 2, 1)).squeeze()
        y = subject["image_y"].numpy().transpose((0, 3, 2, 1)).squeeze()

        if self.return_mode == 6:
            labels_x = subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze()
            labels_y = subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze()
            return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                float).squeeze(), labels_y.astype(float).squeeze(), keypoints_x, keypoints_y
        elif self.return_mode == 4:
            labels_x = subject["labels_x"].numpy().transpose((0, 3, 2, 1)).squeeze()
            labels_y = subject["labels_y"].numpy().transpose((0, 3, 2, 1)).squeeze()
            return x.astype(float).squeeze(), y.astype(float).squeeze(), labels_x.astype(
                float).squeeze(), labels_y.astype(float).squeeze()
        else:
            return x.astype(float).squeeze(), y.astype(float).squeeze()

