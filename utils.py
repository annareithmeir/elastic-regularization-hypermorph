import os
from typing import Optional, Tuple, Union
from collections.abc import Generator

import utils
import datasets
import plot_utils
import numpy as np
from voxelmorph import voxelmorph as vxm
import hypermorph_spatially_adaptive as vxm_sa
import decimal
import tensorflow as tf
import tensorflow.keras.backend as K
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from scipy.ndimage import map_coordinates

def float_to_str(f: float) -> str:
    return str(f)


def float_to_str_multiple(f: list[float]) -> str:
    """
    Convert the given list of floats to a string,
    without resorting to scientific notation. Needed for filenames
    """
    ctx = decimal.Context()
    ctx.prec = 20
    tmp=""
    for fi in f:
        d1 = ctx.create_decimal(repr(fi))
        tmp+=format(d1, 'f')+"__"
    return tmp[:-2]

########################
# generators
########################

def scan_to_scan_generator(dataset: datasets.Dataset, batch_size: Optional[int] = 1) -> Generator:
    """
    The basis generator for the desired dataset from the original voxelmorph code
    :param dataset: dataset
    :param bidir:
    :param batch_size: batch size
    :param no_warp: no initial warp given
    :return: Generator with data of form (invols[m,f], outvols[m,f], segvols[m,f], kps[m,f])
    """

    return_mode = dataset.return_mode

    if batch_size>1:
        while True:
            indices = np.random.randint(len(dataset), size=batch_size) # e.g. for training, radnom batches
            print("-----INDICES!!", indices)

            if return_mode == 2:
                imgs_x=list()
                imgs_y=list()
                for idx in indices:
                    x, y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                x=np.concatenate(imgs_x, axis=0)
                y=np.concatenate(imgs_y, axis=0)
            if return_mode == 4:
                imgs_x = list()
                imgs_y = list()
                segs_x = list()
                segs_y = list()
                for idx in indices:
                    x, y, seg_x, seg_y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    seg_x = seg_x[np.newaxis, ..., np.newaxis]
                    seg_y = seg_y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                    segs_x.append(seg_x)
                    segs_y.append(seg_y)
                x = np.concatenate(imgs_x, axis=0)
                y = np.concatenate(imgs_y, axis=0)
                seg_x = np.concatenate(segs_x, axis=0)
                seg_y = np.concatenate(segs_y, axis=0)
            if return_mode == 6:
                imgs_x = list()
                imgs_y = list()
                segs_x = list()
                segs_y = list()
                kps_x = list()
                kps_y = list()
                for idx in indices:
                    x, y, seg_x, seg_y, kp_x, kp_y = dataset.__getitem__(idx)
                    x = x[np.newaxis, ..., np.newaxis]
                    y = y[np.newaxis, ..., np.newaxis]
                    seg_x = seg_x[np.newaxis, ..., np.newaxis]
                    seg_y = seg_y[np.newaxis, ..., np.newaxis]
                    imgs_x.append(x)
                    imgs_y.append(y)
                    segs_x.append(seg_x)
                    segs_y.append(seg_y)
                    kps_x.append(kp_x)
                    kps_y.append(kp_y)
                x = np.concatenate(imgs_x, axis=0)
                y = np.concatenate(imgs_y, axis=0)
                seg_x = np.concatenate(segs_x, axis=0)
                seg_y = np.concatenate(segs_y, axis=0)
                kp_x=kps_x
                kp_y=kps_y

            shape = x.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

            invols = [x, y]
            outvols = [y]
            outvols.append(zeros)

            # if return_mode==2:
            #     outvols.append(zeros)
            # else:
            #     print('give back img maskssss')
            #     img_mask = np.ones(y.shape)
            #     img_mask[y > 1e-9] = 0
            #     outvols.append(img_mask)

            idx += 1
            if idx == len(dataset):
                idx = 0

            if return_mode > 4:
                kps = [kp_x, kp_y]
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols, kps)
            elif return_mode > 2:
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols)
            else:
                yield (invols, outvols)
    else:
        idx=0
        while True:
            print("-----idx!!", idx)
            if return_mode == 2:
                x, y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
            if return_mode == 4:
                x, y, seg_x, seg_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]

            if return_mode == 6:
                x, y, seg_x, seg_y, kp_x, kp_y = dataset.__getitem__(idx)
                x = x[np.newaxis, ..., np.newaxis]
                y = y[np.newaxis, ..., np.newaxis]
                seg_x = seg_x[np.newaxis, ..., np.newaxis]
                seg_y = seg_y[np.newaxis, ..., np.newaxis]


            shape = x.shape[1:-1]
            zeros = np.zeros((batch_size, *shape, len(shape)))

            invols = [x, y]
            outvols = [y]
            outvols.append(zeros)

            # if return_mode == 2:
            #     outvols.append(zeros)
            # else:
            #     print('give back seg maskssss')
            #     img_mask = np.ones(y.shape)
            #     img_mask[np.where((y < 1e-9) & (seg_y == 0))] = 0
            #     outvols.append(img_mask)

            idx += 1
            if idx == len(dataset):
                idx = 0

            if return_mode > 4:
                kps = [kp_x, kp_y]
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols, kps)
            elif return_mode > 2:
                segvols = [seg_x, seg_y]
                yield (invols, outvols, segvols)
            else:
                yield (invols, outvols)





def random_hyperparam() -> float:
    """
    Randomly samples a hyperparameter. oversample_rate is the fraction of samples from the range boundaries, i.e. 0 or 1
    :return: random value in [0,1]
    """
    oversample_rate = 0.2
    if np.random.rand() < oversample_rate:
        return np.random.choice([0, 1])
    else:
        return np.random.rand()


def random_hyperparam_in_range(range: Optional[list] = [0, 1]):
    """
        Randomly samples a hyperparameter from a predefined range [min,max]. oversample_rate is the fraction of samples from the range boundaries, i.e. min or max
        :return: random value in [min, max]
    """
    oversample_rate = 0.2
    if np.random.rand() < oversample_rate:
        return np.random.choice([range[0], range[1]])
    else:
        return np.random.uniform(low=range[0], high=range[1])


def hyp_generator(batch_size: int, base_generator: Generator, ranges: Optional[list] = None, spatially_adaptive=False, nb_classes=1) -> Generator:
    """
    Generator for training of Hypermorph.
    :param nb_classes: nb_classes times sampling performed, if spatially adaptive is true. else once sampled and same weight is returned x times
    :param spatially_adaptive: creates reg weight array with nb_classes weights sampled, if true. if false only once sampled
    :param batch_size: batch size
    :param base_generator: the base generator for the desired dataset
    :param ranges: if desired, range for hyperparameter [min,max]. If not given then hyperparameter drawn from [0,1]
    :return: Generator with data of form (invols[m,f, hyperparam], outvols[m,f], segvols[m,f], kps[m,f]) OR
                    spattially-adaptive: (invols[m,f, hyp_tensor], outvols[m,f], segvols[m,f], kps[m,f])
    """
    while True:
        inputs, outputs, segs = next(base_generator)
        if ranges is None:
            hyp = np.expand_dims([random_hyperparam() for _ in range(batch_size)], -1)
        else:
            hyp = np.expand_dims([random_hyperparam_in_range(ranges) for _ in range(batch_size)], -1)

        inputs = (*inputs, hyp)
        outputs[1]=np.zeros(outputs[1].shape)
        # outputs = (np.concatenate([outputs[0], outputs[1]], axis=-1), outputs[1]) #[[y, mask_y],mask_y]
        print('inside hypgen', np.unique(outputs[1]), hyp)
        yield inputs, outputs


def hyp_generator_val(batch_size: int, base_generator: Generator, return_mode: int,
                      val_hyp: Optional[float] = 0.1) -> Generator:
    """
    Generator for validation of Hypermorph.
    :param spatially_adaptive:
    :param val_hyp: either float or weight list with weight per class (spatially-adaptive case)
    :param return_mode: of validation/test dataset
    :param batch_size: batch size
    :param base_generator: the base generator for the desired dataset
    :return: Generator with data of form (invols[m,f, hyperparam], outvols[m,f], segvols[m,f], kps[m,f]) OR
                                        (invols[m,f, hyp_tensor], outvols[m,f], segvols[m,f], kps[m,f]) with hyp based on seg_map of moving image
    """
    while True:
        if return_mode == 4:
            inputs, outputs, segs = next(base_generator)
        elif return_mode == 6:
            inputs, outputs, segs, kps = next(base_generator)
        else:
            print("Wrong return mode in hyp_generator_val()!", return_mode)

        hyp = np.expand_dims([val_hyp for _ in range(batch_size)], -1)

        inputs = (*inputs, hyp)
        if return_mode == 4:
            yield (inputs, outputs, segs)
        else:
            yield (inputs, outputs, segs, kps)


def hyp_generator_elastic(batch_size: int, base_generator: Generator, ranges: Optional[list] = None, spatially_adaptive=False, nb_classes=1):
    """
    Generator for training of linear elastic Hypermorph.
    :param batch_size: batch size
    :param base_generator: data generator
    :param ranges: if desired, predefined ranges for the Lame parameters mu and lambda: [[lambda_min, lambda_max],[mu_min, mu_max]]
    :return: Generator with data of form (invols[m,f, lambda, mu], outvols[m,f])
    """
    while True:
        inputs, outputs = next(base_generator)
        # lam = random_hyperparam()
        # mu = random_hyperparam_in_range([0, 1 - lam])  # constrain elastic params to max sum up to 1
        # hyp = np.array([[lam, mu] for _ in range(batch_size)])
        # print(hyp.shape)

        hyp1 = [random_hyperparam() for _ in range(batch_size)]  # [bs], lambda
        hyp2 = [random_hyperparam_in_range([0, 1 - hyp1[b]]) for b in range(batch_size)]  # [bs], mu
        hyp = np.stack([hyp1, hyp2], axis=-1)  # [bs,2]
        print(hyp.shape)

        inputs = (*inputs, hyp)  # outputs: [y, zeros]
        # print('inside elastic hypgen', np.unique(outputs[1]), hyp)
        yield (inputs, outputs)


def hyp_generator_elastic_val(batch_size: int, base_generator: Generator, return_mode: int,
                              val_lambda: Optional[float] = 0.1, val_mu: Optional[float] = 0.1) -> Generator:
    """
    Generator for training of linear elastic Hypermorph.
    :param batch_size: batch size
    :param base_generator: data generator
    :param return_mode: 4 (seg maps) or 6 (seg maps and keypoints)
    :param val_lambda: validation value for lambda
    :param val_mu: validation value for mu
    :return: Generator with data of form (invols[m,f, lambda, mu], outvols[m,f], segvols[m,f], kps[m,f])
    """
    while True:
        hyp = np.array([[val_lambda, val_mu] for _ in range(batch_size)])
        print(hyp.shape)
        if return_mode == 4:
            inputs, outputs, segs = next(base_generator)
            inputs = (*inputs, hyp)
            yield (inputs, outputs, segs)
        if return_mode == 6:
            inputs, outputs, segs, kps = next(base_generator)
            inputs = (*inputs, hyp)
            yield (inputs, outputs, segs, kps)

