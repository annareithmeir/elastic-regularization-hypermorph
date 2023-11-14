import numpy as np
import torch
import tensorflow as tf
from typing import Union, Optional, Tuple

RONOVSKY_2017_LAMBDA = 0
RONOVSKY_2017_MU=1.5

SCHENKENFELDER_2021_LAMBDA=3.1034
SCHENKENFELDER_2021_MU=0.3448

YANOVSKY_2022_LAMBDA=2000
YANOVSKY_2022_MU=2000

HARTMANN_LAMBDA=12483.3
HARTMANN_MU=25




def elastic_regularizer_modersitzki_tf(u: tf.Tensor, lam: float, mu: float, pixel_spacing:Optional[np.ndarray]=None, img_mask:Optional[np.ndarray]=None, average : Optional[bool]=True, reduce: Optional[bool]=True, batch: Optional[bool]=True) -> tf.Tensor:
    """
    :param u: displacement field in shape of (bs, x,y,z,3) or (bs, x,y, 2)
    :param mu: Elasticity constant 1
    :param lam: Elasticity constant 2
    :param pixel_spacing: spacing in shape of [x,y (,z)]
    :return:
    """
    # same as torch version up to a precision of 1e-8

    if batch:
        shape = u.shape
        dim = len(shape) - 2  # last dim is vector
        if pixel_spacing is None:
            pixel_spacing = np.ones(dim)

        if dim == 3:  # u=[bs, y,x,z,3]
            assert(u.shape[-1]==3)
            dx = u[:, :, :, 1:, :] - u[:, :, :, :-1, :]  # lxmxnx3
            paddings = tf.constant([[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]])
            dx = tf.pad(dx, paddings, "CONSTANT")  # (3zzxxyy)
            dy = u[:, :, 1:, :, :] - u[:, :, :-1, :, :]  # lxmxnx3
            paddings = tf.constant([[0, 0], [0, 0], [0, 1], [0, 0], [0, 0]])
            dy = tf.pad(dy, paddings, "CONSTANT")  # (3zzxxyy)
            dz = u[:, 1:, :, :, :] - u[:, :-1, :, :, :]  # lxmxnx3
            paddings = tf.constant([[0, 0], [0, 1], [0, 0], [0, 0], [0, 0]])
            dz = tf.pad(dz, paddings, "CONSTANT")  # (3zzxxyy)

            loss = dx[..., 0] + dy[..., 1] + dz[..., 2]  # div
            loss= tf.math.square(loss)*(lam / 2)

            derivs = [dx, dy, dz]

            for i in range(dim):
                for j in range(dim):
                    part1 = tf.square(tf.math.add(derivs[j][:,:,:,:,i],(derivs[i][:,:,:,:,j])))*(mu / 4)
                    part1 = tf.squeeze(part1)
                    loss=tf.math.add(loss, part1)

            if img_mask is not None:
                print("in elastic loss, masking out")
                loss = loss * img_mask

    else:
        pass

    if not reduce:
        return loss
    elif average:
        return tf.reduce_mean(loss)
    else:
        return torch.reduce_sum(loss)

