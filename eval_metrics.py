import numpy as np
import scipy
from typing import Optional

from matplotlib import pyplot as plt



def dice(image1 : np.ndarray, image2 : np.ndarray, img_mask:Optional[np.ndarray]=None) -> float:
    """
    Taken from github of conditional LapIRN by Tony Mok
    :param image1: pred
    :param image2:true
    :return:
    """
    unique_class = np.unique(image2)
    dice = 0
    num_count = 0

    if img_mask is not None:
        image1[img_mask==0]=0
        # image2[img_mask==0]=0

    for i in unique_class:
        if (i == 0) or ((image1 == i).sum() == 0) or ((image2 == i).sum() == 0):
            continue

        sub_dice = np.sum(image2[image1 == i] == i) * 2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
        dice += sub_dice
        num_count += 1
    if num_count==0:
        return 0
    else:
        return dice / num_count


def dice_per_class(image1 : np.ndarray, image2 : np.ndarray, classes : list[int], img_mask:Optional[np.ndarray]=None) -> list[float]:
    """
    Computes dice scores per class labels. Based on Tony Mok LapIRN implementation
    :param image1:
    :param image2:
    :param classes: list of labels to compute dice on
    :return: list of dice scores
    """

    dice_ls=[]

    if img_mask is not None:
        image1[img_mask==0]=0

    for i in classes:
        if (i == 0) or (np.sum(image1 == i) == 0) or (np.sum(image2 == i) == 0):
            dice_ls.append(0) #TODO check if correct
            continue

        sub_dice = np.sum(image2[image1 == i] == i) * 2.0 / (np.sum(image1 == i) + np.sum(image2 == i))
        dice_ls.append(sub_dice)
    return dice_ls


def TRE(points1 : np.ndarray, points2 : np.ndarray, spacing : Optional[np.ndarray]=np.array([1,1,1]), squared : Optional[bool]=False) -> float: # TODO test
    """
    Mean target registration error, i.e. vector L1 norm, of keypoints. Based on Airlab implementation
    :param points1: shape [n,3] or [n,2], corresponding points
    :param points2: shape [n,3] or [n,2], corresponding points
    :param spacing: list of 2 or 3 values, pixel spacing
    :param squared: if true returns L2 norm, else L1
    :return: float
    """
    assert points1.shape[0]==points2.shape[0] , "shapes are "+str(points1[0].shape)+" and "+str(points2[0].shape)
    if not squared:
        return np.sqrt(np.mean(np.linalg.norm((points1 - points2)*spacing, axis=1)))
    else:
        return np.mean(np.linalg.norm(((points1 - points2)*spacing), axis=1))


def calculate_tre(F_X_Y, moving_keypoints, fixed_keypoints):
    # input keypoints torch tensors of size [n,3] or [1,n,3]
    F_X_Y_xyz = np.zeros(F_X_Y.shape, dtype=F_X_Y.dtype)
    _, _, x, y, z = F_X_Y.shape
    F_X_Y_xyz[0, 0] = F_X_Y[0, 2] * (x - 1) / 2
    F_X_Y_xyz[0, 1] = F_X_Y[0, 1] * (y - 1) / 2
    F_X_Y_xyz[0, 2] = F_X_Y[0, 0] * (z - 1) / 2

    F_X_Y_xyz_cpu = F_X_Y_xyz[0]
    fixed_keypoints=fixed_keypoints.squeeze()
    moving_keypoints=moving_keypoints.squeeze()

    fixed_disp_x = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[0], fixed_keypoints.transpose())
    fixed_disp_y = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[1], fixed_keypoints.transpose())
    fixed_disp_z = scipy.ndimage.map_coordinates(F_X_Y_xyz_cpu[2], fixed_keypoints.transpose())
    lms_fixed_disp = np.array((fixed_disp_x, fixed_disp_y, fixed_disp_z)).transpose()
    warped_fixed_keypoint = fixed_keypoints + lms_fixed_disp

    tre_score = TRE(warped_fixed_keypoint, moving_keypoints)
    return tre_score


def jacobian_determinant(disp):
    #copied from cLapIRN code
    # J = y_pred + sample_grid
    # dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    # dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    # dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]
    #
    # Jdet0 = dx[:,:,:,:,0] * (dy[:,:,:,:,1] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,1])
    # Jdet1 = dx[:,:,:,:,1] * (dy[:,:,:,:,0] * dz[:,:,:,:,2] - dy[:,:,:,:,2] * dz[:,:,:,:,0])
    # Jdet2 = dx[:,:,:,:,2] * (dy[:,:,:,:,0] * dz[:,:,:,:,1] - dy[:,:,:,:,1] * dz[:,:,:,:,0])
    #
    # Jdet = Jdet0 - Jdet1 + Jdet2
    #
    # return Jdet

    """
    From voxelmorph code
    :param y_pred:
    :param sample_grid:
    :return:

    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.

    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims

    Returns:
        jacobian determinant (scalar)
    """
    import pystrum.pynd.ndutils as nd

    # check inputs
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2


def min_max_jacobian_determinant(jac_det):
    return np.min(jac_det), np.max(jac_det)


def SDlogJ(jac_det):
    # std of log of jacdet
    log_jac_det = np.log(jac_det)
    return log_jac_det.std()

def mean_neg_jdet(jac_det):
    if (jac_det < 0).sum() == 0 :
        return 0
    else:
        return (jac_det[jac_det < 0]).mean()


def foldings_negative_fraction(jac_det):
    num_foldings = (jac_det < 0).sum()
    # print("num foldings",num_foldings)
    return float(num_foldings)/float(jac_det.size)



