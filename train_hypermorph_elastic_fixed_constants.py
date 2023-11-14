#!/usr/bin/env python

"""
Adaptation of official example script for training a HyperMorph model to tune the
regularization weight hyperparameter.
"""
import argparse
import datetime
import os
import datasets

import glob
from regularizers import elastic_regularizer_modersitzki_tf
import plot_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from voxelmorph import voxelmorph as vxm
from tensorflow.keras import backend as K
import utils
from test_hypermorph_diffusion import test_and_plot


class ValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, datapath):
        self.datapath=datapath

    def on_epoch_end(self, epoch, logs=None):

        # logging to file
        with open(log_file_train, 'a') as f:
            f.write("epoch: " + str(epoch) + ",loss: " + str(logs["loss"]) + ",sim-loss: " + str(logs["hyper_vxm_dense_transformer_loss"]) + ",reg-loss: " + str(logs["hyper_vxm_dense_flow_loss"]) + "\n")

        # validation
        if epoch > 0 and epoch % 50 == 0:

            if dataset == "LungCTCT":  # lung CT inh/exh
                train_dataset = datasets.Learn2RegLungCTDataset(self.datapath, mode="val", return_mode=6, minmax=(0, 1))
                use_keypoints = True

            if dataset == "OASIS":
                imgs = sorted(glob.glob(self.datapath + '/OASIS_OAS1_*_MR1/aligned_norm.nii.gz'))
                labels = sorted(glob.glob(self.datapath + '/OASIS_OAS1_*_MR1/aligned_seg4.nii.gz'))
                train_dataset = datasets.OASIS("val", imgs, labels=labels, return_mode=4, norm=True)
                use_keypoints=False

            inshape = train_dataset.imgshape
            print(len(train_dataset))

            base_generator = utils.scan_to_scan_generator(
                train_dataset, batch_size=args.batch_size, add_feat_axis=add_feat_axis)

            val_generator = utils.hyp_generator_val(1, base_generator, train_dataset.return_mode)

            warp_model = vxm.networks.Transform(inshape, interp_method='nearest')

            dice_score=[]
            tre_score=[]
            dice_score_per_class=[]
            idx=0
            while idx<len(train_dataset):
                if train_dataset.return_mode==4:
                    val_input, val_output, val_segs = next(val_generator)
                else:
                    val_input, val_output, val_segs, kps = next(val_generator)

                val_pred = self.model.predict(val_input)

                Y_pred = val_pred[0].squeeze()
                warp = val_pred[1]

                X=val_input[0]
                Y=val_input[1]
                label_Y_numpy=val_segs[1]

                # dice overlap
                label_Y_pred = warp_model.predict([val_segs[0], warp])
                dice_sample = utils.dice(np.floor(label_Y_pred), np.floor(label_Y_numpy))
                dice_sample_per_class = utils.dice_per_class(np.floor(label_Y_pred), np.floor(label_Y_numpy))
                dice_score.append(dice_sample)
                dice_score_per_class.append(dice_sample_per_class)

                
                print("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice: " + str(dice_sample) + ",dice per class: " + str(dice_sample_per_class) + "\n")

                with open(log_file_val, 'a') as f:
                        f.write("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice: " + str(dice_sample) + ",dice per class: " + str(
                            dice_sample_per_class) + "\n")

                idx+=1


def image_loss(y_true, y_pred):
    hyp = (1 - tf.squeeze(model.references.hyper_val))
    return hyp * image_loss_func(y_true, y_pred)


HARTMANN_LAMBDA = 12483.3
HARTMANN_MU = 25

KUMARESAN_LAMBDA = 540.811
KUMARESAN_MU = 22.5338

TADA_LAMBDA = 8060.3
TADA_MU = 164.3

LAIFOOK_LAMBDA=45.333
LAIFOOK_MU=8

BROCK_LAMBDA=15.517
BROCK_MU=1.724

MU = BROCK_MU * 0.1
LAMBDA=BROCK_LAMBDA * 0.1


def grad_loss(y_true, y_pred):
    hyp = tf.squeeze(model.references.hyper_val)
    lam = LAMBDA
    mu = MU
    elastic_loss = elastic_regularizer_modersitzki_tf(y_pred, lam, mu, reduce=True, average=True)
    tf.print(elastic_loss)
    return hyp * elastic_loss


# disable eager execution
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument('--img-list', help='line-seperated list of training files')
parser.add_argument('--img-prefix', help='optional input image file prefix')
parser.add_argument('--img-suffix', help='optional input image file suffix')
parser.add_argument('--atlas', help='atlas filename')
parser.add_argument('--model-dir', default='models',
                    help='model output directory (default: models)')
parser.add_argument('--multichannel', action='store_true',
                    help='specify that data has multiple channels')
parser.add_argument('--test-reg', nargs=3,
                    help='example registration pair and result (moving fixed moved) to test')

# training parameters
parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
parser.add_argument('--epochs', type=int, default=6000,
                    help='number of training epochs (default: 6000)')
parser.add_argument('--steps-per-epoch', type=int, default=100,
                    help='steps per epoch (default: 100)')
parser.add_argument('--load-weights', help='optional weights file to initialize with')
parser.add_argument('--initial-epoch', type=int, default=0,
                    help='initial epoch number (default: 0)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')

# network architecture parameters
parser.add_argument('--enc', type=int, nargs='+',
                    help='list of unet encoder filters (default: 16 32 32 32)')
parser.add_argument('--dec', type=int, nargs='+',
                    help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
parser.add_argument('--int-steps', type=int, default=7,
                    help='number of integration steps (default: 7)')
parser.add_argument('--int-downsize', type=int, default=2,
                    help='flow downsample factor for integration (default: 2)')

# loss hyperparameters
parser.add_argument('--image-loss', default='mse',
                    help='image reconstruction loss - can be mse or ncc (default: mse)')
parser.add_argument('--image-sigma', type=float, default=0.05,
                    help='estimated image noise for mse image scaling (default: 0.05)')
parser.add_argument('--oversample-rate', type=float, default=0.2,
                    help='hyperparameter end-point over-sample rate (default 0.2)')

parser.add_argument("--machine", type=str,
                    dest="machine", default="ganymed",
                    help="machine")

args = parser.parse_args()


dataset="LungCTCT"
model_name="fixed_constants_brock_0_1"

datapath = "~/datasets/LungCT"
train_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="train", return_mode=2, minmax=(0, 1))
use_keypoints = True

args.epochs=250
args.image_loss = "ncc"
val_hyp = 0.1
add_feat_axis = not args.multichannel

# logging stuff
log_dir = args.model_dir + "/log"
log_file_train = args.model_dir + "/log/train.txt"
log_file_val = args.model_dir + "/log/val.txt"

if not os.path.isdir(log_dir):
    os.makedirs(log_dir, exist_ok=True)

with open(log_file_train, 'w') as f:
    f.write(model_name + " -- " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n")
with open(log_file_val, 'w') as f:
    f.write(model_name + " -- " + datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y") + "\n")

args.steps_per_epoch = len(train_dataset)
base_generator = utils.scan_to_scan_generator(train_dataset, batch_size=args.batch_size, add_feat_axis=add_feat_axis)
train_generator = utils.hyp_generator(1, base_generator)

sample_shape = (1, train_dataset.imgshape[0], train_dataset.imgshape[1], train_dataset.imgshape[2], 1)
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# lambda_param = 0.05
nb_features = [
    [32, 32, 32, 32],  # encoder features
    [32, 32, 32, 32, 32, 16]  # decoder features
]
ndim = 3
unet_input_features = 2
loss_name = args.image_loss
int_steps = 7
int_downsize = 1
image_sigma = 0.05  # estimated image noise for mse image scaling
initial_epoch = 0
lr = 1e-4

print("GPU available? ", tf.config.list_physical_devices('GPU'))
device = 'cuda'

# unet architecture
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

# prepare model checkpoint save path
save_filename = os.path.join(args.model_dir, '{epoch:04d}.h5')

# tensorflow device handling
device, nb_devices = vxm.tf.utils.setup_device('GPU')

# build the model
model = vxm.networks.HyperVxmDense(
    inshape=inshape,
    nb_unet_features=[enc_nf, dec_nf],
    int_steps=int_steps,
    int_resolution=int_downsize,
    src_feats=nfeats,
    trg_feats=nfeats,
    svf_resolution=1,
)

# prepare image loss
if loss_name == 'ncc':
    image_loss_func = vxm.losses.NCC().loss
elif loss_name == 'mse':
    scaling = 1.0 / (image_sigma ** 2)
    image_loss_func = lambda x1, x2: scaling * K.mean(K.batch_flatten(K.square(x1 - x2)), -1)
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % loss_name)

model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss=[image_loss, grad_loss])
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=1000)

model.fit(train_generator,
          initial_epoch=initial_epoch,
          epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch,
          callbacks=[save_callback, ValidationCallback(datapath)],
          verbose=1)

# save final weights
model_path = os.path.join(args.model_dir, 'model_final.h5')
model.save(model_path)
