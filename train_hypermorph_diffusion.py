#!/usr/bin/env python

"""
Adaptation of official example script for training a HyperMorph model to tune the
regularization weight hyperparameter.
"""
import argparse
import datetime
import os
import datasets
import eval_metrics
import plot_utils
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.compat.v1.Session(config=config)
#
# import gc
# gc.collect()

tf.keras.backend.clear_session()

from voxelmorph import voxelmorph as vxm
from tensorflow.keras import backend as K
from test_hypermorph_diffusion import test_and_plot
import hypermorph_spatially_adaptive as vxm_sa


class ValidationCallback(tf.keras.callbacks.Callback):

    def __init__(self, datapath):
        self.datapath = datapath

    def on_epoch_end(self, epoch, logs=None):

        # logging to file
        with open(log_file_train, 'a') as f:
            f.write("epoch: " + str(epoch) + ",loss: " + str(logs["loss"]) + ",sim-loss: " + str(
                logs["hyper_vxm_dense_transformer_loss"]) + ",reg-loss: " + str(
                logs["hyper_vxm_dense_flow_loss"]) + "\n")

        # validation
        if epoch > 0 and (epoch+1) % val_interval == 0:
            if dataset == "LungCTCT":  # lung CT inh/exh
                val_dataset = datasets.Learn2RegLungCTDataset(datapath, "val", normalize_mode=True)
                use_keypoints = True
            elif dataset == "NLST23":
                val_dataset = datasets.NLST2023Dataset(datapath, mode="val", normalize_mode=True)
                use_keypoints = True
            else:
                print("Wrong dataset name given!")

            inshape = val_dataset.imgshape

            base_generator = utils.scan_to_scan_generator(
                val_dataset, batch_size=1)
            val_generator = utils.hyp_generator_val(1, base_generator, val_dataset.return_mode, val_hyp=val_hyp)

            warp_model = vxm.networks.Transform(inshape, interp_method='nearest')

            dice_score = list()
            tre_score = list()
            dice_score_per_class = list()
            idx = 0
            while idx < len(val_dataset):
                if val_dataset.return_mode == 4:
                    val_input, val_output, val_segs = next(val_generator)
                else:
                    val_input, val_output, val_segs, kps = next(val_generator)

                img_mask = None
                if dataset == "LungCTCT":
                    print("masking...")
                    img_mask = np.ones(val_input[1].shape)
                    img_mask[np.where((val_input[1] < 1e-9) & (val_segs[1] == 0))] = 0

                val_pred = self.model.predict(val_input)
                Y_pred = val_pred[0].squeeze()
                warp = val_pred[1]

                X = val_input[0]
                Y = val_input[1]
                label_Y_numpy = val_segs[1]

                # dice overlap
                label_Y_pred = warp_model.predict([val_segs[0], warp])
                dice_sample = eval_metrics.dice(np.floor(label_Y_pred), np.floor(label_Y_numpy), img_mask=img_mask)
                dice_sample_per_class = eval_metrics.dice_per_class(np.floor(label_Y_pred), np.floor(label_Y_numpy), classes=val_dataset.classes, img_mask=img_mask)
                dice_score.append(dice_sample)
                dice_score_per_class.append(dice_sample_per_class)

                # # TRE
                tre_sample = -1
                if use_keypoints:
                    kps[0]=np.array(kps[0])
                    kps[1]=np.array(kps[1])
                    tre_sample = eval_metrics.calculate_tre(warp, kps[0], kps[1])
                    tre_score.append(tre_sample)

                    with open(log_file_val, 'a') as f:
                        f.write("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice per class: " + str(
                            dice_sample_per_class) + ",TRE: " + str(tre_sample) + "\n")

                    print("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice: " + str(
                        dice_sample) + ",dice per class: " + str(
                        dice_sample_per_class) + ",TRE: " + str(tre_sample) + "\n")
                else:
                    print("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice: " + str(
                        dice_sample) + ",dice per class: " + str(
                        dice_sample_per_class) + "\n")

                    with open(log_file_val, 'a') as f:
                        f.write("Val sample: " + str(idx) + ",epoch: " + str(epoch) + ",dice: " + str(
                            dice_sample) + ",dice per class: " + str(
                            dice_sample_per_class) + "\n")

                idx += 1


def image_loss(y_true, y_pred):
    hyp = (1 - tf.squeeze(model.references.hyper_val))
    return hyp * image_loss_func(y_true, y_pred)


def grad_loss(y_true, y_pred):
    hyp = tf.squeeze(model.references.hyper_val)
    return hyp * vxm_sa.losses.Grad('l2', loss_mult=int_downsize).loss(None, y_pred) # ignores first arg y_true


# disable eager execution
tf.compat.v1.experimental.output_all_intermediates(True)
tf.compat.v1.disable_eager_execution()

# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
# parser.add_argument('--img-list', help='line-seperated list of training files')
# parser.add_argument('--img-prefix', help='optional input image file prefix')
# parser.add_argument('--img-suffix', help='optional input image file suffix')
# parser.add_argument('--atlas', help='atlas filename')
# parser.add_argument('--model-dir', default='models',
#                     help='model output directory (default: models)')
# parser.add_argument('--multichannel', action='store_true',
#                     help='specify that data has multiple channels')
# parser.add_argument('--test-reg', nargs=3,
#                     help='example registration pair and result (moving fixed moved) to test')

# training parameters
# parser.add_argument('--gpu', default='0', help='GPU ID numbers (default: 0)')
# parser.add_argument('--batch-size', type=int, default=1, help='batch size (default: 1)')
# parser.add_argument('--epochs', type=int, default=6000,
#                     help='number of training epochs (default: 6000)')
# parser.add_argument('--steps-per-epoch', type=int, default=100,
#                     help='steps per epoch (default: 100)')
# parser.add_argument('--load-weights', help='optional weights file to initialize with')
# parser.add_argument('--initial-epoch', type=int, default=0,
#                     help='initial epoch number (default: 0)')
# parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
#
# # network architecture parameters
# parser.add_argument('--enc', type=int, nargs='+',
#                     help='list of unet encoder filters (default: 16 32 32 32)')
# parser.add_argument('--dec', type=int, nargs='+',
#                     help='list of unet decorder filters (default: 32 32 32 32 32 16 16)')
# parser.add_argument('--int-steps', type=int, default=7,
#                     help='number of integration steps (default: 7)')
# parser.add_argument('--int-downsize', type=int, default=2,
#                     help='flow downsample factor for integration (default: 2)')
#
# # loss hyperparameters
# parser.add_argument('--image-loss', default='mse',
#                     help='image reconstruction loss - can be mse or ncc (default: mse)')
# parser.add_argument('--image-sigma', type=float, default=0.05,
#                     help='estimated image noise for mse image scaling (default: 0.05)')
# parser.add_argument('--oversample-rate', type=float, default=0.2,
#                     help='hyperparameter end-point over-sample rate (default 0.2)')


args = parser.parse_args()

# data stuff
dataset = "LungCTCT"
model_name = "500epochs_ncc"

args.epochs = 250
val_interval=50
args.image_loss = "ncc"
args.batch_size = 2
args.lr=1e-4
val_hyp = 0.1
# add_feat_axis = True
load_model_str=None


data_base_path="~/datasets"

if dataset == "LungCTCT":  # lung CT inh/exh
    datapath = data_base_path + "/LungCT"
    train_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="train4", normalize_mode=True)
    val_dataset = datasets.Learn2RegLungCTDataset(datapath, mode="val4", normalize_mode=True)
    use_keypoints = True
elif dataset == "NLST23":
    datapath = data_base_path + "/NLST23/NLST"
    train_dataset = datasets.NLST2023Dataset(datapath, mode="train", normalize_mode=True)
    val_dataset = datasets.NLST2023Dataset(datapath, mode="val2", normalize_mode=True)
    use_keypoints = True
else:
    print("Wrong dataset name specified!")
print("Using data ", dataset, " of length (train) ", len(train_dataset), " and (val) ", len(val_dataset), " and batchsize ", args.batch_size)
print("GPU available? ", tf.config.list_physical_devices('GPU'))

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

base_generator = utils.scan_to_scan_generator(train_dataset, batch_size=args.batch_size)
val_base_generator = utils.scan_to_scan_generator(val_dataset, batch_size=args.batch_size)
train_generator = utils.hyp_generator(args.batch_size, base_generator)
val_generator = utils.hyp_generator(args.batch_size, val_base_generator)

args.steps_per_epoch = len(train_dataset)/args.batch_size

sample_shape = (args.batch_size, train_dataset.imgshape[0], train_dataset.imgshape[1], train_dataset.imgshape[2], 1)
inshape = sample_shape[1:-1]
nfeats = sample_shape[-1]

# training parameters
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

if load_model_str:
    model = vxm.networks.HyperVxmDense.load(load_model_str, input_model=None)
    print("LOADEDMODEL FROM", load_model_str)

# prepare image loss
if loss_name == 'ncc':
    image_loss_func = vxm_sa.losses.NCC().loss
# elif loss_name == 'mi':
#     image_loss_func = vxm.losses.MutualInformation().loss
elif loss_name == 'mse':
    scaling = 1.0 / (image_sigma ** 2)
    image_loss_func = lambda x1, x2: scaling * K.mean(K.batch_flatten(K.square(x1 - x2)), -1)
else:
    raise ValueError('Image loss should be "mse" or "ncc", but found "%s"' % loss_name)


model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=args.lr), loss=[image_loss, grad_loss])
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, save_freq=500)

model.fit(train_generator,
          initial_epoch=initial_epoch,
          epochs=args.epochs,
          steps_per_epoch=args.steps_per_epoch,
          validation_data=val_generator,
          validation_freq=val_interval,
          validation_steps=len(val_dataset),
          callbacks=[save_callback, ValidationCallback(datapath)],
          verbose=1)

# save final weights
model_path = os.path.join(args.model_dir, 'model_final.h5')
model.save(model_path)
