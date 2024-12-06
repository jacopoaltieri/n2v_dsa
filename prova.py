import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppressing warnings 
import tensorflow as tf

from n2v.models import N2VConfig, N2V
import numpy as np
from csbdeep.utils import plot_history
from n2v.utils.n2v_utils import manipulate_val_data
from n2v.internals.N2V_DataGenerator import N2V_DataGenerator
import matplotlib.pyplot as plt

#### Configuration ####
patch_size = 128
train_ratio = 0.8 # The rest in validation
train_batch = 128
#### Main script ####
datagen = N2V_DataGenerator()

imgs = datagen.load_imgs_from_directory(directory = '/home/jaltieri/tifs/', 
                                        filter='00107_9ea6c142-46ca-48df-865e-308cf11d111e_20221117_110426.7411_unfiltered.tif',dims='TYX')

patch_shape = (patch_size, patch_size)
patches = datagen.generate_patches_from_list(imgs, shape=patch_shape)


train_val_split = int(patches.shape[0] * train_ratio)
X = patches[:train_val_split]
X_val = patches[train_val_split:]


config = N2VConfig(X, unet_kern_size=3, 
                   unet_n_first=64, unet_n_depth=3, train_steps_per_epoch=int(X.shape[0]/train_batch), train_epochs=20, train_loss='mse', 
                   batch_norm=True, train_batch_size=train_batch, n2v_perc_pix=0.198, n2v_patch_shape=(patch_size, patch_size), 
                   n2v_manipulator='uniform_withCP', n2v_neighborhood_radius=5, single_net_per_channel=False)



# a name used to identify the model --> change this to something sensible!
model_name = 'n2v_prova'
# the base directory in which our model will live
basedir = 'models'
# We are now creating our network model.
model = N2V(config, model_name, basedir=basedir)
     
history = model.train(X, X_val) 
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'])