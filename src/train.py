from utils import *
import os
import voxelmorph as vxm
import numpy as np

data_repo = ""
save_filename =os.path.join("./models/", 'network_{epoch:04d}.h5')
enc_nf = [16, 32, 32, 32]
dec_nf = [32, 32, 32, 32, 32, 16, 16]

ct_no_contrast_baseline = read_data_files(100,'baseline', 'no_contrast')
ct_no_contrast_deformed = read_data_files(100, 'deformed', 'no_contrast')
ct_no_contrast_intraop = ct_no_contrast_deformed + ct_no_contrast_deformed
vm_preop = read_data_files(100, data_repo, 'baseline', 'vm') + read_data_files(100, data_repo, 'deformed', 'vm')
vm_intraop = read_data_files(100, data_repo, 'baseline', 'vm') + read_data_files(100, data_repo, 'deformed', 'vm')


# Shuffle data 
p = np.random.permutation(len(ct_no_contrast_intraop)) 
p2 = np.random.permutation(len(ct_no_contrast_intraop)) 
ct_no_contrast_intraop = np.array(ct_no_contrast_intraop)
vm_preop = np.array(vm_preop)
vm_intraop = np.array(vm_intraop)
ct_no_contrast_intraop = ct_no_contrast_intraop[p].tolist()
vm_preop = vm_preop[p2].tolist()
vm_intraop = vm_intraop[p].tolist()

generator = unet_generator(
    ct_no_contrast_intraop=ct_no_contrast_intraop, vm_preop=vm_preop, vm_intraop=vm_intraop)


# extract shape from sampled input
sample_shape = next(generator)[0][0].shape
inshape = sample_shape[1:]

# configure unet features 
nb_features = [
    enc_nf, # encoder features
    dec_nf  # decoder features
    ]

# build model
model = vxm.networks.Unet(inshape=inshape, nb_features=nb_features)
cts_tensor = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', name='cts_tensor')(model.output)

# using keras, we can easily form new models via tensor pointers
model = tf.keras.models.Model(model.inputs, cts_tensor, name="CTS_finder_model")

# Training 
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=tf.keras.losses.MeanSquaredError())
model.load_weights('/home/sidaty/Juan/models/Unet_2_inputs_256_latest_trained_processed_max/unet_2_inputs_1200.h5')
save_callback = tf.keras.callbacks.ModelCheckpoint(save_filename, period=20)
model.fit_generator(generator,
                    initial_epoch=1,
                    epochs=1500,
                    steps_per_epoch=100,
                    callbacks=[save_callback],
                    verbose=1
                    )