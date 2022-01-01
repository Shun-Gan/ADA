#########################################################################
# MODEL PARAMETERS														#
#########################################################################
# version (0 for alexnet and 1 for VGG )
version = 0
# batch size
video_b_s = 2
image_b_s = 20
# number of frames
num_frames = 8 #5

# origin shape c
normal_shape_c = 640
# origin shape r
normal_shape_r = 360

# number of cols of input images
shape_c = 320

if version == 1:
    # number of rows of input images
    shape_r = 256
    # number of cols of model outputs
    shape_c_out = 80
    # number of rows of model outputs
    shape_r_out = 44

else:
    # number of rows of input images
    shape_r = 256#240
    # number of cols of model outputs
    shape_c_out = 78
    # number of rows of model outputs
    shape_r_out = 42 #120


# number of rows of attention
shape_r_attention = 64
# number of cols of attention
shape_c_attention = 80

# number of rows of downsampled maps
shape_r_gt = 32
# number of cols of downsampled maps
shape_c_gt = 40

# final upsampling factor
upsampling_factor = 16
# number of epochs
nb_epoch = 10
# number of timestep
nb_timestep = 4
# number of learned priors
nb_gaussian = 16

# path of continuous saliency map
maps_path = '/maps/'
# path of fixation maps
fixs_path = '/fixation/maps/'
# path of images
frames_path = '/images/'

# number of training videos
nb_train = 100
# number of validation videos
nb_videos_val = 150
# number of iterations for training using pytorch
nb_train_pt = 100000
# number of small test batch while training
nb_small_test_batch = 10

#########################################################################
# TRAINING SETTINGS										            	#
#########################################################################
# path of training videos
videos_train_paths = ['./train/']
# path of validation videos
videos_val_paths = ['./test/']
# path of validation videos
videos_test_path = './test/'

videos_small_test_path = './train/'
output_path = './output/'

# path of training maps
maps_path = '/maps/'
# path of training fixation maps
fixs_path = '/fixation/maps/'

frames_path = '/images/'

# path of training images
imgs_path = 'D:/code/attention/staticimages/training/'