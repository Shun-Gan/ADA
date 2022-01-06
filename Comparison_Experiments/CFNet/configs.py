batch_size = 1
nb_train = 500
nb_epoch = 5
nb_videos_val = 100
# score_batch = 100

# -----
smooth_weight = 0.5

node = 16
input_t = 16  # 表示一次输入多少张图片
full_size = (448, 448)
input_shape = (112, 112)
small_output_shape = (int(input_shape[0] / 8), int(input_shape[1] / 8))
out_dims = (None, input_t) + small_output_shape + (512,)
output_dim = (input_t,) + small_output_shape + (1,)

shape_r, shape_c = input_shape[0], input_shape[1]
# shape_r_out, shape_c_out = int(input_shape[0] / 2), int(input_shape[1] / 2)
shape_r_out, shape_c_out = input_shape[0], input_shape[1]


mode = 'score'  # 'train' or 'test' or 'score'
train_data_pth = 'E:/DADA_dataset/train'
val_data_pth = 'E:/DADA_dataset/val'

data_root = '/media/acl/maindisk/0_Codes/unisal-master/data/'
dada_train_valid=[696, 797, 1013]
bdda_train_valid=[927, 1127, 1429]
dreye_train_valid=[379, 406, 780]
dt16_train_valid=[96, 115, 153]

sources=['DADA','BDDA','DReye','EyeTrack']

data_split={
    # 'DADA':[696, 797, 1013],
    'DADA':[0, 1, 220],
    'BDDA':[927, 1127, 1429],
    'DReye':[379, 406, 780],
    'EyeTrack':[96, 115, 153],
}

score_batch={
    'DADA':200,
    'BDDA':10,
    'DReye':10,
    'EyeTrack':10,
}

pre_train_path ={
    'DADA':'./models/DADA_02_2.6074.h5',
    'BDDA':'./models/BDDA_03_1.3438.h5',
    'DReye':'./models/DReye_03_1.4152.h5',
    'EyeTrack':'./models/EyeTrack_04_1.4680.h5',
}

output_dict={'trained source':[],
            'predicted source':[],
            'kld':[],
            'cc':[],
            'sim':[],
            'nss':[],
            'auc_j':[]}

pre_train = False
# pre_train_path = './models/Eyetrack_01_-0.0046.h5'

test_path = './inferences/DADA'
test_save_path = './predicts/'

# test_path = 'E:/UCF/val/'
# test_save_path = 'E:/predicts/UCF/MyNet/'


