import numpy as np
import os
from configs import *
import glob
import random
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions


def process_test_data(test_pth):
    Xims = np.zeros((1, len(test_pth), input_shape[0], input_shape[1], 3))
    X = preprocess_images(test_pth, input_shape[0], input_shape[1])
    Xims[0, 0:len(test_pth), :] = np.copy(X)
    return Xims  #


def generator_data(video_b_s, phase_gen='train', source='EyeTrack'):
    num_frames = input_t

    if phase_gen == 'train':
        # train_pth = os.path.join(train_data_pth, '*', 'images')
        # images_seq = glob.glob(train_pth)
        
        DADA_root =data_root+'EyeTrack/'
        DADA_folders = sorted(os.listdir(DADA_root))
        images_seq = DADA_folders[:dt16_train_valid[0]]

        datas = []
        for image_pth in images_seq:
            images = sorted(glob.glob(DADA_root+image_pth + '/images/*'))
            segs = [xx.replace('images', 'segmentations') for xx in images]
            maps = [xx.replace('images', 'maps') for xx in images]
            # fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            # fixs = [xx.replace('.png', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    # [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> train data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas)-video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas)-video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            # Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                # Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                # Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            # yield [Xims, Xsegs], [Ymaps, Ymaps, Yfixs]  #
            yield [Xims, Xsegs], [Ymaps, Ymaps]  #

    elif phase_gen == 'val':

        DADA_root =data_root+'EyeTrack/'
        DADA_folders = sorted(os.listdir(DADA_root))
        images_seq = DADA_folders[dt16_train_valid[0]:dt16_train_valid[1]]

        datas = []
        for image_pth in images_seq:
            images = sorted(glob.glob(DADA_root+image_pth + '/images/*'))
            segs = [xx.replace('images', 'segmentations') for xx in images]
            maps = [xx.replace('images', 'maps') for xx in images]
            # fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            # fixs = [xx.replace('.png', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    # [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> val data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas) - video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas) - video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            # Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                # Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                # Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            # yield [Xims, Xsegs], [Ymaps, Ymaps, Yfixs]  #
            yield [Xims, Xsegs], [Ymaps, Ymaps]  #

    elif phase_gen == 'score':
        if source=='DReye':
            DADA_root=data_root+'New_DReye/'
        elif source=='DADA':
            DADA_root=data_root+'DADA_test/'
        else:
            DADA_root =data_root+source+'/'
        DADA_folders = sorted(os.listdir(DADA_root))
        images_seq = DADA_folders[data_split[source][1]:data_split[source][2]]

        datas = []
        for image_pth in images_seq:
            images = sorted(glob.glob(DADA_root+image_pth + '/images/*'))
            segs = [xx.replace('images', 'segmentations') for xx in images]
            if source in ['DReye', 'BDDA']:
                maps = [xx.replace('images', 'new_maps') for xx in images]
            else:
                maps = [xx.replace('images', 'maps') for xx in images]
            fixs = [xx.replace('images', 'fixations/map') for xx in images]
            fixs = [xx.replace('.png', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> score data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas) - video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas) - video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            yield [Xims, Xsegs], [Ymaps, Ymaps, Yfixs]  #
            # yield [Xims, Xsegs], [Ymaps, Ymaps]  #


def generator_data_TS(video_b_s, phase_gen='train'):
    num_frames = input_t

    if phase_gen == 'train':
        train_pth = os.path.join(train_data_pth, '*', 'maps')
        maps_seq = glob.glob(train_pth)

        datas = []
        for maps_pth in maps_seq:
            maps = sorted(glob.glob(maps_pth + '/*'))[6:-5]
            images = [xx.replace('maps', 'images') for xx in maps]
            segs = [xx.replace('images', 'semantic') for xx in images]
            fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> train data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas)-video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas)-video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            yield [Xims, Xsegs], [Ymaps, Ymaps, Yfixs]  #
    else:
        val_pth = os.path.join(val_data_pth, '*', 'maps')
        maps_seq = glob.glob(val_pth)

        datas = []
        for maps_pth in maps_seq:
            maps = sorted(glob.glob(maps_pth + '/*'))[6:-5]
            images = [xx.replace('maps', 'images') for xx in maps]
            segs = [xx.replace('images', 'semantic') for xx in images]

            fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> val data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas) - video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas) - video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            yield [Xims, Xsegs], [Ymaps, Ymaps, Yfixs]  #

def generator_data_DR(video_b_s, phase_gen='train'):
    num_frames = input_t

    if phase_gen == 'train':
        train_pth = os.path.join(train_data_pth, '*', 'maps')
        maps_seq = glob.glob(train_pth)

        datas = []
        for maps_pth in maps_seq:
            maps = sorted(glob.glob(maps_pth + '/*'))[:-500]
            images = [xx.replace('maps', 'images') for xx in maps]

            segs = [xx.replace('images', 'semantic') for xx in images]
            segs = [xx.replace('.jpg', '.png') for xx in segs]

            # fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            # fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    # [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> train data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas)-video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas)-video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            # Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                # Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                # Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            yield [Xims, Xsegs], [Ymaps, Ymaps]  #
    else:
        val_pth = os.path.join(val_data_pth, '*', 'maps')
        maps_seq = glob.glob(val_pth)

        datas = []
        for maps_pth in maps_seq:
            maps = sorted(glob.glob(maps_pth + '/*'))[-500:]
            images = [xx.replace('maps', 'images') for xx in maps]

            segs = [xx.replace('images', 'semantic') for xx in images]
            segs = [xx.replace('.jpg', '.png') for xx in segs]

            # fixs = [xx.replace('images', 'fixation/maps') for xx in images]
            # fixs = [xx.replace('.jpg', '.mat') for xx in fixs]

            for jj in range(len(images) - input_t):
                datas.append((
                    images[jj: jj + input_t],
                    segs[jj: jj + input_t],
                    [maps[jj + input_t], ],
                    # [fixs[jj + input_t], ]
                ))
        counts = 0
        print('len -> val data :', len(datas), 'batch_data ->:', len(datas) // video_b_s)
        random.shuffle(datas)

        while True:
            if counts >= (len(datas) - video_b_s):
                random.shuffle(datas)
            counts = counts % (len(datas) - video_b_s)

            Xims = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Xsegs = np.zeros((video_b_s, num_frames, shape_r, shape_c, 3))
            Ymaps = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001
            # Yfixs = np.zeros((video_b_s, shape_r_out, shape_c_out, 1)) + 0.001

            for i in range(0, video_b_s):
                X1 = preprocess_images(datas[counts][0], shape_r, shape_c)
                X2 = preprocess_images(datas[counts][1], shape_r, shape_c)
                Y = preprocess_maps(datas[counts][2], shape_r_out, shape_c_out)
                # Y_fix = preprocess_fixmaps(datas[counts][3], shape_r_out, shape_c_out)

                Xims[i, :] = np.copy(X1)
                Xsegs[i, :] = np.copy(X2)
                Ymaps[i, :] = np.copy(Y[0])
                # Yfixs[i, :] = np.copy(Y_fix[0])
                #  ---------------------------------------------------
                counts += 1

            yield [Xims, Xsegs], [Ymaps, Ymaps]  #

if __name__ == '__main__':
    import tqdm
    generate = generator_data_DR(4)
    for _ in tqdm.tqdm(range(10000)):
        a=next(generate)
        # print(1)

