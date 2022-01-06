from __future__ import division

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from nets import my_net
from configs import *
from data_processing import process_test_data, generator_data
from loss_function import kl_loss, cc_loss, nss_loss, sim, auc_judd
import tqdm
from utilities import postprocess_predictions
from imageio import imread, imsave
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import keras.backend as K

from keras.backend.tensorflow_backend import dtype, set_session


import tensorflow as tf
import keras.backend as K
 
 
def get_flops(model):
    run_meta = tf.RunMetadata()
    opts = tf.profiler.ProfileOptionBuilder.float_operation()
 
    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)
 
    return flops.total_float_ops  # Prints the "flops" of the model.
 
 
# .... Define your model here ....

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))


def schedule_mynet(epoch):
    lr = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-6, 1e-6,
          1e-6, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
# #     # lr = [1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7]
    return lr[epoch]


if __name__ == '__main__':
    if mode == 'train':
        stateful = False
        # 好像错了input_shape = (256, 192)，应该是高宽
        x = Input(batch_shape=(batch_size, input_t, input_shape[0], input_shape[1], 3))
        y = Input(batch_shape=(batch_size, input_t, input_shape[0], input_shape[1], 3))
        m = Model(inputs=[x, y], outputs=my_net(x, y, stateful))
        m.summary()
        print("Compiling My_Net")
        # m.compile(Adam(lr=1e-4), loss=[kl_loss, cc_loss, nss_loss], loss_weights=[1, 0.1, 0.1])  #
        m.compile(Adam(lr=1e-4), loss=[kl_loss, cc_loss], loss_weights=[1, 0.1])  #
        print("Training My_Net")
        if pre_train:
            m.load_weights(pre_train_path)

        m.fit_generator(generator_data(video_b_s=batch_size, phase_gen='train'), nb_train, epochs=nb_epoch,
                        validation_data=generator_data(video_b_s=batch_size, phase_gen='val'),
                        validation_steps=nb_videos_val,
                        callbacks=[  # EarlyStopping(patience=10),
                            ModelCheckpoint('./models/Eyetrack_{epoch:02d}_{val_loss:.4f}.h5', save_best_only=False),])
                            # LearningRateScheduler(schedule=schedule_mynet)])

    elif mode == 'test':
        stateful = True
        x = Input(batch_shape=(1, input_t, input_shape[0], input_shape[1], 3))
        y = Input(batch_shape=(1, input_t, input_shape[0], input_shape[1], 3))
        m = Model(inputs=[x, y], outputs=my_net(x, y, stateful))
        m.summary()
        print(pre_train_path)
        m.load_weights(pre_train_path)
        path = test_path
        save_paths = test_save_path
        seqs = os.listdir(path)
        ss = 0
        for seq in seqs:
            ss += 1
            pre_save_path = os.path.join(save_paths, seq)
            if not os.path.exists(pre_save_path):
                os.makedirs(pre_save_path)
            seq = os.path.join(path, seq, 'images')
            # seq = os.path.join(path, seq)
            imgs = os.listdir(seq)
            imgs.sort()
            imgs = [os.path.join(seq, xx) for xx in imgs]
            segs = [xx.replace('images', 'segmentations') for xx in imgs]

            original_image = imread(imgs[0])
            original_size = original_image.shape[1], original_image.shape[0]

            for i in tqdm.tqdm(range(len(imgs) - input_t), desc='{:03d}/{:03d}'.format(ss, len(seqs))):
                x_in = process_test_data(imgs[i: i + input_t])
                y_in = process_test_data(segs[i: i + input_t])

                pre_map = m.predict(x=[x_in, y_in], batch_size=1)
                pre = pre_map[-1][0, :, :, 0]
                count = 0
                save_name = os.path.basename(imgs[i + input_t])

                res = postprocess_predictions(pre, original_image.shape[0], original_image.shape[1])
                res = res.astype(int)
                imsave(os.path.join(pre_save_path, save_name), res)


    elif mode == 'score':
        stateful = True
        x = Input(batch_shape=(batch_size, input_t, input_shape[0], input_shape[1], 3))
        y = Input(batch_shape=(batch_size, input_t, input_shape[0], input_shape[1], 3))
        m = Model(inputs=[x, y], outputs=my_net(x, y, stateful))

        # print(get_flops(m))

        m.summary()
        # for weight_source in sources:
        for weight_source in ['DADA']:
            print(pre_train_path[weight_source])
            m.load_weights(pre_train_path[weight_source])
            for source in ['DADA']:
                metrics={}
                test_data = generator_data(video_b_s=batch_size, 
                            phase_gen='score', source=source)
                kld_sum, cc_sum, sim_sum, nss_sum, auc_sum = 0, 0, 0, 0, 0
                for idx, batch_data in enumerate(test_data):
                    if idx > score_batch[source]:
                        break
                    [x_in, y_in], [Ymaps, _, Y_fixs] =batch_data
                    pre_map = m.predict(x=[x_in, y_in], batch_size=batch_size)
                    kld_batch = kl_loss(Ymaps, tf.cast(pre_map[-1], tf.float64)) # tf.stack
                    cc_batch = cc_loss(Ymaps,tf.cast(pre_map[-1], tf.float64))
                    nss_batch = nss_loss(Y_fixs, tf.cast(pre_map, tf.float64))
                    with tf.Session() as sess:
                        kld_np = kld_batch.eval()
                        cc_np = cc_batch.eval()
                        nss_np = nss_batch.eval()
                        # pre_map = pre_map[-1].eval()

                    pred_np = np.squeeze(pre_map)
                    fix_np = np.squeeze(Y_fixs)
                    auc_sum += auc_judd(pred_np, fix_np)
                    sim_sum += sim(pre_map[-1], Ymaps)
                    kld_sum += np.mean(kld_np)
                    cc_sum += np.mean(cc_np)
                    cc_sum += np.mean(cc_np)

                # metrics['kld']=kld_sum/idx
                # metrics['cc'] = -cc_sum/idx
                # metrics['sim'] = sim_sum/idx
                output_dict['trained source'].append(weight_source)
                output_dict['predicted source'].append(source)
                output_dict['kld'].append(kld_sum/idx)
                output_dict['sim'].append(sim_sum/idx)
                output_dict['cc'].append(-cc_sum/idx)

            output_pd_set = pd.DataFrame(output_dict)
            output_pd_set.to_csv('./%s_dataset_1.csv'%weight_source)
        output_pd = pd.DataFrame(output_dict)
        output_pd.to_csv('./result_mertrics_1.csv')
        