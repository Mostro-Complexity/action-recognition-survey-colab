import os

import cv2
import h5py
import numpy as np
from keras import backend
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential, load_model
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils, np_utils
from sklearn.metrics import accuracy_score

DATASET = ['MSRAction3D']

N_ACTIONS, N_SUBJECTS, N_INSTANCES = 20, 10, 3
VIDEO_LENGTH, IM_LENGTH, IM_WIDTH = 38, 32, 32


def get_videos_info(validity):
    n_valid = len(validity[validity == 1])
    video_names = np.empty(n_valid, dtype=object)
    action_labels = np.empty(n_valid, dtype=int)
    subject_labels = np.empty(n_valid, dtype=int)
    instance_labels = np.empty(n_valid, dtype=int)
    count = 0

    for a in range(N_ACTIONS):
        for s in range(N_SUBJECTS):
            for e in range(N_INSTANCES):
                if validity[e, s, a] == 1:
                    video_names[count] = "a%02d_s%02d_e%02d_sdepth.mat" % (
                        a + 1, s + 1, e + 1)
                    action_labels[count] = a  # indices(labels) start from 0
                    subject_labels[count] = s
                    instance_labels[count] = e
                    count += 1

    return video_names, action_labels, subject_labels, instance_labels


def split_subjects(video_info, tr_subjects, te_subjects):
    video_names, action_labels, subject_labels, instance_labels = video_info

    tr_subject_ind = np.isin(subject_labels, tr_subjects)
    te_subject_ind = np.isin(subject_labels, te_subjects)

    tr_labels = action_labels[tr_subject_ind]
    te_labels = action_labels[te_subject_ind]

    tr_names = video_names[tr_subject_ind]
    te_names = video_names[te_subject_ind]
    return tr_names, tr_labels, te_names, te_labels


def batches_generator(video_dir, video_names, video_labels, n_classes, batch_size):
    n_sequence = len(video_names)

    X_batch = np.empty((batch_size, 1, VIDEO_LENGTH,
                        IM_LENGTH, IM_WIDTH), dtype=float)
    y_batch = np.empty((batch_size, n_classes), dtype=int)

    while True:
        for i in range(int(np.ceil(n_sequence / batch_size))):
            # print('genetating batch %d' % (i + 1))

            if (i + 1) * batch_size > n_sequence:  # last one batch
                for j in range(n_sequence - i * batch_size):
                    filepath = os.path.join(
                        video_dir, video_names[i * batch_size + j])
                    f = h5py.File(filepath, 'r')
                    video = f['video_array'][:].swapaxes(1, 2)

                    X_batch[j, 0, :, :, :] = video
                    y_batch[j, :] = video_labels[i * batch_size + j, :]

                yield X_batch[:n_sequence - i * batch_size], y_batch[:n_sequence - i * batch_size]
                continue  # end the loop once

            for j in range(batch_size):
                filepath = os.path.join(
                    video_dir, video_names[i * batch_size + j])
                f = h5py.File(filepath, 'r')
                video = f['video_array'][:].swapaxes(1, 2)

                X_batch[j, 0, :, :, :] = video
                y_batch[j, :] = video_labels[i * batch_size + j, :]

            yield X_batch, y_batch


if __name__ == "__main__":
    # Channels order
    backend.set_image_dim_ordering('th')

    for dataset in DATASET:
        video_dir = ''.join(['data/', dataset, '/Depth_Mat'])

        splits_path = ''.join(['data/', dataset, '/tr_te_splits.mat'])
        f = h5py.File(splits_path, 'r')
        tr_subjects = f['tr_subjects'][:].T
        te_subjects = f['te_subjects'][:].T

        skeletal_data_path = ''.join(['data/', dataset, '/skeletal_data.mat'])
        f = h5py.File(skeletal_data_path, 'r')
        validity = f['skeletal_data_validity'][:]

        video_names, action_labels, subject_labels, instance_labels = get_videos_info(
            validity)

        video_info = (video_names, action_labels,
                      subject_labels, instance_labels)

        n_tr_te_splits = tr_subjects.shape[0]

        batch_size = 20
        img_row, img_col, n_frames = 32, 32, 38
        n_classes = len(np.unique(action_labels))

        for i in [0]:
            model = Sequential()

            model.add(Convolution3D(
                128,  # number of kernel
                kernel_dim1=7,  # depth
                kernel_dim2=5,  # rows
                kernel_dim3=5,  # cols
                input_shape=(1, n_frames, img_row, img_col),
                activation='relu',strides=1
            ))
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))
            model.add(Convolution3D(
                36,  # number of kernel
                kernel_dim1=5,  # depth
                kernel_dim2=5,  # rows
                kernel_dim3=5,  # cols
                activation='relu'
            ))
            model.add(MaxPooling3D(pool_size=(2, 2, 2)))

            model.add(Dropout(0.5))

            model.add(Flatten())
            #model.add(Dense(2056, init='normal', activation='linear'))
            # model.add(Dropout(0.5))
            #model.add(Dense(512, init='normal', activation='linear'))
            model.add(Dropout(0.5))
            model.add(Dense(128, init='normal', activation='linear'))
            model.add(Dense(n_classes, init='normal'))

            model.add(Activation('softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='RMSprop', metrics=['mse', 'accuracy'])

            tr_names, tr_labels, te_names, te_labels = split_subjects(
                video_info, tr_subjects[i, :], te_subjects[i, :])

            # convert class vectors to binary class matrices
            tr_labels = np_utils.to_categorical(tr_labels, n_classes)
            te_labels = np_utils.to_categorical(te_labels, n_classes)

            n_steps = np.ceil(tr_labels.shape[0] / batch_size)

            gen = batches_generator(
                video_dir, tr_names, tr_labels, n_classes, batch_size)

            model.fit_generator(generator=gen, epochs=50,
                                steps_per_epoch=n_steps)

            n_steps = np.ceil(te_labels.shape[0] / batch_size)

            gen = batches_generator(
                video_dir, te_names, te_labels, n_classes, batch_size)

            pr_labels = model.predict_generator(generator=gen, steps=n_steps)

            # convert binary class matrices to class vectors
            te_labels = np.argmax(te_labels, axis=-1)
            pr_labels = np.argmax(pr_labels, axis=-1)

            accuracy = accuracy_score(te_labels, pr_labels)
            print('Split %d finished, accuracy:%f' % (i + 1, accuracy))

    pass
