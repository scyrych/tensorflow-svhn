import ast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

from bboxer import process_bounding_boxes
from helpers import subtract_mean
from image_processor import create_images_data_frame, crop_and_resize, rgb_to_gray
from svhn_downloader import maybe_download_and_extract

plt.rcParams['figure.figsize'] = (16.0, 4.0)


def _create_dataset(df, img_size):
    x = np.zeros(shape=(df.shape[0], img_size[0], img_size[0], 3), dtype='uint8')
    y = np.full((df.shape[0], 5), 10, dtype=int)

    for i, (index, img) in enumerate(df.iterrows()):
        x[i] = crop_and_resize(img, img_size)

        labels = np.array(ast.literal_eval(img['labels']))
        labels[labels == 10] = 0
        y[i, 0:labels.shape[0]] = labels

    return x, y


def _create_datasets(df, image_size):
    x_train, y_train = _create_dataset(
        df[df.filename.str.contains('train')], image_size)
    print("Training", x_train.shape, y_train.shape)

    x_test, y_test = _create_dataset(
        df[df.filename.str.contains('test')], image_size)
    print("Test", x_test.shape, y_test.shape)

    x_extra, y_extra = _create_dataset(
        df[df.filename.str.contains('extra')], image_size)
    print('Extra', x_extra.shape, y_extra.shape)

    return x_extra, x_test, x_train, y_extra, y_test, y_train


def _random_sample(n, k):
    mask = np.array([True] * k + [False] * (n - k))
    np.random.shuffle(mask)
    return mask


def _create_validation_set(x_extra, x_train, y_extra, y_train):
    train_sample = _random_sample(x_train.shape[0], 4000)
    extra_sample = _random_sample(x_extra.shape[0], 2000)

    x_val = np.concatenate([x_train[train_sample], x_extra[extra_sample]])
    y_val = np.concatenate([y_train[train_sample], y_extra[extra_sample]])

    print('Validation', x_val.shape, y_val.shape)

    x_train = np.concatenate([x_train[~train_sample], x_extra[~extra_sample]])
    y_train = np.concatenate([y_train[~train_sample], y_extra[~extra_sample]])

    print("Training", x_train.shape, y_train.shape)

    return x_train, x_val, y_train, y_val


def _store_data(file_name, x_test, x_train, x_val, y_test, y_train, y_val):
    h5f = h5py.File(file_name, 'w')
    h5f.create_dataset('train_dataset', data=x_train)
    h5f.create_dataset('train_labels', data=y_train)
    h5f.create_dataset('test_dataset', data=x_test)
    h5f.create_dataset('test_labels', data=y_test)
    h5f.create_dataset('valid_dataset', data=x_val)
    h5f.create_dataset('valid_labels', data=y_val)
    h5f.close()


def preprocess_data():
    print("Process bounding boxes - started")

    bbox_file = 'data/bounding_boxes.csv'
    bbox_df = process_bounding_boxes(bbox_file)

    print("Process bounding boxes - completed")

    print("Image processing - started")

    image_data_file = 'data/image_data.csv'
    df = create_images_data_frame(bbox_df, image_data_file)

    print("Image processing - completed")

    print("Creating datasets - started")

    image_size = (32, 32)

    x_extra, x_test, x_train, \
    y_extra, y_test, y_train = _create_datasets(df, image_size)

    del df

    x_train, x_val, y_train, y_val = \
        _create_validation_set(x_extra, x_train, y_extra, y_train)

    del x_extra, y_extra

    print("Creating datasets - completed")

    print("Storing data - started")

    file_name = 'data/svhn_multi_32.h5'

    _store_data(file_name, x_test, x_train,
                x_val, y_test, y_train, y_val)

    print("Data stored")

    x_train = rgb_to_gray(x_train).astype(np.float32)
    x_test = rgb_to_gray(x_test).astype(np.float32)
    x_val = rgb_to_gray(x_val).astype(np.float32)

    file_name = 'data/svhn_multi_grey_32.h5'
    _store_data(file_name, x_test, x_train,
                x_val, y_test, y_train, y_val)

    print("Greyscale data stored")

    print("Storing data - completed")


if __name__ == '__main__':
    maybe_download_and_extract()
    preprocess_data()


def get_datasets():
    h5f = h5py.File('data/svhn_multi_grey_32.h5', 'r')
    x_train = h5f['train_dataset'][:]
    y_train = h5f['train_labels'][:]
    x_val = h5f['valid_dataset'][:]
    y_val = h5f['valid_labels'][:]
    x_test = h5f['test_dataset'][:]
    y_test = h5f['test_labels'][:]
    h5f.close()

    x_train, y_train = shuffle(x_train, y_train)
    x_val, y_val = shuffle(x_val, y_val)

    x_train = subtract_mean(x_train)
    x_val = subtract_mean(x_val)
    x_test = subtract_mean(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test
