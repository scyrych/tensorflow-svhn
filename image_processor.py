import os

import PIL.Image
import numpy as np
import pandas as pd
from scipy.misc import imresize
from scipy.ndimage import imread


def _get_image_size(filepath):
    image = PIL.Image.open(filepath)
    return image.size


def _get_image_sizes(folder):
    image_sizes = []
    images = [img for img in os.listdir(folder)
              if img.endswith('.png')]

    for image in images:
        w, h = _get_image_size(folder + image)
        image_size = {'filename': folder + image,
                      'image_width': w,
                      'image_height': h}
        image_sizes.append(image_size)

    return pd.DataFrame(image_sizes)


def create_images_data_frame(bbox_data_frame, image_data_file):
    if not os.path.isfile(image_data_file):
        train_sizes = _get_image_sizes('data/train/')
        test_sizes = _get_image_sizes('data/test/')
        extra_sizes = _get_image_sizes('data/extra/')

        image_sizes = pd.concat([train_sizes, test_sizes, extra_sizes])

        del train_sizes, test_sizes, extra_sizes

        df = pd.merge(bbox_data_frame, image_sizes, on='filename', how='inner')

        del image_sizes

        # Correct bounding boxes not contained by image
        df.loc[df['x0'] < 0, 'x0'] = 0
        df.loc[df['y0'] < 0, 'y0'] = 0
        df.loc[df['x1'] > df['image_width'], 'x1'] = df['image_width']
        df.loc[df['y1'] > df['image_height'], 'y1'] = df['image_height']

        # Count the number of images by number of digits
        print(df.num_digits.value_counts(sort=False))

        # Keep only images less than 6 digits
        df = df[df.num_digits < 6]

        df.to_csv(image_data_file)
    else:
        df = pd.read_csv(image_data_file)

    return df


def crop_and_resize(image, img_size):
    image_data = imread(image['filename'])
    crop = image_data[image['y0']:image['y1'], image['x0']:image['x1'], :]
    return imresize(crop, img_size)


def rgb_to_gray(images):
    greyscale = np.dot(images, [0.2989, 0.5870, 0.1140])
    return np.expand_dims(greyscale, axis=3)
