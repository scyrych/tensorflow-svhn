import h5py
from helpers import plot_images


def test_data():
    h5f = h5py.File('data/svhn_multi_grey_32.h5', 'r')
    x_train = h5f['train_dataset'][:]
    y_train = h5f['train_labels'][:]
    x_test = h5f['test_dataset'][:]
    y_test = h5f['test_labels'][:]
    x_val = h5f['valid_dataset'][:]
    y_val = h5f['valid_labels'][:]
    h5f.close()
    print("Training", x_train.shape, y_train.shape)
    print('Validation', x_val.shape, y_val.shape)
    print('Test', x_test.shape, y_test.shape)
    plot_images(x_train, 3, 6, y_train)
    plot_images(x_test, 3, 6, y_test)
    plot_images(x_val, 3, 6, y_val)
    single_digit = (y_train != 10).sum(1) == 1
    plot_images(x_train[single_digit], 4, 8, y_train[single_digit])
    five_digits = (y_train != 10).sum(1) == 5
    plot_images(x_train[five_digits], 4, 6, y_train[five_digits])


if __name__ == '__main__':
    test_data()
