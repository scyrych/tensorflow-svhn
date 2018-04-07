import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (16.0, 4.0)


def subtract_mean(a):
    """ Helper function for subtracting the mean of every image
    """
    for i in range(a.shape[0]):
        a[i] -= a[i].mean()
    return a


def plot_images(images, nrows, ncols, cls_true, cls_pred=None):
    # Initialize figure
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2 * nrows))

    # Randomly select nrows * ncols images
    rs = np.random.choice(images.shape[0], nrows * ncols)

    # For every axes object in the grid
    for i, ax in zip(rs, axes.flat):

        # Pretty string with actual number
        true_number = ''.join(str(x) for x in cls_true[i] if x != 10)

        if cls_pred is None:
            title = "True: {0}".format(true_number)
        else:
            # Pretty string with predicted number
            pred_number = ''.join(str(x) for x in cls_pred[i] if x != 10)
            title = "True: {0}, Pred: {1}".format(true_number, pred_number)

        ax.imshow(images[i, :, :, 0], cmap='binary')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()
