import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix



def plot_cmatrix_wrapper(y_true, y_pred, classes, **kwargs):
    """
    Wrapper function to plot a confusion matrix with plot_confusion_matrix()
    :param y_true:
        [array] Ground truth (correct) target values.
    :param y_pred:
        [array] Estimated targets as returned by a classifier.
    :param classes:
        [list] String labels for each classes identified
    :param kwargs:
        kwargs for plot_confusion_matrix()
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, **kwargs)



def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=plt.cm.Blues, ax=None, verbose=False):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if title is None:
        title = 'Confusion matrix'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        message = "Normalized confusion matrix"
    else:
        message = 'Confusion matrix, without normalization'

    if verbose:
        print(message)
        print(cm)

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.1)
    plt.colorbar(im, cax=cax, orientation='vertical')

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')