import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from astropy.table import Table

#======================================================================================================================#

class TestResults:

    def __init__(self, results_table):
        self.table = Table.read(results_table, format='ascii')
        self.mask_2v_good = self.table['NCOMP'] == 2
        self.is2compFit = self.table['NCOMP_FIT'] == 2

        self.table['true_vsep'] = np.abs(self.table['VLSR1'] - self.table['VLSR2'])
        self.table['snr'] = self.table['TMAX']/self.table['RMS']
        self.table['sig_min'] = np.nanmin(np.array([self.table['SIG1'], self.table['SIG2']]), axis=0)
        self.table['sig_max'] = np.nanmax(np.array([self.table['SIG1'], self.table['SIG2']]), axis=0)
        self.table['sig_ratio'] = self.table['sig_min']/self.table['sig_max']


    def plot_cmatrix(self, **kwargs):
        if kwargs is None:
            kwargs = {}

        if not 'classes' in kwargs:
            kwargs['classes'] = ["1 comp", "2 comp"]
        if not 'title' in kwargs:
            kwargs['title'] = 'All pixels'
        if not 'normalize' in kwargs:
            kwargs['normalize'] = True

        plot_cmatrix_wrapper(self.table['NCOMP'], self.table['NCOMP_FIT'], **kwargs)


    def plot_success_rate(self, X_Key, mask=None, bins=10, linestyle=None, lw=None, **kwargs):

        if linestyle is None:
            kwargs['linestyle'] = '-'
        if lw is None:
            kwargs['lw'] = 3

        X = self.table[X_Key]
        Y = self.is2compFit

        if mask is None:
            mask = self.mask_2v_good
        else:
            mask = np.logical_and(self.mask_2v_good, mask)

        plot_success_rate(X[mask], Y[mask], bins, **kwargs)#linestyle='-', lw=3)


    def plot_success_rate_wZBins(self, X_Key, Z_Key, bin_edges, bins=10, qname="", **kwargs):

        if kwargs['ax'] is None:
            fig, kwargs['ax'] = plt.subplots(1, 1, figsize=(6, 4))
        ax = kwargs['ax']

        edges = bin_edges
        val = self.table[Z_Key]

        # first bin
        mask = np.logical_and(self.mask_2v_good, val < edges[0])
        self.plot_success_rate(X_Key, mask, bins, **kwargs)
        legtext = ["{0} < {1}".format(qname, bin_edges[0])]

        # middle bins
        mid_edges = bin_edges[:-1]
        if len(mid_edges) > 0:
            for i in range(len(mid_edges)):
                mask = np.logical_and(self.mask_2v_good, val > edges[i])
                mask = np.logical_and(mask, val < edges[i + 1])
                self.plot_success_rate(X_Key, mask, bins, **kwargs)
                legtext.append("{1} < {0} < {2}".format(qname, edges[i], edges[i + 1]))

        # last bin
        mask = np.logical_and(self.mask_2v_good, val > edges[-1])
        self.plot_success_rate(X_Key, mask, bins, **kwargs)
        legtext.append("{0} > {1}".format(qname, edges[-1]))

        # ax.set_title('Identifying 2 components')
        ax.set_ylabel('Fraction of True Postives')
        ax.set_xlabel('True $\Delta \mathrm{v}_\mathrm{LSR}$ (km s$^{-1}$)')

        if qname is not "":
            ax.legend(legtext, frameon=False)


#======================================================================================================================#
# For plotting confusion matrix

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

#======================================================================================================================#

def plot_success_rate(X, isTruePos, nbins=30, range=None, ax=None, **kwargs):
    """
    Plots the success rate of true positive identifications as a function of X values
    :param X:
        [array] Parameter to be plotted on the x-axis
    :param Y:
        [array] Boolean values of whether or not a true positive has been successfully identified
    :param nbins:
        [int] Number of bins to bin the X into
    :param range:
        [int] Range of values for X to be binned into
    :param ax:
    :param title:
    :param kwargs:
    :return:
    """
    from scipy.stats import binned_statistic

    # mean value of isTruePos (boolean) is the fraction of true postive id's out of all the id's
    mean = binned_statistic(x=X.copy(), values=isTruePos.copy(), statistic=np.nanmean, bins=nbins, range=range)

    bin_edges = mean.bin_edges
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(bin_centers, mean.statistic, **kwargs)

    return ax


