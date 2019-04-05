import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from astropy.table import Table
from scipy.stats import binned_statistic
from astropy.stats import mad_std

#======================================================================================================================#

class TestResults:

    def __init__(self, results_table):
        self.table = Table.read(results_table, format='ascii')
        self.mask_1v_truepos = np.logical_and(self.table['NCOMP'] == 1, self.table['NCOMP_FIT'] == 1)

        self.mask_2v_good = self.table['NCOMP'] == 2
        self.is2compFit = self.table['NCOMP_FIT'] == 2
        self.mask_2v_truepos = np.logical_and(self.mask_2v_good, self.is2compFit)

        self.table['true_vsep'] = np.abs(self.table['VLSR1'] - self.table['VLSR2'])
        self.table['snr'] = self.table['TMAX']/self.table['RMS']
        self.table['snr-1'] = self.table['TMAX-1'] / self.table['RMS']
        self.table['snr-2'] = self.table['TMAX-2'] / self.table['RMS']

        self.table['sig_min'] = np.nanmin(np.array([self.table['SIG1'], self.table['SIG2']]), axis=0)
        self.table['sig_max'] = np.nanmax(np.array([self.table['SIG1'], self.table['SIG2']]), axis=0)
        self.table['sig_ratio'] = self.table['sig_min']/self.table['sig_max']

        self.table['snr_min'] = np.nanmin(np.array([self.table['snr-1'], self.table['snr-2']]), axis=0)
        self.table['snr_max'] = np.nanmax(np.array([self.table['snr-1'], self.table['snr-2']]), axis=0)
        self.table['snr_ratio'] = self.table['snr_min'] / self.table['snr_max']

        # sort the two components (true errors will be calculated following the sorting)
        self.sort_2comps()


    def plot_cmatrix(self, mask = None, **kwargs):
        if kwargs is None:
            kwargs = {}

        if not 'classes' in kwargs:
            kwargs['classes'] = ["1 comp", "2 comp"]
        if not 'title' in kwargs:
            kwargs['title'] = 'All pixels'
        if not 'normalize' in kwargs:
            kwargs['normalize'] = True

        if mask is None:
            plot_cmatrix_wrapper(self.table['NCOMP'], self.table['NCOMP_FIT'], **kwargs)
        else:
            plot_cmatrix_wrapper(self.table['NCOMP'][mask], self.table['NCOMP_FIT'][mask], **kwargs)


    def plot_cmatrix_wZBins(self, Z_Key, bin_edges, qname="", ncols=2, figsize=(10, 8), **kwargs):

        if not 'fig' in kwargs:
            kwargs['fig'] = plt.figure(figsize=figsize)
            # if no fig is given, return the figure object
            returnFig = True
        else:
            returnFig = False

        fig = kwargs['fig']

        edges = bin_edges
        val = self.table[Z_Key]

        n_plot = len(bin_edges) + 1
        if ncols != 2:
            ncols = int(np.sqrt(n_plot))

        nrows = n_plot/ncols
        if n_plot%ncols !=0:
            nrows = nrows + 1

        classes = ["1 comp", "2 comp"]
        #title = 'All pixels'
        kwargs2 = {'classes': classes, 'normalize': True}#, 'title': title}

        # first bin
        mask = val < edges[0]
        kwargs2['title'] = "{0} < {1}".format(qname, bin_edges[0])
        self.plot_cmatrix(mask, ax=fig.add_subplot(nrows, ncols, 1), **kwargs2)

        # middle bins
        mid_edges = bin_edges[:-1]
        if len(mid_edges) > 0:
            for i in range(len(mid_edges)):
                kwargs2['title'] = "{1} < {0} < {2}".format(qname, edges[i], edges[i + 1])
                mask = val > edges[i]
                mask = np.logical_and(mask, val < edges[i + 1])
                self.plot_cmatrix(mask, ax=fig.add_subplot(nrows, ncols, i+2), **kwargs2)

        # last bin
        kwargs2['title'] = "{0} > {1}".format(qname, edges[-1])
        mask = val > edges[-1]
        self.plot_cmatrix(mask, ax=fig.add_subplot(nrows, ncols, n_plot), **kwargs2)

        fig.subplots_adjust(wspace=0.1, hspace=0.7)

        if returnFig:
            return fig




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


    def plot_error(self, X_Key, Y_Key, mask=None, range=None, ax=None, bins=30):
        '''
        Plot values specified in the X_Key, such as errors, as a function of  values specified in the Y_Key
        X_Key and Y_Key can be lists of the same length and their values will be binned together
        :param X_Key:
        :param Y_Key:
        :param mask:
        :param range:
        :param ax:
        :param bins:
        :return:
        '''

        XList = []
        YList = []

        if isinstance(X_Key, str):
            X_Key = [X_Key]

        if isinstance(Y_Key, str):
            Y_Key = [Y_Key]

        for xkey, ykey in zip(X_Key, Y_Key):
            X = self.table[xkey]
            Y = self.table[ykey]

            if mask is not None:
                X = X[mask]
                Y = Y[mask]

            XList.append(X)
            YList.append(Y)

        plot_err(np.array(XList).ravel(), np.array(YList).ravel(), Err=None, bins=bins, range=range, ax=ax, title=None)


    def sort_2comps(self):
        # currently sorted proximity to the true vlsr value (which has bias that I'll need to fix

        def sort_comps2(swapmask, compAry):
            compAry[0][swapmask], compAry[1][swapmask] = compAry[1][swapmask], compAry[0][swapmask]

        VLSR_m = np.array([self.table['VLSR1_FIT'], self.table['VLSR2_FIT']])#.copy()
        VLSR_t = np.array([self.table['VLSR1'], self.table['VLSR2']])#.copy()
        diffVLSR1 = np.abs(VLSR_t[0] - VLSR_m[0])
        diffVLSR2 = np.abs(VLSR_t[0] - VLSR_m[1])

        swapmask = diffVLSR1 > diffVLSR2
        # only apply to where two component true postives exists
        mask = np.logical_and(self.mask_2v_good, self.is2compFit)
        swapmask[~mask] = False

        # note: the following list is not complete yet, it only sorts fitted vlsr & sigma
        sortList = []
        sortList.append([self.table['VLSR1_FIT'], self.table['VLSR2_FIT']])
        sortList.append([self.table['SIG1_FIT'], self.table['SIG2_FIT']])
        sortList.append([(self.table['eVLSR1_FIT']), (self.table['eVLSR2_FIT'])])
        sortList.append([(self.table['eSIG1_FIT']), (self.table['eSIG2_FIT'])])
        sortList.append([(self.table['TMAX-1']), (self.table['TMAX-2'])])
        sortList.append([(self.table['snr-1']), (self.table['snr-2'])])

        #sortList.append()

        for i, q in enumerate(sortList):
            sort_comps2(swapmask, sortList[i])

        # recalculate some values
        self.table['true_vErr1'] = self.table['VLSR1'] - self.table['VLSR1_FIT']
        self.table['true_vErr2'] = self.table['VLSR2'] - self.table['VLSR2_FIT']



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

    # mean value of isTruePos (boolean) is the fraction of true postive id's out of all the id's
    mean = binned_statistic(x=X.copy(), values=isTruePos.copy(), statistic=np.nanmean, bins=nbins, range=range)

    bin_edges = mean.bin_edges
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    ax.plot(bin_centers, mean.statistic, **kwargs)

    return ax


def plot_medNStd(X, Y, bins=30, range=None, ax=None, title=None):

    med = binned_statistic(x=X, values=Y, statistic='median', bins=bins, range=range)
    std = binned_statistic(x=X, values=Y, statistic=mad_std, bins=bins, range=range)

    ax = plot_fillscat(med.statistic, std.statistic, std.bin_edges, ax=ax, title=title)
    return ax


def plot_err(X, Y, Err=None, bins=30, range=None, ax=None, title=None):

    if ax is None:
        ax = plt.subplot(1, 1, 1)

    std = binned_statistic(x=X, values=Y, statistic=mad_std, bins=bins, range=range)

    bin_edges = std.bin_edges
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    ax.plot(bin_centers, std.statistic, '-o', lw=3)

    if Err is not None:
        err = binned_statistic(x=X, values=Err, statistic='median', bins=bins, range=range)
        ax.plot(bin_centers, err.statistic, '-o', lw=3)

    ax.set_ylabel('Error')
    #ax.set_xlabel('SNR')
    if title is not None:
        ax.set_title(title)

    return ax


#======================================================================================================================#
# analysis (non-plotting) functions



#======================================================================================================================#
# generic plotting functions

def plot_fillscat(med, std, bin_edges, ax=None, title=None):
    if ax is None:
        ax = plt.subplot(1, 1, 1)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    ax.plot(bin_centers, med, '-o', lw=3, color='0.5')
    ax.fill_between(bin_centers, med - std, med + std, alpha=0.25, color=colors[1])
    ax.set_ylabel('True - fit')
    ax.set_xlabel('SNR')
    if title is not None:
        ax.set_title(title)

    return ax



