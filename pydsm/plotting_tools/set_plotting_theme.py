import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np

def set_theme():
    theme_folder = os.path.dirname(os.path.realpath(__file__))
    style_path = os.path.join(theme_folder,  'theme/.matplotlib/mpl_configdir/stylelib/gadfly.mplstyle')
    plt.style.use(style_path)

    # plt.rcParams.update({
    #     "text.usetex": True,
    #     "font.family": "sans-serif",
    #     "font.sans-serif": "Helvetica",
    # })

    SMALL_SIZE = 8*1.9
    MEDIUM_SIZE = 10*1.9
    BIGGER_SIZE = 12*2


    plt.rc('font', size=SMALL_SIZE)             # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)        # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)       # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)      # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)      # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)      # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE*2)   # fontsize of the figure title

colors = [(0.0, 0.7450980392156863, 1.0),
          (0.8313725490196079, 0.792156862745098, 0.22745098039215686),
          (1.0, 0.42745098039215684, 0.6823529411764706),
          (0.403921568627451, 0.8823529411764706, 0.7098039215686275),
          (0.9215686274509803, 0.6745098039215687, 0.9803921568627451),
          (0.6196078431372549, 0.6196078431372549, 0.6196078431372549),
          (0.9450980392156862, 0.596078431372549, 0.5568627450980392),
          (0.36470588235294116, 0.6941176470588235, 0.35294117647058826),
          (0.8862745098039215, 0.5215686274509804, 0.26666666666666666),
          (0.3215686274509804, 0.7215686274509804, 0.6666666666666666)]



def make_colormap(seq):
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def diverge_map(high=colors[0], low=colors[2]):
    c = mcolors.ColorConverter().to_rgb
    low = c(low)
    high = c(high)

    if isinstance(low, str): low = c(low)
    if isinstance(high, str): high = c(high)
    return make_colormap([low, c('yellow'), 1/3, c('yellow'), high, 2/3, high, c('darkviolet')])


### PLOT SCALING CONFIG
def plot_scaling(scale=9.5):
    SMALL_SIZE = 8*1.9/scale
    MEDIUM_SIZE = 10*2.5/scale
    BIGGER_SIZE = 12*2.5/scale

    mpl.rcParams['xtick.major.size'] = 1/scale
    mpl.rcParams['xtick.major.width'] = 1/scale
    mpl.rcParams['ytick.major.size'] = 1/scale
    mpl.rcParams['ytick.major.width'] = 1/scale
    mpl.rcParams['xtick.minor.size'] = 0
    mpl.rcParams['xtick.minor.width'] = 0
    mpl.rcParams['patch.linewidth'] = 1/scale

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize


def align_yaxis(ax1, ax2):
    y_lims = np.array([ax.get_ylim() for ax in [ax1, ax2]])

    # force 0 to appear on both axes, comment if don't need
    y_lims[:, 0] = y_lims[:, 0].clip(None, 0)
    y_lims[:, 1] = y_lims[:, 1].clip(0, None)

    # normalize both axes
    y_mags = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_mags

    # find combined range
    y_new_lims_normalized = np.array([np.min(y_lims_normalized), np.max(y_lims_normalized)])

    # denormalize combined range to get new axes
    new_lim1, new_lim2 = y_new_lims_normalized * y_mags
    ax1.set_ylim(new_lim1)
    ax2.set_ylim(new_lim2)