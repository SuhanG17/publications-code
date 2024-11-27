import matplotlib.pyplot as plt
import numpy as np

# save path
save_path = '../figures/abl_ts/'


x_label = np.arange(1, 11, 1)
x_label

titles = ['Bias', 'ESE', 'Coverage', 'L-Coverage']
model_types = ['CC Uniform', 'CW Uniform', 'WC Uniform', 'Flex Uniform',
               'CC Random',  'CW Random', 'WC Random', 'Flex Random']
marker_types = ['o', 's', 'D', 'v', 'P', '*', 'X', 'H']

bias_improvement = [[-0.237, -0.049, 0.102, 0.133, 0.103, 0.119, 0.120, 0.112, 0.146, 0.102],
                    [-0.136, 0.089, 0.152, 0.185, 0.202, 0.209, 0.204, 0.160, 0.227, 0.192],
                    [-0.223, -0.044, 0.116, 0.147, 0.112, 0.130, 0.132, 0.123, 0.157, 0.112],
                    [-0.250, 0.005, 0.105, 0.157, 0.124, 0.150, 0.115, 0.142, 0.132, 0.088],
                    [-0.187, 0.042, 0.085, 0.095, 0.047, 0.077, 0.050, 0.095, 0.096, 0.058],
                    [-0.103, 0.097, 0.051, 0.127, 0.122, 0.110, 0.131, 0.132, 0.133, 0.137],
                    [-0.173, 0.049, 0.085, 0.097, 0.051, 0.078, 0.054, 0.098, 0.100, 0.061],
                    [-0.153, 0.030, 0.078, 0.116, 0.102, 0.101, 0.088, 0.103, 0.087, 0.080]]

ese_improvement = [[0.035, -0.100, -0.015, -0.021, -0.026, -0.012, -0.028, -0.052, 0.023, -0.018],
                   [0.086, -0.065, -0.046, -0.014, 0.036, 0.001, 0.011, -0.048, 0.017, -0.022],
                   [0.021, -0.110, -0.017, -0.014, -0.027, -0.009, -0.027, -0.050, 0.022, -0.023],
                   [0.011, -0.085, -0.057, -0.039, -0.037, -0.022, -0.058, -0.053, -0.015, -0.059],
                   [0.047, -0.038, 0.006, -0.007, -0.030, -0.023, -0.014, -0.045, 0.028, -0.004],
                   [0.042, -0.023, -0.053, -0.013, 0.021, -0.007, -0.001, -0.017, 0.007, -0.040],
                   [0.035, -0.039, 0.003, -0.006, -0.029, -0.023, -0.013, -0.048, 0.030, -0.006],
                   [0.023, -0.044, -0.052, -0.021, -0.027, -0.018, -0.011, -0.036, 0.006, -0.023]]

cover_improvement = [[-0.121, 0.191, 0.214, 0.407, 0.326, 0.316, 0.344, 0.335, 0.388, 0.300],
                     [-0.144, 0.237, 0.191, 0.353, 0.402, 0.423, 0.356, 0.330, 0.414, 0.377],
                     [-0.100, 0.188, 0.349, 0.381, 0.288, 0.274, 0.289, 0.298, 0.346, 0.268],
                     [-0.146, 0.168, 0.175, 0.375, 0.286, 0.309, 0.247, 0.311, 0.291, 0.204],
                     [-0.181, -0.207, 0.077, 0.111, 0.053, 0.072, -0.012, 0.118, 0.114, 0.005],
                     [-0.174, -0.132, 0.142, 0.142, 0.133, 0.182, 0.181, 0.167, 0.191, 0.188],
                     [-0.130, -0.196, 0.077, 0.086, 0.028, 0.063, 0.026, 0.100, 0.125, 0.002],
                     [0.160, 0.077, 0.360, 0.432, 0.314, 0.347, 0.332, 0.360, 0.293, 0.309]]

lcover_improvement = [[-0.116, 0.212, 0.225, 0.432, 0.339, 0.332, 0.365, 0.363, 0.414, 0.326],
                      [-0.147, 0.258, 0.212, 0.360, 0.419, 0.439, 0.365, 0.353, 0.428, 0.396],
                      [-0.098, 0.216, 0.368, 0.402, 0.305, 0.289, 0.321, 0.314, 0.367, 0.286],
                      [-0.147, 0.191, 0.184, 0.400, 0.298, 0.325, 0.265, 0.319, 0.311, 0.235],
                      [-0.167, -0.209, 0.081, 0.132, 0.079, 0.098, 0.009, 0.128, 0.114, 0.018],
                      [-0.177, -0.125, 0.151, 0.158, 0.133, 0.207, 0.191, 0.179, 0.218, 0.198],
                      [-0.095, -0.188, 0.093, 0.118, 0.068, 0.093, 0.054, 0.128, 0.147, 0.023],
                      [0.182, 0.088, 0.393, 0.453, 0.342, 0.360, 0.356, 0.375, 0.312, 0.346]]

def plot_individual_imp(ax, x, improvement, model_types, marker_types,
                        font, title, show_x_label=False):
    for imp, model_type, marker in zip(improvement, model_types, marker_types):
        string = marker + '-'
        ax.plot(x, imp, string, ms=10, linewidth=3, label=model_type)
    # set grid
    ax.grid(color = 'green', linestyle = '--', linewidth = 0.5)
    # set tick limit and label 
    ax.set(xlim=(0, 11), xticks=np.arange(0, 11, 1),
    ylim=(-0.4, 0.5), yticks=np.arange(-0.4, 0.5, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=20)
    # set axis label
    if show_x_label:
        ax.set_xlabel('Time Steps', fontdict=font)
    # ax.set_ylabel('Bias', fontdict=font)
    # add legend
    ax.legend(ncol=4, loc='lower left', fontsize=20)
    # show zero line
    ax.axhline(y=0, color='gray', linestyle='-.')
    # set title
    ax.set_title(title, fontdict=font)

# Set font
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 35,
        }

fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')
plot_individual_imp(ax, x_label, bias_improvement, model_types, marker_types,
                    font, 'Bias', show_x_label=True)


# import matplotlib as mpl
# mpl._version.get_versions()

# plot combined improvement
# source: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
# get latex \textwidth: https://tex.stackexchange.com/questions/39383/determine-text-width
def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)	


metrics_improvement = [bias_improvement, ese_improvement, cover_improvement, lcover_improvement]

page_width = 516 
fig_w, fig_h = set_size(page_width, fraction=0.5, subplots=(2, 2))
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_w*10, fig_h*10), facecolor='white')
for i in range(4):
    show_x_label = i//2 == 1
    plot_individual_imp(axs[i//2, i%2], x_label, metrics_improvement[i], model_types, marker_types,
                        font, titles[i], show_x_label=show_x_label)
fig.tight_layout(pad=15, h_pad=6, w_pad=6)
fig.savefig(save_path + 'imp_ts_combined' + '.png', dpi=100) 