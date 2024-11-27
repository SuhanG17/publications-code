import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# save path
save_path = '../figures/abl_uda/'

# define x
x = np.arange(1, 5, 1)
x
width=0.25

# inside the list are average results of 100*0, 200*0, 400*0, 500*0
model_types = ['Uniform \n$N=500$', 'Random \n$N=500$', 'Uniform \n$N=1000$', 'Random \n$N=1000$']

bias_imp_nouda = [0.006, 0.015, 0.002, -0.014]
bias_imp_uda = [0.156, 0.109, 0.115, 0.071]

ese_imp_nouda = [-0.003, 0.000, -0.002, 0.004]
ese_imp_uda = [-0.022, -0.012, 0.008, -0.067]

cover_imp_nouda = [-0.041, -0.024, 0.031, -0.054]
cover_imp_uda = [0.379, 0.193, 0.239, 0.166]

lcover_imp_nouda = [-0.048, -0.030, 0.062, -0.058]
lcover_imp_uda = [0.398, 0.215, 0.251, 0.201]

# Set font
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# plot bias + ESE
fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')

ax.bar(x-0.5*width, bias_imp_nouda, width=width, edgecolor="white", linewidth=0.7, label='Improvement w/o UDA')
ax.bar(x+0.5*width, bias_imp_uda, width=width, edgecolor="white", linewidth=0.7, label='Improvement w/ UDA')

# set tick limit and label 
ax.set(xlim=(0, 5), xticks=np.arange(1, 5, 1), xticklabels=model_types, 
       ylim=(-0.1, 0.3), yticks=np.arange(-0.1, 0.3, 0.05))
ax.tick_params(axis='both', which='major', labelsize=15)

# add legend
ax.legend(loc='upper left', fontsize=15)
# show zero line
ax.axhline(y=0, color='gray', linestyle='--')
# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

def plot_improvement_uda(x, width, improvement_1, improvement_2, legend_position='upper left', save_path='./', figure_name='improvement'):
    fig, ax = plt.subplots(figsize=(12, 9), facecolor='white')

    ax.bar(x-0.5*width, improvement_1, width=width, color='#425bd9', edgecolor="white", linewidth=0.7, label='Improvement w/o UDA')
    ax.bar(x+0.5*width, improvement_2, width=width, color='#258b52', edgecolor="white", linewidth=0.7, label='Improvement w/ UDA')

    # show grid
    plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

    # set tick limit and label 
    ax.set(xlim=(0, 5), xticks=np.arange(1, 5, 1), xticklabels=model_types, 
           ylim=(-0.1, 0.45), yticks=np.arange(-0.1, 0.45, 0.05))
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set axis label
    # ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
    # ax.set_ylabel('Ground Truth $\psi$', fontdict=font)

    # add legend
    ax.legend(loc=legend_position, fontsize=15)
    # show zero line
    ax.axhline(y=0, color='gray', linestyle='--')
    # show grid
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    # plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)

plot_improvement_uda(x, width, bias_imp_nouda, bias_imp_uda, legend_position='upper left',
                     save_path=save_path, figure_name='bias_imp_uda')
plot_improvement_uda(x, width, ese_imp_nouda, ese_imp_uda, legend_position='upper left',
                     save_path=save_path, figure_name='ese_imp_uda')
plot_improvement_uda(x, width, cover_imp_nouda, cover_imp_uda, legend_position='upper right',
                     save_path=save_path, figure_name='cover_imp_uda')
plot_improvement_uda(x, width, lcover_imp_nouda, lcover_imp_uda, legend_position='upper right', 
                     save_path=save_path, figure_name='lcover_imp_uda')

# plot combined plots

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


def plot_individual_improvement_uda(ax, x, width, improvement_1, improvement_2, model_types, 
                                    legend_position, title):
	ax.bar(x-0.5*width, improvement_1, width=width, color='#425bd9', edgecolor="white", linewidth=0.7, label='Improvement w/o UDA')
	ax.bar(x+0.5*width, improvement_2, width=width, color='#258b52', edgecolor="white", linewidth=0.7, label='Improvement w/ UDA')

	# show grid
	# plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

	# set tick limit and label 
	ax.set(xlim=(0, 5), xticks=np.arange(1, 5, 1), xticklabels=model_types, 
			ylim=(-0.1, 0.45), yticks=np.arange(-0.1, 0.45, 0.05))
	ax.tick_params(axis='both', which='major', labelsize=25)
	# # set axis label
	# if show_x_label_and_ticks:
	# 	ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
	# if show_y_label_and_ticks:
	# 	ax.set_ylabel('Ground Truth $\psi$', fontdict=font)
	# ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
	# ax.set_ylabel('Ground Truth $\psi$', fontdict=font)

	# set title
	ax.set_title(title, fontdict={'fontsize':35}, pad=20)

	# add legend
	ax.legend(loc=legend_position, fontsize=25)
	# show zero line
	ax.axhline(y=0, color='gray', linestyle='-.', linewidth=3)
	# show grid
	ax.grid(color = 'green', linestyle = '--', linewidth = 1)

titles = ['Bias', 'ESE', 'Coverage', 'L-Coverage']
legned_position = ['upper right', 'upper right', 'upper right', 'upper right']
improvement_1 = [bias_imp_nouda, ese_imp_nouda, cover_imp_nouda, lcover_imp_nouda]
improvement_2 = [bias_imp_uda, ese_imp_uda, cover_imp_uda, lcover_imp_uda]

# Set font
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 35,
        }

page_width = 516 
fig_w, fig_h = set_size(page_width, fraction=0.5, subplots=(2, 2))
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(fig_w*10, fig_h*10), facecolor='white')
for i in range(4):
	plot_individual_improvement_uda(axs[i//2, i%2], x, width, improvement_1[i], improvement_2[i], model_types,
									legned_position[i], titles[i])
fig.tight_layout(pad=15, h_pad=6, w_pad=6)
fig.savefig(save_path + 'imp_uda_combined' + '.png', dpi=100) 