import matplotlib.pyplot as plt
import numpy as np

x_label = np.array([0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 
           0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 
           0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950])

# x = np.arange(1, len(x_label)+1, 1)

# mode=all; all_time=9
# LR
y_lr = [-0.004, 0.008, 0.111, 0.006, 0.170, 0.213, 
        0.236, 0.281, 0.290, 0.285, 0.244, 0.298, 
        0.291, 0.299, 0.324, 0.333, 0.354, 0.360, 0.368]
y_err_lr = [0.037, 0.047, 0.059, 0.069, 0.085, 0.093, 
            0.111, 0.127, 0.140, 0.153, 0.173, 0.196, 
            0.217, 0.240, 0.257, 0.276, 0.296, 0.316, 0.321]
cover_lr = [0.533, 0.400, 0.167, 0.467, 0.133, 0.100, 
            0.133, 0.133, 0.133, 0.167, 0.167, 0.167, 
            0.167, 0.167, 0.167, 0.167, 0.167, 0.200, 0.200]

# DL
y_dl = [-0.103, -0.070, 0.032, -0.042, 0.180, 0.131,
        0.112, 0.198, 0.134, 0.068, 0.082, 0.094, 
        0.057, 0.105, 0.056, 0.076, 0.076, 0.055, 0.018]
y_err_dl = [0.080, 0.076, 0.130, 0.172, 0.148, 0.189,
            0.218, 0.189, 0.207, 0.119, 0.159, 0.119,
            0.102, 0.219, 0.104, 0.152, 0.159, 0.156, 0.073]
cover_dl = [0.267, 0.267, 0.300, 0.300, 0.300, 0.567, 
            0.667, 0.533, 0.800, 0.733, 0.733, 0.833, 
            0.733, 0.633, 0.767, 0.733, 0.567, 0.800, 0.767]

# Set font
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# plot bias + ESE
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
# ax.plot(x, y, 'o-', linewidth=2)
ax.errorbar(x_label-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', label='LR')
ax.errorbar(x_label+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', label='DL')
# show zero line
ax.axhline(y=0, color='gray', linestyle='--')
# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

# set tick limit and label 
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
       ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
ax.tick_params(axis='both', which='major', labelsize=15)
# set axis label
ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
ax.set_ylabel('Bias', fontdict=font)
# add legend
ax.legend(loc='upper left', fontsize=15)

plt.show()
fig.savefig('bias_ese.png', dpi=300)

def plot_bias_ese(x, y_lr, y_dl, y_err_lr, y_err_dl, 
                  font, save_path='./', figure_name='bias_ese'):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    # ax.plot(x, y, 'o-', linewidth=2)
    ax.errorbar(x-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', color='#ed36d1', label='LR')
#     ax.errorbar(x+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', color='#4d992e',label='DL')
    ax.errorbar(x+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', color='#0021ff',label='DL')
    # show zero line
    ax.axhline(y=0, color='gray', linestyle='-.', linewidth=2)
    # show grid
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    # set tick limit and label 
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
    ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set axis label
    ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
    ax.set_ylabel('Bias', fontdict=font)
    # add legend
    ax.legend(loc='upper left', fontsize=15)

    # plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)

plot_bias_ese(x_label, y_lr, y_dl, y_err_lr, y_err_dl, font, save_path='./', figure_name='bias_ese')


# # plot cover
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
# ax.plot(x_label, cover_lr, 'D-')
# ax.plot(x_label, cover_dl, 'D-')

# ax.stem(x_label, cover_lr, linefmt='C0-', markerfmt='C0D', basefmt='C0-')
# ax.stem(x_label, cover_dl, linefmt='C1-', markerfmt='C1D', basefmt='C0-')

ax.bar(x_label-0.01, cover_lr, width=0.02, edgecolor="white", linewidth=0.7, label='LR')
ax.bar(x_label+0.01, cover_dl, width=0.02, edgecolor="white", linewidth=0.7, label='DL')

# show grid
plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

# set tick limit and label 
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
       ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
ax.tick_params(axis='both', which='major', labelsize=15)
# set axis label
ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
ax.set_ylabel('Coverage', fontdict=font)
# add legend
ax.legend(loc='upper left', fontsize=15)

plt.show()
fig.savefig('cover.png', dpi=300)

def plot_cover(x, cover_lr, cover_dl, font, save_path='./', figure_name='cover'):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')

    ax.bar(x-0.01, cover_lr, width=0.02, color='#425bd9', edgecolor="white", linewidth=0.7, label='LR')
    ax.bar(x+0.01, cover_dl, width=0.02, color='#258b52', edgecolor="white", linewidth=0.7, label='DL')

    # show grid
    plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

    # set tick limit and label 
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
           ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set axis label
    ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
    ax.set_ylabel('Coverage', fontdict=font)
    # add legend
    ax.legend(loc='upper left', fontsize=15)

    # plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)

plot_cover(x_label, cover_lr, cover_dl, font, save_path='./', figure_name='cover')


# plot with double y axis
fig, ax1 = plt.subplots(figsize=(16, 9), facecolor='white')

color = 'tab:red'
# ax.plot(x, y, 'o-', linewidth=2)
# # ax1.errorbar(x_label+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', color='#4d992e',label='DL_Bias_ESE')
ax1.errorbar(x_label-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', 
             color='#ed36d1', label='LR_Bias_ESE')
# # ax1.errorbar(x_label+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', 
                # color='#4d992e',label='DL_Bias_ESE')
ax1.errorbar(x_label+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', 
             color='#0021ff',label='DL_Bias_ESE')
# show zero line
ax1.axhline(y=0, color='grey', linestyle='-.', linewidth=2)

# set tick limit and label 
ax1.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
        ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
ax1.tick_params(axis='both', which='major', labelsize=15)
ax1.tick_params(axis='y', labelcolor=color)

# set axis label
ax1.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
ax1.set_ylabel('Bias', fontdict=font, color=color)

# add legend
# ax1.legend(loc='upper left', fontsize=15)

ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
color = 'tab:blue'

# ax2.bar(x_label-0.01, cover_lr, width=0.02, color='#425bd9', edgecolor="grey", linewidth=0.7, alpha=0.3, label='LR')
# ax2.bar(x_label+0.01, cover_dl, width=0.02, color='#258b52', edgecolor="grey", linewidth=0.7, alpha=0.3, label='DL')

ax2.bar(x_label-0.01, cover_lr, width=0.02, fill=False, hatch='///', 
        edgecolor="#425bd9", linewidth=1, alpha=0.5, label='LR_Coverage')
ax2.bar(x_label+0.01, cover_dl, width=0.02, fill=False, hatch='..', 
        edgecolor="#258b52", linewidth=1, alpha=0.5, label='DL_Coverage')

# set tick limit and label
ax2.set(ylim=(0, 1), yticks=np.arange(0, 1, 0.05))
ax2.tick_params(axis='y', labelcolor=color)
ax2.tick_params(axis='both', which='major', labelsize=15)

# set axis label
ax2.set_ylabel('Coverage', fontdict=font, color=color)  # we already handled the x-label with ax1

# add legend
# ax2.legend(loc='upper left', fontsize=15)

fig.legend(loc='center', bbox_to_anchor=[0.14, 0.89],
           fontsize=15)

# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

def plot_bias_ese_with_cover(x, y_lr, y_dl, y_err_lr, y_err_dl,
                             cover_lr, cover_dl,
                             font, save_path='./', figure_name='bias_ese_cover'):
	# plot with double y axis
	fig, ax1 = plt.subplots(figsize=(16, 9), facecolor='white')

	# plot figure 1 with left y axis
	color = 'tab:red'
	# ax.plot(x, y, 'o-', linewidth=2)
	ax1.errorbar(x-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', color='#ed36d1', label='LR_Bias_ESE')
	ax1.errorbar(x+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', color='#0021ff',label='DL_Bias_ESE')
	# show zero line
	ax1.axhline(y=0, color='grey', linestyle='-.', linewidth=2)

	# set tick limit and label 
	ax1.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
			ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
	ax1.tick_params(axis='both', which='major', labelsize=15)
	ax1.tick_params(axis='y', labelcolor=color)

	# set axis label
	ax1.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
	ax1.set_ylabel('Bias', fontdict=font, color=color)

	# plot figure 2 with right y axis
	ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
	color = 'tab:blue'

	ax2.bar(x-0.01, cover_lr, width=0.02, fill=False, hatch='///', 
			edgecolor="#425bd9", linewidth=1, alpha=0.5, label='LR_Coverage')
	ax2.bar(x+0.01, cover_dl, width=0.02, fill=False, hatch='..', 
			edgecolor="#258b52", linewidth=1, alpha=0.5, label='DL_Coverage')

	# set tick limit and label
	ax2.set(ylim=(0, 1), yticks=np.arange(0, 1, 0.05))
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.tick_params(axis='both', which='major', labelsize=15)

	# set axis label
	ax2.set_ylabel('Coverage', fontdict=font, color=color)  # we already handled the x-label with ax1

	# add legend
	# for single plot
	fig.legend(loc='center', bbox_to_anchor=[0.14, 0.89],
	        fontsize=15)
    # # for combined plot
	# ax1.legend(loc='center left', bbox_to_anchor=[0.0, 0.95], fontsize=15)
	# ax2.legend(loc='center left', bbox_to_anchor=[0.0, 0.85], fontsize=15)

	# show grid
	plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

	# save figure
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	# plt.show()
	save_path = save_path + figure_name + '.png'
	fig.savefig(save_path, dpi=300)

plot_bias_ese_with_cover(x_label, y_lr, y_dl, y_err_lr, y_err_dl, cover_lr, cover_dl, font, save_path='./', figure_name='bias_ese_cover')
    


import pandas as pd
networktype = ['10010', '10020', '10030', '10040',
               '20010', '20020', '20030', '20040',
               '40010', '40020', '40030', '40040',
               '50010', '50020', '50030', '50040',
               '60010', '60020', '60030', '60040',
               '70010', '70020', '70030', '70040']
titles = ['CC: uniform 500', 'CW: uniform 500', 'WC: uniform 500', 'Flexible: uniform 500',
		  'CC: random 500', 'CW: random 500', 'WC: random 500', 'Flexible: random 500',
		  'CC: uniform 1000', 'CW: uniform 1000', 'WC: uniform 1000', 'Flexible: uniform 1000',
		  'CC: random 1000', 'CW: random 1000', 'WC: random 1000', 'Flexible: random 1000',
		  'CC: uniform 2000', 'CW: uniform 2000', 'WC: uniform 2000', 'Flexible: uniform 2000',
		  'CC: random 2000', 'CW: random 2000', 'WC: random 2000', 'Flexible: random 2000']

# csv dir
# # mode=all; time=9
# dir_path = '../results_csv/mode_all/'
# ts='9'
# save_path='../figures/mode_all/'
# # mode=top_50; time=10
dir_path = '../results_csv/mode_top_50/'
ts='10'
save_path='../figures/mode_top_50/'

# load csv data for LR
lr_bias = pd.read_csv(dir_path+'LR_bias_ts'+ts+'.csv', header=None)
lr_bias_array = lr_bias.to_numpy()
lr_bias_array.shape

lr_ese = pd.read_csv(dir_path+'LR_ese_ts'+ts+'.csv', header=None)
lr_ese_array = lr_ese.to_numpy()
lr_ese_array.shape

lr_cover = pd.read_csv(dir_path+'LR_cover_ts'+ts+'.csv', header=None)
lr_cover_array = lr_cover.to_numpy()
lr_cover_array.shape

lr_lcover = pd.read_csv(dir_path+'LR_lcover_ts'+ts+'.csv', header=None)
lr_lcover_array = lr_lcover.to_numpy()
lr_lcover_array.shape


# load csv data for DL
dl_bias = pd.read_csv(dir_path+'DL_bias_ts'+ts+'.csv', header=None)
dl_bias_array = dl_bias.to_numpy()
dl_bias_array.shape

dl_ese = pd.read_csv(dir_path+'DL_ese_ts'+ts+'.csv', header=None)
dl_ese_array = dl_ese.to_numpy()
dl_ese_array.shape

dl_cover = pd.read_csv(dir_path+'DL_cover_ts'+ts+'.csv', header=None)
dl_cover_array = dl_cover.to_numpy()
dl_cover_array.shape

dl_lcover = pd.read_csv(dir_path+'DL_lcover_ts'+ts+'.csv', header=None)
dl_lcover_array = dl_lcover.to_numpy()
dl_lcover_array.shape

# plot figures
for i, network_string in enumerate(networktype):
    # separate plots
#     plot_bias_ese(x_label, lr_bias_array[i], dl_bias_array[i], lr_ese_array[i], dl_ese_array[i], 
#                   font, save_path=save_path, figure_name='bias_ese_'+network_string)
#     plot_cover(x_label, lr_cover_array[i], dl_cover_array[i], 
#                font, save_path=save_path, figure_name='cover_'+network_string)
    plot_cover(x_label, lr_lcover_array[i], dl_lcover_array[i], 
               font, save_path=save_path, figure_name='lcover_'+network_string)
    # combined plots
    plot_bias_ese_with_cover(x_label, lr_bias_array[i], dl_bias_array[i], lr_ese_array[i], dl_ese_array[i],
                            lr_cover_array[i], dl_cover_array[i], font, save_path=save_path, figure_name='bias_ese_cover_'+network_string)


# plot all sceanrios in one plot

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

fig_width, fig_height = set_size(516.0, fraction=1., subplots=(6, 4))
fig_width
fig_height

def plot_individual_scenario(ax, x, y_lr, y_dl, y_err_lr, y_err_dl, 
							cover_lr, cover_dl, font, title,
                            show_x_label_and_ticks=False, show_left_y_label_and_ticks=False, show_right_y_label_and_ticks=False):
	ax1=ax
	# plot figure 1 with left y axis
	color = 'tab:red'
	# ax.plot(x, y, 'o-', linewidth=2)
	ax1.errorbar(x-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', color='#ed36d1', label='LR_Bias_ESE')
	ax1.errorbar(x+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', color='#0021ff',label='DL_Bias_ESE')
	# show zero line
	ax1.axhline(y=0, color='grey', linestyle='-.', linewidth=2)

	# set tick limit and label 
	ax1.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
			ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
	ax1.tick_params(axis='both', which='major', labelsize=20)
	ax1.tick_params(axis='y', labelcolor=color)

	# set axis label
	if show_x_label_and_ticks:
		ax1.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
	# else: # hide ticks
	# 	ax1.set_xticklabels([])
	if show_left_y_label_and_ticks:
		ax1.set_ylabel('Bias', fontdict=font, color=color)
	# else:
	# 	ax1.set_yticklabels([])

	# set subfigure title
	ax1.set_title(title, fontdict={'fontsize':35}, pad=20)

	# plot figure 2 with right y axis
	ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
	color = 'tab:blue'

	ax2.bar(x-0.01, cover_lr, width=0.02, fill=False, hatch='///', 
			edgecolor="#425bd9", linewidth=1, alpha=0.5, label='LR_Coverage')
	ax2.bar(x+0.01, cover_dl, width=0.02, fill=False, hatch='..', 
			edgecolor="#258b52", linewidth=1, alpha=0.5, label='DL_Coverage')

	# set tick limit and label
	ax2.set(ylim=(0, 1), yticks=np.arange(0, 1, 0.05))
	ax2.tick_params(axis='y', labelcolor=color)
	ax2.tick_params(axis='both', which='major', labelsize=20)

	# set axis label
	if show_right_y_label_and_ticks:
		ax2.set_ylabel('Coverage', fontdict=font, color=color)  # we already handled the x-label with ax1
	# else:
	# 	ax2.set_yticklabels([])

	# add legend
	# fig.legend(loc='center', bbox_to_anchor=[0.14, 0.89],
	# 		fontsize=15)
	# figsize (100, 100)
	# ax1.legend(loc='center left', bbox_to_anchor=[0.0, 0.97], fontsize=15)
	# ax2.legend(loc='center left', bbox_to_anchor=[0.0, 0.91], fontsize=15)
	# # figsize (100, 80)
	# ax1.legend(loc='center left', bbox_to_anchor=[0.0, 0.96], fontsize=15)
	# ax2.legend(loc='center left', bbox_to_anchor=[0.0, 0.89], fontsize=15)
	# figsize (fig_w*10, fig_h*10) 
	ax1.legend(loc='center left', bbox_to_anchor=[0.0, 0.95], fontsize=15)
	ax2.legend(loc='center left', bbox_to_anchor=[0.0, 0.87], fontsize=15)
	
	# show grid
	plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)


# figure size: (16*6=96, 9*4=36) (100, 45)
# Set font
font = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 35,
        }


# fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(60, 55), facecolor='white')
width = 516 
fig_w, fig_h = set_size(width, fraction=1., subplots=(6, 4))
fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(fig_w*10, fig_h*10), facecolor='white')
# fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(60, 55), facecolor='white')
for i, network_string in enumerate(networktype):
	show_x_label_and_ticks, show_left_y_label_and_ticks, show_right_y_label_and_ticks = False, False, False
	if i//4 == 5: # show x label and ticks for the last row
		show_x_label_and_ticks = True
	if i%4 == 0: # show left y label and ticks for the first column
		show_left_y_label_and_ticks = True
	if i%4 == 3: # show right y label and ticks for the last column
		show_right_y_label_and_ticks = True

	plot_individual_scenario(axs[i//4, i%4], x_label, lr_bias_array[i], dl_bias_array[i], lr_ese_array[i], dl_ese_array[i], 
							 lr_cover_array[i], dl_cover_array[i], font, titles[i],
							 show_x_label_and_ticks, show_left_y_label_and_ticks, show_right_y_label_and_ticks)
fig.tight_layout(pad=15, h_pad=6, w_pad=6)
fig.savefig(save_path + 'bias_cover_combined' + '.png', dpi=100) 
# dpi is set to 100 instead of 300 because figsize is set with *10

def plot_individual_lcover(ax, x, cover_lr, cover_dl, font,title,
                            show_x_label_and_ticks=False, show_y_label_and_ticks=False):
	
	ax.bar(x-0.01, cover_lr, width=0.02, color='#425bd9', edgecolor="white", linewidth=0.7, label='LR')
	ax.bar(x+0.01, cover_dl, width=0.02, color='#258b52', edgecolor="white", linewidth=0.7, label='DL')

	# show grid
	plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

	# set tick limit and label 
	ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
			ylim=(0, 1), yticks=np.arange(0, 1, 0.1))
	ax.tick_params(axis='both', which='major', labelsize=20)
	# set axis label
	if show_x_label_and_ticks:
		ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
	if show_y_label_and_ticks:
		ax.set_ylabel('Coverage', fontdict=font)
	# add legend
	ax.legend(loc='upper left', fontsize=15)
	# set subfigure title
	ax.set_title(title, fontdict={'fontsize':35}, pad=20)

width = 516 
fig_w, fig_h = set_size(width, fraction=1., subplots=(6, 4))
fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(fig_w*10, fig_h*10), facecolor='white')
# fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(60, 55), facecolor='white')
for i, network_string in enumerate(networktype):
	show_x_label_and_ticks, show_y_label_and_ticks = False, False
	if i//4 == 5: # show x label and ticks for the last row
		show_x_label_and_ticks = True
	if i%4 == 0: # show y label and ticks for the first column
		show_y_label_and_ticks = True

	plot_individual_lcover(axs[i//4, i%4], x_label, lr_lcover_array[i], dl_lcover_array[i], font, titles[i], 
						   show_x_label_and_ticks, show_y_label_and_ticks)
fig.tight_layout(pad=15, h_pad=6, w_pad=6)
fig.savefig(save_path + 'lcover_combined' + '.png', dpi=100) 
# dpi is set to 100 instead of 300 because figsize is set with *10