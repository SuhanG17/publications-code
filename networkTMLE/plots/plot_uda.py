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