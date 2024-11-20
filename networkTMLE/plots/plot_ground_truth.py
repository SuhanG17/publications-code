import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from beowulf import truth_values
network = 'uniform' # 'random'
dgm = 'quarantine'
restricted_degree = False
shift = False
n = 500
percent_candidates = 0.3
mode='all'

uni_500_all = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                           n=n, percent_candidates=percent_candidates, mode=mode)
# x=list(uni_500_all.keys())
# y=list(uni_500_all.values())

uni_500_50top = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                             n=n, percent_candidates=0.5, mode='top')
uni_500_50bottom = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                             n=n, percent_candidates=0.5, mode='bottom')

x_uni_500 = [list(uni_500_all.keys()), list(uni_500_50top.keys()), list(uni_500_50bottom.keys())]
y_uni_500 = [list(uni_500_all.values()), list(uni_500_50top.values()), list(uni_500_50bottom.values())]

# Set font
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

ran_500_bot = truth_values(network='random', dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                           n=n, percent_candidates=0.5, mode='bottom')
x=list(ran_500_bot.keys())
y=list(ran_500_bot.values())



# plot Ground Truth
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
ax.plot(x, y, 'o-', linewidth=2, label='mode=50%+Bottom')

# ax.plot(x_uni_500[0], y_uni_500[0], 'o-', linewidth=2, label='mode=100%')
# ax.plot(x_uni_500[1], y_uni_500[1], 'o-', linewidth=2, label='mode=50%+Top')
# ax.plot(x_uni_500[2], y_uni_500[2], 'o-', linewidth=2, label='mode=50%+Bottom')

# ax.errorbar(x_label-0.01, y_lr, yerr=y_err_lr, capsize=5, fmt='-o', label='LR')
# ax.errorbar(x_label+0.01, y_dl, yerr=y_err_dl, capsize=5, fmt='-o', label='DL')
# show zero line
# ax.axhline(y=0, color='gray', linestyle='--')
# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

# set tick limit and label 
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
       ylim=(0, 1.1), yticks=np.arange(0, 1.1, 0.1))
ax.tick_params(axis='both', which='major', labelsize=15)
# set axis label
ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
ax.set_ylabel('Ground Truth $\psi$', fontdict=font)
# add legend
ax.legend(loc='upper left', fontsize=15,ncol=3)

plt.show()
# fig.savefig('bias_ese.png', dpi=300)

def plot_ground_truth(x_list, y_list,
                      font, save_path='./', figure_name='ground_truth'):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    ax.plot(x_list[0], y_list[0], 'o-', linewidth=2, label='mode=100%')
    ax.plot(x_list[1], y_list[1], 'o-', linewidth=2, label='mode=50%+Top')
    ax.plot(x_list[2], y_list[2], 'o-', linewidth=2, label='mode=50%+Bottom')
    # show grid
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    # set tick limit and label 
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
           ylim=(0, 1.1), yticks=np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set axis label
    ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
    ax.set_ylabel('Ground Truth $\psi$', fontdict=font)
    # add legend
    ax.legend(loc='upper left', fontsize=15, ncol=3)

    # plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)

# network = 'uniform'
# n = 500

save_path = '../figures/ground_truth/'
for network in ['uniform', 'random']:
    for n in [500, 1000, 2000]:
        all = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                            n=n, percent_candidates=0.5, mode='all')
        top = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                            n=n, percent_candidates=0.5, mode='top')
        bottom = truth_values(network=network, dgm=dgm, restricted_degree=restricted_degree, shift=shift, 
                                n=n, percent_candidates=0.5, mode='bottom')
        
        x_ls = [list(all.keys()), list(top.keys()), list(bottom.keys())]
        y_ls = [list(all.values()), list(top.values()), list(bottom.values())]

        name_string = network + '_' + str(n) + '_' + 'p50' + '_ground_truth'
        plot_ground_truth(x_ls, y_ls, font, save_path=save_path, figure_name=name_string)