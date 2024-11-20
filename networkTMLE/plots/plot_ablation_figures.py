import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Dir path
dir_path_30 = '../results_csv/improvement/mode_top_30/'
dir_path_50 = '../results_csv/improvement/mode_top_50/'
ts='10'
save_path='../figures/top30vs50/'

# load x label
x_label = np.array([0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 
                    0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 
                    0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950])

# load csv data for top 30
metric='bias'

bias_imp_30 = pd.read_csv(dir_path_30+metric+'_improvement_ts'+ts+'.csv', header=None)
bias_imp_30_array = bias_imp_30.to_numpy()
bias_imp_30_array.shape

bias_imp_50 = pd.read_csv(dir_path_50+metric+'_improvement_ts'+ts+'.csv', header=None)
bias_imp_50_array = bias_imp_50.to_numpy()
bias_imp_50_array.shape

metric='ese'
ese_imp_30 = pd.read_csv(dir_path_30+metric+'_improvement_ts'+ts+'.csv', header=None)
ese_imp_30_array = ese_imp_30.to_numpy()
ese_imp_30_array.shape

ese_imp_50 = pd.read_csv(dir_path_50+metric+'_improvement_ts'+ts+'.csv', header=None)
ese_imp_50_array = ese_imp_50.to_numpy()
ese_imp_50_array.shape

metric='cover'
cover_imp_30 = pd.read_csv(dir_path_30+metric+'_improvement_ts'+ts+'.csv', header=None)
cover_imp_30_array = cover_imp_30.to_numpy()
cover_imp_30_array.shape

cover_imp_50 = pd.read_csv(dir_path_50+metric+'_improvement_ts'+ts+'.csv', header=None)
cover_imp_50_array = cover_imp_50.to_numpy()
cover_imp_50_array.shape

metric='lcover'
lcover_imp_30 = pd.read_csv(dir_path_30+metric+'_improvement_ts'+ts+'.csv', header=None)
lcover_imp_30_array = lcover_imp_30.to_numpy()
lcover_imp_30_array.shape

lcover_imp_50 = pd.read_csv(dir_path_50+metric+'_improvement_ts'+ts+'.csv', header=None)
lcover_imp_50_array = lcover_imp_50.to_numpy()
lcover_imp_50_array.shape

# Set font
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# plot bias + ESE
fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
# ax.plot(x_label, abs(lr_bias_imp_30_array[0, :]), 'o-', linewidth=2, label='30% most connected')
# ax.plot(x_label, abs(lr_bias_imp_50_array[0, :]), 'o-', linewidth=2, label='50% most connected')

ax.bar(x_label-0.01, bias_imp_30_array[7, :], width=0.02, edgecolor="white", linewidth=0.7, label='30% most connected')
ax.bar(x_label+0.01, bias_imp_50_array[7, :], width=0.02, edgecolor="white", linewidth=0.7, label='50% most connected')

# set tick limit and label 
ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
       ylim=(-0.2, 0.3), yticks=np.arange(-0.2, 0.3, 0.1))

# add legend
ax.legend(loc='upper left', fontsize=15)
# show zero line
ax.axhline(y=0, color='gray', linestyle='--')
# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

def plot_improvement(x, improvement_1, improvement_2, improvement_1_label, improvement_2_label,
                     font, save_path='./', figure_name='improvement'):
    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')

    ax.bar(x-0.01, improvement_1, width=0.02, color='#425bd9', edgecolor="white", linewidth=0.7, label=improvement_1_label)
    ax.bar(x+0.01, improvement_2, width=0.02, color='#258b52', edgecolor="white", linewidth=0.7, label=improvement_2_label)

    # show grid
    plt.grid(axis='y', color = 'green', linestyle = '--', linewidth = 0.5)

    # set tick limit and label 
    ## bias
    # ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
    #        ylim=(-0.2, 0.4), yticks=np.arange(-0.2, 0.4, 0.05))
    ## cover
    ax.set(xlim=(0, 1), xticks=np.arange(0, 1, 0.05),
           ylim=(-1, 1), yticks=np.arange(-1, 1, 0.1))
    ax.tick_params(axis='both', which='major', labelsize=15)
    # set axis label
    ax.set_xlabel('Exposure Probability $p_{\omega}$', fontdict=font)
    ax.set_ylabel('Improvement over benchmark', fontdict=font)
    # add legend
    ax.legend(loc='upper left', fontsize=15)

    plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)


networktype = ['10010', '10020', '10030', '10040',
               '20010', '20020', '20030', '20040']

for i, network_string in enumerate(networktype):
    plot_improvement(x_label, bias_imp_30_array[i], bias_imp_50_array[i], 
                    '30% most connected', '50% most connected', 
                    font, save_path=save_path, figure_name='bias_improvement_'+network_string)
    # plot_improvement(x_label, ese_imp_30_array[i], ese_imp_50_array[i], 
    #                  '30% most connected', '50% most connected', 
    #                 font, save_path=save_path, figure_name='ese_improvement_'+network_string)
    plot_improvement(x_label, cover_imp_30_array[i], cover_imp_50_array[i], 
                     '30% most connected', '50% most connected', 
                    font, save_path=save_path, figure_name='cover_improvement_'+network_string)
    # plot_improvement(x_label, lcover_imp_30_array[i], lcover_imp_50_array[i], 
    #                  '30% most connected', '50% most connected', 
    #                  font, save_path=save_path, figure_name='lcover_improvement_'+network_string)


# plot p_{\omega} \in {0.05-0.95} and p_{\omega} \in {0.5-0.95} 
# Set save path
save_path='figures/top30vs50/'
# Set font
font = {'family': 'Times New Roman',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }

# Inside each list are improvements of bias, ese, cover, lcover
uniform_30_all = [0.000, -0.108, 0.071, 0.089]
uniform_30_half = [0.023, -0.142, 0.172, 0.202]
uniform_50_all = [0.019, -0.102, 0.123, 0.150]
uniform_50_half = [0.076, -0.127, 0.310, 0.342]

random_30_all = [-0.014, -0.089, -0.071, -0.048]
random_30_half = [0.045, -0.128, 0.078, 0.108]
random_50_all = [0.014, -0.088, 0.062, 0.059]
random_50_half = [0.059, -0.120, 0.143, 0.171]


metrics = ("Bias", "ESE", "Cover", "L-Cover")
metrics_improvement_uniform = {
    '30% $0.05 \leq p_{\omega} \leq 0.95}$': uniform_30_all,
    '30% $0.5 \leq p_{\omega} \leq 0.95}$': uniform_30_half,
    '50% $0.05 \leq p_{\omega} \leq 0.95}$': uniform_50_all,
    '50% $0.5 \leq p_{\omega} \leq 0.95}$': uniform_50_half,
}
metrics_improvement_random = {
    '30% $0.05 \leq p_{\omega} \leq 0.95}$': random_30_all,
    '30% $0.5 \leq p_{\omega} \leq 0.95}$': random_30_half,
    '50% $0.05 \leq p_{\omega} \leq 0.95}$': random_50_all,
    '50% $0.5 \leq p_{\omega} \leq 0.95}$': random_50_half,
}


# Start
x = np.arange(len(metrics))  # the label locations
width = 0.2  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
for attribute, measurement in metrics_improvement_random.items():
    offset = width * multiplier
    rects = ax.barh(x + offset, measurement, width, label=attribute)
    print(x + offset)
#     ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set(xlim=(-0.4, 0.4), xticks=np.arange(-0.4, 0.4, 0.05),
       ylim=(-0.2, 4), yticks=x+2*width-0.1, yticklabels=metrics)
ax.tick_params(axis='both', which='major', labelsize=15)

# add legend
ax.legend(loc='upper left', fontsize=15)
# show zero line
ax.axvline(x=0, color='gray', linestyle='-')
# show grid
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

def plot_horizontal_bar_abl(metrics, metrics_improvement, font, save_path='./', figure_name='improvement'):
    x = np.arange(len(metrics))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(figsize=(16, 9), facecolor='white')
    for attribute, measurement in metrics_improvement.items():
            offset = width * multiplier
            rects = ax.barh(x + offset, measurement, width, label=attribute)
            multiplier += 1

    # set tick limit and label
    ax.set(xlim=(-0.4, 0.4), xticks=np.arange(-0.4, 0.4, 0.05),
        #    ylim=(0, 4), yticks=x+2*width, yticklabels=metrics)
           ylim=(-0.2, 4), yticks=x+2*width-0.1, yticklabels=metrics)
    ax.tick_params(axis='both', which='major', labelsize=15)

    # set axis label
    ax.set_xlabel('Improvement over benchmark $p_{\omega}$', fontdict=font)
    # ax.set_ylabel('Improvement over benchmark', fontdict=font)
    # add legend
    ax.legend(loc='upper left', fontsize=15)
    # show zero line
    ax.axvline(x=0, color='gray', linestyle='-')
    # show grid
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    plt.show()
    save_path = save_path + figure_name + '.png'
    fig.savefig(save_path, dpi=300)

plot_horizontal_bar_abl(metrics, metrics_improvement_uniform, font, save_path=save_path, figure_name='uniform_improvement')
plot_horizontal_bar_abl(metrics, metrics_improvement_random, font, save_path=save_path, figure_name='random_improvement')