"""if you use this code, please cite:
    Tokovarov M.
    Convolutional neural networks with reusable full-dimension-long layers for feature selection and classification of motor imagery in EEG signals
    ICANN2020
    2020
"""

import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

def make_2d_el_map(arr, electrodes_cnt_to_highlight = None):
    #the mapping is provided under the following link:
    #https://physionet.org/content/eegmmidb/1.0.0/64_channel_sharbrough.pdf
    mapping = {21:(0,4), 22:(0,5), 23:(0,6), 
               24:(1,3), 25:(1,4), 26:(1,5), 27:(1,6), 28:(1,7), 
               29:(2,1), 30:(2,2), 31:(2,3), 32:(2,4), 33:(2,5), 34:(2,6), 35:(2,7), 36:(2,8), 37:(2,9),
               38:(3,1), 0:(3,2),  1:(3,3),  2:(3,4),  3:(3,5),  4:(3,6),  5:(3,7),  6:(3,8),  39:(3,9),
               42:(4,0), 40:(4,1), 7:(4,2),  8:(4,3),  9:(4,4),  10:(4,5), 11:(4,6), 12:(4,7), 13:(4,8), 41:(4,9), 43:(4,10),
               44:(5,1), 14:(5,2), 15:(5,3), 16:(5,4), 17:(5,5), 18:(5,6), 19:(5,7), 20:(5,8), 45:(5,9),
               46:(6,1), 47:(6,2), 48:(6,3), 49:(6,4), 50:(6,5), 51:(6,6), 52:(6,7), 53:(6,8), 54:(6,9),
               55:(7,3), 56:(7,4), 57:(7,5), 58:(7,6), 59:(7,7),
               60:(8,4), 61:(8,5), 62:(8,6),
               63:(9,5)}


    electrodes_2d = np.zeros((10,11))
    
    els_selected = arr.copy()
    electrode_cnt = arr.shape[-1]
    #in case if we want to show only selected electrodes
    if (electrodes_cnt_to_highlight is not None and
        electrodes_cnt_to_highlight<=electrode_cnt and
        electrodes_cnt_to_highlight>0):
        nums = np.argsort(-arr)
        #els_selected[:] = 1
        to_zeros = nums[electrodes_cnt_to_highlight:]
        els_selected[to_zeros] = np.max(arr)*0.07
    for i, e in enumerate(els_selected):
        electrodes_2d[mapping[i]] = e
    return electrodes_2d

#type the name of the file you want to analyze
path="filename.npz"
el_cnt = 14

fs_part_means_list = []
base_part_means_list = []
fig, ax = pyplot.subplots(2, 1, figsize=(20, 15))
pyplot.rcParams.update({'font.size': 30})

with np.load(path) as data:
    results = data['total_val_accs']
    el_weights = data['total_electrode_weights']
mean_electrode_weights = np.mean(el_weights, axis = (0,1))

electrodes_mapped_2d = make_2d_el_map(mean_electrode_weights)
ax[0, 0].imshow(electrodes_mapped_2d, cmap = 'jet') #row=0, col=0
electrodes_mapped_2d = make_2d_el_map(mean_electrode_weights, el_cnt)
im = ax[1, 0].imshow(electrodes_mapped_2d, cmap = 'jet') #row=0, col=0
ax[0, 0].set_xticklabels([])
ax[0, 0].set_yticklabels([])
ax[1, 0].set_xticklabels([])
ax[1, 0].set_yticklabels([])
fs_part_means = np.array([np.mean(results[:i,...]) for i in range(1, results.shape[0]+1)])

fig, ax = pyplot.subplots(1, 1, figsize=(20, 15))
ax.plot(fs_part_means)
