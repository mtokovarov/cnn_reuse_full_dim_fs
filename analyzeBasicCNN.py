import numpy as np
from matplotlib import pyplot

#type the name of the file you want to analyze
data = np.load('rep_2_shift_10_noise_0.10_reg_0.00050.npz')

val_accs = data['total_val_accs']

maxes = np.max(val_accs, axis = -1)

means = np.mean(maxes, axis = -1)

part_means = np.array([np.mean(maxes[:i,...]) for i in range(1, maxes.shape[0]+1)])
part_std = np.array([np.std(part_means[:i,...]) for i in range(1, maxes.shape[0]+1)])

maxes_inds = np.argmax(val_accs, axis = -1)

del data

  
fig, axs = pyplot.subplots(2)
pyplot.rcParams.update({'font.size': 8})
axs[0].plot(part_means[:np.sum(means!=0)])
axs[0].set_ylabel("Partial mean accuracy")
axs[1].plot(part_std[:np.sum(means!=0)])
axs[1].set_ylabel("Partial deviation of accuracy")
pyplot.xlabel("Epoch number")


