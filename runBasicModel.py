import numpy as np
import datetime
from FullDimLReuseCNN import FullDimLReuseCNN
import os
if (__name__ == '__main__'):
    file_path = '/data.npz'

    data = np.load(file_path)
    X = data["X"]
    y = data["y"]
   
    learning_rate = 0.001
    block_cnt = 1
    sampleCnt = 480
    input_size = (sampleCnt,) + X.shape[-2:]
    nclasses = y.shape[-1]
    batch_size = 16
    reg_rate = 0.0005
    
    max_random_shift = 30
    cv_folds_cnt = 5
    act_func = 'tanh'
    noise_level = 0.5
    layer_reuse_repeat = 2#s = [1, 2, 3, 4]
    epoch_cnt = 45
    step_per_epoch = 45
    cv_repeats = 30
    dropout = 0.5
    threshold_acc = 0.75
    
    dt_now = str(datetime.datetime.now())
    
    save_folder_name = dt_now.split('.')[0]
    save_folder_name = save_folder_name.replace(':', '_')
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    
    model = FullDimLReuseCNN(learning_rate, block_cnt, input_size, nclasses, batch_size, layer_reuse_repeat, 
                     reg_rate,act_func, X, y, max_random_shift, noise_level, dropout,
                     cv_folds_cnt, save_folder_name, threshold_acc)  
        
    model.perform_repeated_cv_train_test(cv_repeats, epoch_cnt, step_per_epoch)
        