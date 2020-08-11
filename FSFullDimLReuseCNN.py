"""if you use this code, please cite:
    Tokovarov M.
    Convolutional neural networks with reusable full-dimension-long layers for feature selection and classification of motor imagery in EEG signals
    ICANN2020
    2020
"""

from FullDimLReuseCNN import FullDimLReuseCNN
import numpy as np
import utilityFunctions as uf

from keras.layers import Input, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Permute, GaussianNoise
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import initializers
from keras import regularizers

from keras.optimizers import Adam






class FSFullDimLReuseCNN(FullDimLReuseCNN):
    def __init__(self, learning_rate, block_cnt, input_size, nclasses, batch_size, repeat,
                 reg_rate,act_func, X, y, max_random_shift, noise_level, dropout,
                 cv_folds_cnt, save_folder_name, threshold_acc):
        #parent constructor
        FullDimLReuseCNN.__init__(self, learning_rate, block_cnt, input_size, nclasses, batch_size, repeat,
                 reg_rate, act_func, X, y, max_random_shift, noise_level, dropout,
                 cv_folds_cnt, save_folder_name, threshold_acc)
     
    def getWeightsFromFilters(self, model):
        for layer in model.layers:
            if 'conv' in layer.name:
                layer_weights = layer.get_weights()[0]
                #only spatial filters have dims = 64 
                if self.input_size[1] in layer_weights.shape:
                    weights = layer_weights
                    break
        weights = np.squeeze(weights)
        weights = np.abs(weights)
        return weights
            
    def getFuzzyMeasuresOfElectrodeFilters(self, fs_model):
        #preparing data for FS
        input_x = self.X_train
        input_x = input_x[:,self.start:self.end,...]
        input_x = uf.standardize_array(input_x, self.time_axis)
        
        #make predictions with intermediate feature maps
        y_predicted = fs_model.predict(input_x) 
        #predicted classes are stored in the last tensor
        classes = np.argmax(y_predicted[-1], axis = -1)
        #eliminate dims with shape == 1
        space = np.squeeze(y_predicted[0]) 
        #correlate along feature maps
        corrs = uf.corr_with_class_labels(space, classes, axis = 0) 
        #find max corr coefft along time axis
        #the corr coeffcients are used as the fuzzy measures of the filters, 
        #the higher the measure, the more valuable information contains the filter
        fuzzy_measures = np.max(np.abs(corrs),axis = 0)     
        return fuzzy_measures
    
    def findAggregatedElWeigths(self, epochs, steps_per_epoch, with_aggregation):
        fs_model = self.prepareFSModel(epochs, steps_per_epoch)
        fuzzy_measures = self.getFuzzyMeasuresOfElectrodeFilters(fs_model)
        if (with_aggregation == True):
            fuzzy_lambda = uf.get_fuzzy_lambda(fuzzy_measures, 300)
        else:
            fuzzy_measures = fuzzy_measures/np.sum(fuzzy_measures)
        
        weights = self.getWeightsFromFilters(fs_model)
        
        electrode_weights = np.zeros(weights.shape[0])
        for j in range(weights.shape[0]):
            if (with_aggregation == True):
                electrode_weights[j] = uf.compute_choquet_integral(fuzzy_lambda,
                                   weights[j,:], fuzzy_measures)
            else:
                electrode_weights[j] = np.dot(fuzzy_measures, weights[j,:])    
        return electrode_weights
    
    def make_block_with_intermediate_outputs(self,  tensor, reg_rate, act_func, repeat):
        h,w,d = [int(dim) for dim in tensor.shape[1:]]
        layers = [
                    Conv2D(w, (1, w), activation=act_func,
                       kernel_regularizer=regularizers.l2(reg_rate),
                       kernel_initializer=initializers.he_normal(),
                       padding="valid"),
                    Permute((1,3,2)),
                    GaussianNoise(self.noise_level),
                    Conv2D(h, (h, 1), activation=act_func,
                       kernel_regularizer=regularizers.l2(reg_rate),
                       kernel_initializer=initializers.he_normal(),
                       padding="valid"),
                    Permute((3,2,1)),
                    GaussianNoise(self.noise_level)
                ]
        for_concat = [tensor]
        outputs = []
        for i in range(repeat):
            if (tensor.shape[-1]>1):
                tensor = Conv2D(1, kernel_size=[1,1], padding='same',
                              kernel_initializer=initializers.he_normal(),
                              activation = act_func, 
                              kernel_regularizer=regularizers.l2(reg_rate),
                              )(tensor)
            for layer in layers:
                tensor = layer(tensor)
                if ('permute' in layer.name.lower()):
                    outputs.append(tensor)
            for_concat.append(tensor)
            tensor = concatenate(for_concat)
        return tensor, outputs
    
    def creat_net_with_intermediate_outputs(self, input_size, block_cnt, reg_rate, nclasses, act_func, learning_rate, repeat):
        net_input = Input(input_size)
        last = GaussianNoise(self.noise_level)(net_input)
        last = last
        for i in range(block_cnt):
            last, reuse_outputs = self.make_block_with_intermediate_outputs(last, reg_rate, act_func, repeat)

        last = Conv2D(nclasses, kernel_size=[1,1], padding='same',
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(reg_rate),
                activation=act_func,
                name = 'final_1x1_OUT')(last)
        last = Dropout(self.dropout)(last)
        last = GlobalAveragePooling2D()(last)
        
        last = Dense(nclasses, activation='softmax',
                       kernel_initializer=initializers.he_normal())(last)
        
        model = Model(inputs = net_input, outputs = reuse_outputs+[last])
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate),
                      metrics=['categorical_accuracy'])

        return model
    
    def prepareFSModel(self, epochs, steps_per_epoch):
        #train a model
        self.run_one_session(epochs, steps_per_epoch)
        #save the weights
        save_path = self.save_folder_name + '/fs_model.h5'
        self.model.save_weights(save_path)
        
        #create fs model
        fs_model = self.creat_net_with_intermediate_outputs(self.input_size, self.block_cnt, self.reg_rate, 
                               self.nclasses, self.act_func, 
                               self.learning_rate, self.repeat) 
        #loading saved weights
        fs_model.load_weights(save_path)
        return fs_model
    
    
        
    def spatial_feature_selection_cv(self, epochs, steps_per_epoch, electrode_cnt,
                                  with_aggregation = True):
        self.make_cv_fold_inds()   
        all_electrode_weights = np.zeros((self.cv_folds_cnt, self.input_size[1]))
        val_accs = np.zeros((self.cv_folds_cnt, epochs))
        train_accs = np.zeros((self.cv_folds_cnt, epochs))
        
        for i in range(self.cv_folds_cnt):
            self.make_test_train_cv_dataset(i)
            electrode_weights = self.findAggregatedElWeigths(epochs,
                                            steps_per_epoch, with_aggregation)
            all_electrode_weights[i,...] = electrode_weights
            ranking = np.argsort(-electrode_weights)
            chosen_ranking = ranking[:electrode_cnt]
            
            #using selected positions from electrode ranking as a mask
            self.X_test = self.X_test[...,chosen_ranking,:]
            self.X_train = self.X_train[...,chosen_ranking,:]
            
            #save input_size temporarely as it changes from 
            temp_is = self.input_size
            #change input_size, changing electrode number 
            self.input_size = (self.input_size[0], electrode_cnt, self.input_size[2])        
            train, val = self.run_one_session(epochs, steps_per_epoch)
            #set input size back
            val_accs[i,:] = val
            train_accs[i,:] = train
            self.input_size = temp_is
        return val_accs, train_accs, all_electrode_weights
    
    def perform_repeated_cv_FS_electrodes(self, repeats, epochs,
                        steps_per_epoch, electrode_cnt, with_aggregation):
        total_val_accs= np.zeros((repeats, self.cv_folds_cnt, epochs))
        total_train_accs= np.zeros((repeats, self.cv_folds_cnt, epochs))
        total_electrode_weights = np.zeros((repeats,self.cv_folds_cnt, self.input_size[1]))
        filename = 'FS_rep_%d_reuse_%d_shift_%d_noise_%.2f_reg_%.5f.npz'%(repeats, self.repeat,
                                                     self.max_random_shift, 
                                                     self.noise_level,
                                                     self.reg_rate)
        filename = self.save_folder_name + '/' + filename
        for i in range(repeats):
            val_accs, train_accs,\
                electrode_weights = self.spatial_feature_selection_cv(epochs, 
                                steps_per_epoch, electrode_cnt, with_aggregation)
            total_train_accs[i,...] = train_accs
            total_val_accs[i,...] = val_accs
            total_electrode_weights[i,...] = electrode_weights
            np.savez(filename, total_train_accs = total_train_accs, 
                     total_val_accs = total_val_accs, 
                     total_electrode_weights = total_electrode_weights)   
        return total_train_accs, total_val_accs, total_electrode_weights
    