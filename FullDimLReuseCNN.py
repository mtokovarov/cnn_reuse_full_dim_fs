

from keras.layers import Input, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Dense, Dropout, Permute, GaussianNoise
from keras.layers.convolutional import Conv2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import initializers
from keras import regularizers
from contextlib import redirect_stdout
from keras.optimizers import Adam
import numpy as np
import gc
from keras import backend as K
from keras.callbacks import EarlyStopping
import json
import utilityFunctions as uf

class FullDimLReuseCNN():
    def __init__(self, learning_rate, block_cnt, input_size, nclasses, batch_size, repeat,
                 reg_rate, act_func, X, y, max_random_shift, noise_level, dropout,
                 cv_folds_cnt, save_folder_name, threshold_acc):        
        self.learning_rate = learning_rate
        self.block_cnt = block_cnt
        self.input_size = input_size
        self.nclasses = nclasses
        self.batch_size = batch_size
        self.repeat = repeat
        self.reg_rate = reg_rate
        self.noise_level= noise_level
        self.dropout = dropout
        self.max_random_shift = max_random_shift
        self.cv_folds_cnt = cv_folds_cnt
        self.act_func = act_func
        self.save_folder_name = save_folder_name
        self.threshold_acc = threshold_acc
        
        self.X = X
        self.y = y
        #for convinience the aliases are defined:
        self.time_axis = 1
        self.space_axis = 2
        self.start = 1*160
        self.end = self.start + self.input_size[0]
        np.random.seed(42)
        self.save_parameters_to_txt(self.save_folder_name)
    
    def save_parameters_to_txt(self, save_folder_name):
        """saving parameters to a txt file
        The function is useful for gridsearch, when different parameter sets
        are tested"""
        file_name = save_folder_name + '/parameters.txt'
        fields = self.__dict__
        fields = {key:value for key, value in zip(fields.keys(), fields.values()) 
                    if (key not in ['X', 'y'])}
        with open(file_name, 'w') as outfile:
            json.dump(fields, outfile)
        
    
    def make_block(self, tensor, reg_rate, act_func, repeat):
        """the function defines the procedure of creation of a single reusable
        full dimension factorized convolution block"""
        height,width = (int(dim) for dim in tensor.shape[1:3]) #
        layers = [#electrode-wise convolution
                    Conv2D(width, (1, width), activation=act_func,
                       kernel_regularizer=regularizers.l2(reg_rate),
                       kernel_initializer=initializers.he_normal(),
                       padding="valid"),
                    #transformation of output tensor 
                    #from shapes height-1-width to shapes height-width-1
                    Permute((1,3,2)),
                    #gaussian noise with mean = 0 and std = self.noise_level
                    GaussianNoise(self.noise_level),
                    #sample-wise convolution
                    Conv2D(height, (height, 1), activation=act_func,
                       kernel_regularizer=regularizers.l2(reg_rate),
                       kernel_initializer=initializers.he_normal(),
                       padding="valid"),
                    #transformation of output tensor 
                    #from shapes 1-width-height to shapes height-width-1       
                    Permute((3,2,1)),
                    #gaussian noise with mean = 0 and std = self.noise_level
                    GaussianNoise(self.noise_level)
                ]
        for_concat = [tensor]
        for i in range(repeat):
            if (tensor.shape[-1]>1):
                #transforming input tensor from shapes h-w-d to h-w-1 by 1x1 convolution
                #the transformation makes possible repeated use of the layers
                tensor = Conv2D(1, kernel_size=[1,1], padding='same',
                              kernel_initializer=initializers.he_normal(),
                              activation = act_func, 
                              kernel_regularizer=regularizers.l2(reg_rate),
                              )(tensor)
            for layer in layers:
                tensor = layer(tensor) 
            for_concat.append(tensor)
            tensor = concatenate(for_concat) #dense connection
        return tensor
    
    def create_net(self, input_size, block_cnt, reg_rate, nclasses,
                   act_func, learning_rate, repeat):
        net_input = Input(input_size)
        #gaussian noise with mean = 0 and std = self.noise_level
        last = GaussianNoise(self.noise_level)(net_input)
        for i in range(block_cnt):
            last = self.make_block(last, reg_rate, act_func, repeat)   
        #1x1 convolution transforming the tensor to the shapes h-w-nclasses
        #so we obtain 'nclasses' feature maps
        last = Conv2D(nclasses, kernel_size=[1,1], padding='same',
                kernel_initializer=initializers.he_normal(),
                kernel_regularizer=regularizers.l2(reg_rate),
                activation=act_func,
                name = 'final_1x1_OUT')(last)
        last = Dropout(self.dropout)(last)
        #transforming the input tensor from shapes h-w-nclasses to nclasses-long vector
        last = GlobalAveragePooling2D()(last)

        last = Dense(nclasses, activation='softmax',
                       kernel_initializer=initializers.he_normal())(last)

        model = Model(inputs = net_input, outputs = last)
        model.compile(loss='categorical_crossentropy', 
                      optimizer=Adam(learning_rate),
                      metrics=['categorical_accuracy'])
        #model summary redirected saved to a file - convinient for analysis
        #and debugging
        with open('last_model_summary.txt', 'w') as f:
            with redirect_stdout(f):
                model.summary()
        return model
    
    def merge_two_first_dimensions(self, array):
        d1, d2 = array.shape[:2]
        array = np.reshape(array, (d1*d2,) + array.shape[2:])
        return array
    
    def merge_first_two_dimensions_of_arrays_in_list(self, list_of_arrays):
        return (self.merge_two_first_dimensions(arr) for arr in list_of_arrays)


    def make_cv_fold_inds(self):
        #the method defines the procedure of randomply splitting the dataset 
        #into cross validation folds
        #the data of a patient may appear in only one fold
        patient_nums = np.arange(self.X.shape[0])
        np.random.shuffle(patient_nums)
        self.folds = np.array_split(patient_nums, self.cv_folds_cnt)

    def make_test_train_cv_dataset(self, test_fold_num):
        #the method creates train and test sets out of cv folds for given test_fold_num
        test_inds = self.folds[test_fold_num]
        train_inds = [f for i, f in enumerate(self.folds) if i != test_fold_num]
        train_inds = np.concatenate(train_inds)
        
        self.X_train = self.X[train_inds,...]
        self.X_test = self.X[test_inds,:,self.start:self.end,...]

        
        self.y_train = self.y[train_inds,...]
        self.y_test = self.y[test_inds,...]

        #first two dimensions, i.e. observations and persons are merged for every array
        #so the arrays are transformed from shapes p-o-... to shapes p*o-...
        self.X_train, self.X_test, self.y_train, self.y_test = self.merge_first_two_dimensions_of_arrays_in_list([self.X_train, self.X_test, self.y_train, self.y_test])
        self.X_test = uf.standardize_array(self.X_test, self.time_axis)

        self.train_set_indeces = np.arange(self.X_train.shape[0])
    
    def train_generator(self):
        while True:
            batch_inds = np.random.choice(self.train_set_indeces, self.batch_size)
            if (self.max_random_shift != 0):
                shifts = np.random.randint(-self.max_random_shift, self.max_random_shift, self.batch_size)
            else:
                shifts = np.zeros(self.batch_size).astype(np.uint16)
            starts = self.start + shifts
            ends = self.end + shifts
            batch_samples = np.zeros((self.batch_size,self.end - self.start) + self.X_train.shape[2:])
            for i, (bi, s, e) in enumerate(zip(batch_inds, starts, ends)):
                   batch_samples[i,...] = self.X_train[bi,s:e,...]
                
            batch_samples = uf.standardize_array(batch_samples, self.time_axis)
            batch_labels = self.y_train[batch_inds,...]
                        
            yield batch_samples, batch_labels
        
    def fit_model(self, epochs, steps_per_epoch, isWithCallback = False):    
        gc.collect()
        if(isWithCallback):
            cb = EarlyStopping(monitor='categorical_accuracy', baseline = 0.5, 
                               patience = 7)
            return self.model.fit_generator(self.train_generator(),
                              validation_data = (self.X_test, self.y_test),
                              steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=[cb])
        else:
            return self.model.fit_generator(self.train_generator(),
                              validation_data = (self.X_test, self.y_test),
                              steps_per_epoch=steps_per_epoch, epochs=epochs)

    def initiate_new_model(self):
        #convenience method, calling 'create net' method with standard arguments
        self.model = self.create_net(self.input_size, self.block_cnt, self.reg_rate, 
                                   self.nclasses, self.act_func, 
                                   self.learning_rate, self.repeat)     
        
    def run_one_session(self, epochs, steps_per_epoch):
        max_acc = 0
        while (max_acc < self.threshold_acc):
            #due to high regularization rate sometimes training won't start 
            #as randomly generated weights cause too high regularization loss sometimes
            #threshold_acc is set to 0.75
            K.clear_session() #necessary for performance, 
            #otherwise it drops drastically for high number of repeats
            self.initiate_new_model()
            h = self.fit_model(epochs, steps_per_epoch, isWithCallback = True)
            #training accuracy is used, so the approach is fair
            max_acc = np.max(h.history['categorical_accuracy'])
        val = h.history['val_categorical_accuracy']
        train = h.history['categorical_accuracy']
        epochs_passed = len(val)
        #early stopping is used, so it is possible, 
        #that training is finished before epoch_number == epochs
        #so if it is the case, the training is continued for unused number of epochs
        if (epochs_passed < epochs):
            h = self.fit_model(epochs - epochs_passed, steps_per_epoch)
            val = np.concatenate([val, h.history['val_categorical_accuracy']])
            train = np.concatenate([train, h.history['categorical_accuracy']])
        return train, val
    
    def perform_cv(self, epochs, steps_per_epoch):
        #method describing a single run of cross validation
        self.make_cv_fold_inds()
        val_accs = np.zeros((self.cv_folds_cnt, epochs))
        train_accs = np.zeros((self.cv_folds_cnt, epochs))
        for i in range(self.cv_folds_cnt):
            print("now running: " + str(i) + "-th fold of crossvalidation")
            self.make_test_train_cv_dataset(i)
            train, val = self.run_one_session(epochs, steps_per_epoch)
            val_accs[i,:] = val
            train_accs[i,:] = train
        return train_accs, val_accs

    def perform_repeated_cv_train_test(self, repeats, epochs, steps_per_epoch):
        #method describing complete procedure of repeated cross validation
        total_val_accs= np.zeros((repeats, self.cv_folds_cnt, epochs))
        total_train_accs= np.zeros((repeats, self.cv_folds_cnt, epochs))
        #the file name contains the most important parameters of the model
        filename = 'rep_%d_shift_%d_noise_%.2f_reg_%.5f.npz'%(self.repeat,
                                                     self.max_random_shift, 
                                                     self.noise_level,
                                                     self.reg_rate)
        filename = self.save_folder_name + '/' + filename
        for i in range(repeats):
            train_accs, val_accs = self.perform_cv(epochs, steps_per_epoch)
            total_train_accs[i,...] = train_accs
            total_val_accs[i,...] = val_accs
            #the results are resaved in every turn of the loop in order to 
            #be able to check the progress and simply just in case, 
            #if something goes wrong, to have all saved
            np.savez(filename, total_train_accs = total_train_accs, 
                     total_val_accs = total_val_accs)
        return total_train_accs, total_val_accs