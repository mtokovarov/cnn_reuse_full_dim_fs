"""if you use this code, please cite:
    Tokovarov M.
    Convolutional neural networks with reusable full-dimension-long layers for feature selection and classification of motor imagery in EEG signals
    ICANN2020
    2020
"""

import numpy as np

#the module implementing high presicion computations
import mpmath as mp

def standardize_array(array, axis):
    means = np.mean(array, axis = axis)
    means = np.expand_dims(means, axis)
    stds = np.std(array, axis = axis) 
    
    #in case if std==0 a least significant unit is added to avoid division by zero
    stds += np.spacing(stds)

    stds = np.expand_dims(stds, axis)
    array = (array - means)/stds

    return array


def find_polynomial_derivative(poly):
    derivative = [mp.mpf(len(poly) - i - 1)*p for i, p in enumerate(poly[:-1])]
    return derivative

def newton_raphson(poly, xstart, eps, dps, steps_cnt):
    mp.dps = dps
    xstart = mp.mpf(xstart)
    eps = mp.mpf(eps)
    derivative = find_polynomial_derivative(poly)
    fx = mp.polyval(poly, xstart)
    
    for i in range(steps_cnt):
        if (mp.fabs(fx)<eps):                
            return xstart
        derx = mp.polyval(derivative, xstart)
        xstart -= fx/derx
        fx = mp.polyval(poly, xstart)
        
    raise NameError('Could not converge after %d steps, x = %f, fx = %f'%(steps_cnt, xstart, fx))

def multiply_polynome_by_monome(poly, mono):
    mp.dps = 300
    poly = poly.copy()
    mono = mono.copy()
    for p in poly:
        assert(type(p) == mp.ctx_mp_python.mpf)
    mono = [mp.mpf(float(mono[0])), mp.mpf(float(mono[1]))];
    poly = [mp.mpf(0)] + poly
    for i in range(len(poly)-1):
        poly[i] += poly[i+1]*mono[0]
        poly[i+1] *= mono[1]
    return poly

def make_polynimial(fuzzy_measures):
    poly = [mp.mpf(1)]
    for fm in fuzzy_measures:
        mono = [fm, 1]
        poly = multiply_polynome_by_monome(poly, mono)            
    poly[-1] -= mp.mpf(1)
    poly[-2] -= mp.mpf(1)
    return poly

def get_fuzzy_lambda(fuzzy_measures, dps):
    #otherwise aggregate
    #described in Pedrycz, W., & Gomide, F. 1998. An introduction to fuzzy sets: analysis and design. Mit Press.
    poly = make_polynimial(fuzzy_measures) 
    #we expect only one root, so there is no need for computing the whole set of roots
    #the root expected to be near -1, hence the starting point is near -1
    root = newton_raphson(poly, mp.mpf(-0.5), 1e-7, dps, 10000)
    #as Sugeno (1974) has shown, there exists a unique \lambda \in (-1, \infty) different from zero and satisfying the relationship (9.5) in Pedrycz, W., & Gomide, F. 1998 
    return float(root)

def compute_choquet_integral(lamb, vals, fuzzy_measures):
    #described in Pedrycz, W., & Gomide, F. 1998. An introduction to fuzzy sets: analysis and design. Mit Press.
    #sorting in descending order. Argsort can sort only in asc :(, so the array has to be multiplied by -1
    inds_of_sorted = np.argsort(-vals)
    fuzzy_measures = fuzzy_measures[inds_of_sorted]
    vals = vals[inds_of_sorted]
    subset_measure = fuzzy_measures[0]
    result = 0
    
    for i in range(vals.shape[0]-1):
        result += (vals[i] - vals[i+1])*subset_measure        
        subset_measure += fuzzy_measures[i+1] + fuzzy_measures[i+1]*subset_measure*lamb
        #print("subset measure = %f"%subset_measure)
        #a = input("fm")
    if (np.abs(1 - subset_measure)>1e-3):
        raise NameError('Final fuzzy measure is too far from 1. It is equal to: %f'%subset_measure)
    result += vals[-1]*subset_measure
    return result

def corr_with_class_labels(arr, class_labels, axis):
    arr = standardize_array(arr, axis)
    N = arr.shape[axis]
    if (len(class_labels.shape)>1):
        class_labels = np.squeeze(class_labels)
    class_labels = np.expand_dims(class_labels, -1).astype(np.float32)
    class_labels = standardize_array(class_labels, axis)
    class_labels = np.expand_dims(class_labels, -1)
    corrs = arr * class_labels
    corrs = np.sum(corrs, axis = axis)/N
    return corrs