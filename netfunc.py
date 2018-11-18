
# coding: utf-8

# In[ ]:

import numpy as np
from scipy.spatial import distance as dist

# -- initial network configuration --

def netInit (numFeatures):
    
    if numFeatures < 1:
        print ("Number of features must be at least 1")
        sys.exit()
    weight = np.zeros ((numFeatures, 1))
    net = {'numFeatures': numFeatures, 'numClusters': 1, 'maxNumClusters': 15,  'weight': weight, 'disThreshold': 0.85, 'numEpochs': 10, 'learningRate': 0.6, 'rateDecrease': 0.05}
    return net


# -- learn network on vectors from dataLearn --

def netLearn (net, dataLearn):
    
    # ascertain that # of features equals # of weights per cluster
    if len(dataLearn) != len(net['weight']):
        print ("Number of features in training data does not match the number of weight connections per neuron")
        sys.exit()
    w = net['weight']
    counter = np.zeros(net['maxNumClusters'])
    R = np.zeros(net['maxNumClusters']) # vector for Euclidean distances
    clustLearn = np.zeros(len(dataLearn[0,:]))
    newNet = {}
    
    # data normalization
    for j in range (net['numFeatures']):
        maxVal = max(dataLearn[j,:])
        for k in range (len(dataLearn[0,:])):
            dataLearn[j,k] = dataLearn[j,k] / maxVal
    
    # learn on all epochs
    for i in range(net['numEpochs']):
        w_temp = [] # temporary storage for weights
        v = net['learningRate'] - net['rateDecrease']*i # learning rate for current epoch
        
        # learning on all data vectors within epoch
        for j in range (len(dataLearn[0,:])):
            res = False # label showing absence(False)/presence(True) of weight changes
            
            # for 1st epoch and 1st learning vector
            if (j == 0) and (i == 0):
                w[:,j] = dataLearn[:,j] # equate weight values to input values
                res = True
                clustLearn[j] = 1 # in this case cluster is 1
            else:
                # count distances between current vector and each cluster
                for k in range (len(w[0,:])):
                    R[k] = dist.euclidean(dataLearn[:,j], w[:,k])
                    
                # check if min distance less than threshold
                if min(R[np.nonzero(R)]) <= net['disThreshold']:
                    t = list(R).index(min(R[np.nonzero(R)])) # index of min distance in vector of distances
                    counter[t] += 1
                    clustLearn[j] = t + 1
                    w_temp = w[:,t]
                    s = np.subtract(dataLearn[:,j], w_temp)
                    w[:,t] = w_temp + s*v # update weights
                    
                    # check if weights changed
                    if sum (abs(np.subtract(w[:,t], w_temp))) > 0.001: 
                        res = True
                        
                # check if reached max number of clusters
                elif len(w[0,:]) < net['maxNumClusters']:
                    w = np.c_[w, dataLearn[:,j]] # add data column to the end of weight matrix
                    res = True
                else:
                    clustLearn[j] = -1 # set -1 if category has not been found
                    res = True
                    
            # exit if no weight changes occurred in last epoch
            if (res == False) and (i == net['numEpochs']-1):
                break
        if res == False and (i == net['numEpochs']-1):
            break
            
    cl = len(w[0,:])
    
    # delete unused clusters
    for i in range (net['maxNumClusters']-1, -1, -1):
        if (counter[i] == 0) and (cl > i):
            w = np.delete(w, i, 1)
            for j in range(len(clustLearn)):
                if clustLearn[j] > (i+1):
                    clustLearn[j] -= 1
    
    # cut empty parts of distance vectors
    cl = len(w[0,:])
    R = R [:cl]
    
    # save learned network configuration
    net['weight'] = w
    net['numClusters'] = cl
    net['lastDistances'] = R
    newNet = net
    return newNet, clustLearn


# -- clust vectors from dataCat --

def netClust (net, dataCat):
    
    t = 0
    w = net['weight']
    R = np.zeros(len(w[0,:]))
    clustResult = np.zeros(len(dataCat[0,:]))
    
    # ascertain that # of features equals # of weights per cluster
    if len(dataCat) != len(net['weight']):
        print ('\nNumber of features in testing data does not match the number of weight connections per neuron')
        sys.exit()
    for j in range (net['numFeatures']):
        maxVal = max(dataCat[j,:])
        
        # normalize test data
        for k in range (len(dataCat[0,:])):
            dataCat[j,k] = dataCat[j,k] / maxVal
    
    for j in range (len(dataCat[0,:])):
        
        # count distances between current vector and each cluster
        for k in range (len(w[0,:])):
            R[k] = dist.euclidean(dataCat[:,j], w[:,k])
        
        # determine cluster value
        t = list(R).index(min(R[np.nonzero(R)]))
        clustResult[j] = t + 1
        
    return clustResult

