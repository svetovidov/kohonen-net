
# coding: utf-8

# In[2]:

import netfunc as nf
import numpy as np
import time
from sklearn.manifold import TSNE

start_time = time.time()
kohNet = {}
kohNetLearned = {}

#  load and prepare data from 480.csv

data = np.loadtxt("C:\\documents\\480.csv", delimiter = ';', dtype = np.float32)
data = data.transpose()
data = data[2:-1, :] # cut redundant data
border = round(0.9*len(data[0,:]))
learnVec = data[:, :border] # learning data vectors
testVec = data[:, border:] # test data vectors

# initialize kohonen network

kohNet = nf.netInit (7)

# teach network on learnVec

kohNetLearned, clustLearnVec = nf.netLearn (kohNet, learnVec)

# test network

clustTestVec = nf.netClust(kohNetLearned, testVec)

# reduce data dimension to 3 by using TSNE (for possible visualising)

data_embedded = TSNE(n_components=3, method='barnes_hut').fit_transform(data.transpose())
data_embedded = data_embedded.transpose()

# output results

clustVec = np.concatenate ((clustLearnVec, clustTestVec))
print ("\nClusters of data\n", clustVec, "\n")
print ("Number of classes: ", kohNetLearned['numClusters'])
print ("\nEmbedded data\n", data_embedded)
print ('\n-- The algorithm took', np.float16(time.time() - start_time), 'seconds to complete --' ) # execution time


# In[ ]:



