# kohonen-net

This project is created for breaking data vectors into clusters, when number of classes is unknown beforehand.

The project includes following files:
netfunc.py
netfunc.ipynb
clustering.py
clustering.ipynb
480.csv
README.md

- netfunc.py
The file contains following functions:

	netInit: initializes parameters of neural network. Its input value is number of features, and it returns the network configuration.
	Threshold distance between vectors (disThreshold), learning rate (learningRate), value of rate decrease in each epoch (rateDecrease) and number of epochs (numEpochs)
	can be varied in order to achieve greater performance in a particular task.
	
	netLearn: learns neural network. Its inputs are network configuration and learning data array, and it returns learned network configuration and vector with cluster numbers of learning vectors.
	Learning data array should be an array of numbers with data vectors placed in columns.
	
	netClust: breaks test data into clusters. Its inputs are learned network configuration and test data array, and it returns vector with cluster numbers of test vectors.

- clustering.py
Here clustering is performed. Data from example file 480.csv is used, where each row is a data vector with 7 parameters.
You can process another file, updating "load and prepare data from 480.csv" block
