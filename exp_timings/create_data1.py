import numpy as np

data_sizes = [2**10, 2**11, 2**12, 2**13, 2**14, 2**15]

for i in range(len(data_sizes)):
    X = np.vstack((
    np.random.normal(loc=[0,0], scale=1, size=(data_sizes[i]//3,2)),
    np.random.normal(loc=[0,10], scale=1, size=(data_sizes[i]//3,2)),
    np.random.normal(loc=[10,0], scale=1, size=(data_sizes[i]//3,2))
    ))
    y = np.hstack((
    np.zeros((data_sizes[i]//3)),
    np.zeros((data_sizes[i]//3))+1,
    np.zeros((data_sizes[i]//3))+2
    ))
    #np.savez_compressed('data_' +str(i) +'.npz', X=X, y=y)
    
