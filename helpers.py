import numpy as np
import matplotlib.pyplot as plt

def make_binary(y):
    for i in range(len(y)):
        y[i]=np.sign(y[i])
        if y[i]==0:
            y[i]=1
    return y

def save_csv(y,filename="pred.csv"):
    id_=np.arange(35000,35000+len(y))
    GD_pred=np.c_[id_,y.astype(int)]
    np.savetxt(filename, GD_pred, delimiter=",",header="Id,Prediction")