import numpy as np
import json
with open('data.json') as file:
    data = np.asarray(json.load(file))
#Create numpy columns of all (X**i)*(Y**j) and i+j<=4
X = (data[:,0])
Y = (data[:,1])
X = X.reshape(len(X),1)
Y = Y.reshape(len(Y),1)
label = data[:,-1].astype('int64')
X4 = X**4
Y4 = Y**4
X3Y = (X**3)*Y
X2Y2 = (X**2)*(Y**2)
XY3 = X*(Y**3)
X3 = X**3
X2Y = X*X*Y
XY2 = X*Y*Y
Y3 = Y**3
X2 = X*X
XY = X*Y
Y2 = Y*Y
#Make a matrix of all created features
X = np.hstack((X,Y,X2,XY,Y2,X3,X2Y,XY2,Y3,X4,X3Y,X2Y2,XY3,Y4))
label = label.reshape(len(label),1) #making into column
# training part 
##########################################
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def param_finder(X,label):
    lr = 10 #the learning rate
    W = np.ones((1,14)) 
    b = 1
    for i in range(100000):
        final_result = sigmoid_activation(np.dot(W,X.T) + b)
        grad = (1/6000)*(np.dot(X.T, (final_result-label.T).T))
        db = (1/6000)*(np.sum(final_result-label.T))
        W = W - lr*(grad.T)
        b = b - lr*db    
    return b, W
b,thetas = param_finder(X, label)
############################################
#Determining cost, which turns out to be 0.008
final_result = sigmoid_activation(np.dot(thetas,X.T) + b)
label_T = label.T
cost = (-1/6000)*(np.sum((label_T*np.log(final_result + 0.00000000001)) + ((1-label_T)*(np.log(1-final_result + 0.00000000001)))))
#Creating a list of parameters (thetas) in increasing degree
l = list(thetas[0])
l = [b] + l

#l is ..
#[1.4425492655807501,
# 23.078975674552215,
# 0.3264751456336108,
# 61.58151125716585,
# -2.6401218463948846,
# -37.93634915833223,
# -5.891931639531941,
# 1.467232118700157,
# 52.03935664326999,
# -2.270210770769578,
# 74.33431517798333,
# -4.112237508495853,
# 9.106420679316116,
# 4.5764034989050675,
# -36.18536534220967]