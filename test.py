import numpy as np
import json
with open('input.json') as file:
    data = np.asarray(json.load(file))
#Create numpy columns of all (X**i)*(Y**j) and i+j<=4
X = (data[:,0])
Y = (data[:,1])
X = X.reshape(len(X),1)
Y = Y.reshape(len(Y),1)
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
##########################################
def sigmoid_activation(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def predict(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(m):
        if final_pred[i,0] > 0.5:
            y_pred[0,i] = 1
    return y_pred

# the parameter values obtained from train.py
b = 1.4425492655807501
theta = np.array([23.078975674552215,
 0.3264751456336108,
 61.58151125716585,
 -2.6401218463948846,
 -37.93634915833223,
 -5.891931639531941,
 1.467232118700157,
 52.03935664326999,
 -2.270210770769578,
 74.33431517798333,
 -4.112237508495853,
 9.106420679316116,
 4.5764034989050675,
 -36.18536534220967])
theta = theta.reshape((14,1))
ans = np.dot(X,theta)
ans = ans + b
final_pred = sigmoid_activation(ans)
predicted = predict(final_pred, final_pred.shape[0])
predicted = predicted.reshape(len(predicted),1)
with open('input.json') as file:
    d = np.asarray(json.load(file))
ans = np.hstack((d,predicted))
result = json.dumps(ans.tolist())
with open("result.json", "w") as output_file:
    output_file.write(result)