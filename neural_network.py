import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from sklearn.model_selection import KFold

def Find_Min(Data1, Data2):
    i = 0
    for iterating_var in Data1:
        if(iterating_var > Data2):
            i=i+1
        elif(iterating_var == Data2):
            return i
    return i

X_train = np.loadtxt('Xtrain.txt', delimiter =',')
X_test = np.loadtxt('Xtest.txt', delimiter =',')
plt.scatter(X_train[0, :], X_train[1, :], color='red')
plt.show()
plt.scatter(X_test[0, :], X_test[1, :], color='green')
plt.show()

acc = np.zeros((10, 10))
x_data = X_train[0, :]
y_data = X_train[1, :]
x_data_test = X_test[0, :]
y_data_test = X_test[1, :]
kf = KFold(n_splits=10)
kf_T = 0
ind= np.array([2,3,4,5,6,7,8,9,10,11])
for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    for i in range(10):
        model = Sequential()
        model.add(Dense(ind[i], activation='sigmoid', input_dim=1))
        model.add(Dense(1))
        model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
        model.fit(x_train, y_train, batch_size=1, epochs=9,verbose=1)
        score1 = model.evaluate(x_test, y_test,verbose=0)
        acc[kf_T, i] = score1
    kf_T=kf_T+1
Means_row = np.mean(acc, axis=1)
Best_order1 = Find_Min(Means_row, min(Means_row))
plt.plot(range(2, 12), Means_row)
plt.title('different Preceptron')
plt.xlabel('Perceptrons')
plt.ylabel('loss')
plt.show()

model = Sequential()
model.add(Dense(ind[Best_order1], activation='sigmoid', input_dim=1))
model.add(Dense(1))
model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
model.fit(x_data, y_data,  batch_size=1, epochs=40,verbose=1)
score11 = model.evaluate(x_data_test, y_data_test)


kf_T=0
for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    for i in range(10):
        model = Sequential()
        model.add(Dense(ind[i], activation='softplus', input_dim=1))
        model.add(Dense(1))
        model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
        model.fit(x_train, y_train, batch_size=1, epochs=9,verbose=1)
        score2 = model.evaluate(x_test, y_test,verbose=0)
        acc[kf_T, i] = score2
    kf_T= kf_T+1
Means_row = np.mean(acc, axis=1)
Best_order2 = Find_Min(Means_row, min(Means_row))

plt.plot(range(2, 12), Means_row)
plt.title('different Preceptron')
plt.xlabel('Perceptrons')
plt.ylabel('loss')
plt.show()

model = Sequential()
model.add(Dense(ind[Best_order2], activation='softplus', input_dim=1))
model.add(Dense(1))
model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
model.fit(x_data, y_data,batch_size=1, epochs=40,verbose=1)
score22 = model.evaluate(x_data_test, y_data_test,verbose=0)


if (score11<score22):
    choice='sigmoid'
    Best_order=Best_order1

else:
    choice='softplus'
    Best_order = Best_order2

model = Sequential()
model.add(Dense(ind[Best_order], activation=choice, input_dim=1))
model.add(Dense(1))
model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.adam(lr=0.001))
model.fit(x_data, y_data,batch_size=1, epochs=40,verbose=1)
score3 = model.evaluate(x_data_test, y_data_test,verbose=0)
print('the best result of sigmoid is:  ' + str(score11))
print('the best result of softplus is:  ' + str(score22))
print('the final result of the better function is:  '+ str(score3))

Predict = model.predict(X_test[0, :])
fig = plt.figure()
plt.subplot(121)
plt.scatter(X_test[0, :], X_test[1, :], color='red')
plt.title('Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.subplot(122)
plt.title('Predict_result')
plt.scatter(X_test[0, :], Predict, color='green')
plt.xlabel('X')
plt.ylabel('Predict')
plt.show()
