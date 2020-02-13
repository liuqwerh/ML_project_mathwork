import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

def Find_Max(Data1, Data2):
    i = 0
    for iterating_var in Data1:
        if(iterating_var < Data2):
            i=i+1
        elif(iterating_var == Data2):
            return i
    return i

N = 1000
n = 3
mu = np.array([[2, -2, -2, 2], [2, 2, -2,-2], [3, 3, 3, 3]])
Sigma = np.array([[[1, 0.2, 0.4],[0.2, 1, 0.4],[0.4, 0.4, 1]],
                  [[2, 0.3, 1],[0.3, 2, 0],[1, 0, 2]],
                  [[3, 0.5, 1],[0.5, 3, 0],[1, 0, 3]],
                  [[4, 0.5, 0],[0.5, 4, 0.2],[0, 0.2, 4]]])
q = np.array((0.4, 0.3, 0.2, 0.1))
prior_range = np.array((0, 0.4, 0.7, 0.9, 1))
x1= np.zeros((N,3))
ran1 = np.random.rand(1, N)
Label1 = np.zeros(N)
for l in range(4):
    index1 = np.where((ran1 >= prior_range[l]) & (ran1 < prior_range[l + 1]))
    Label1[index1[1]] = l
    x1[index1[1], :] = np.random.multivariate_normal(mu[:, l], Sigma[l, :, :], np.size(Label1[index1[1]]))
fig = plt.figure()
pic = fig.add_subplot(111, projection='3d')
colorr=('r','b','c','g')
for i in range(1000):
    if(Label1[i]==0):
        pic.scatter(x1[i, 0], x1[i, 1], x1[i, 2], color=colorr[0])
    elif(Label1[i]==1):
        pic.scatter(x1[i, 0], x1[i, 1], x1[i, 2], color=colorr[1])
    elif (Label1[i] == 2):
        pic.scatter(x1[i, 0], x1[i, 1], x1[i, 2], color=colorr[2])
    elif (Label1[i] == 3):
        pic.scatter(x1[i, 0], x1[i, 1], x1[i, 2], color=colorr[3])
pic.set_xlabel('X')
pic.set_ylabel('Y')
pic.set_zlabel('Z')
plt.title('the 1000 samples in Part 1')
plt.show()


N = 10000
x2 = np.zeros((N, 3))
ran2 = np.random.rand(1, N)
Label2  = np.zeros(N)
for l in range(4):
    index2 = np.where((ran2 >= prior_range[l]) & (ran2 < prior_range[l + 1]))
    Label2[index2[1]] = l
    x2[index2[1], :] = np.random.multivariate_normal(mu[:, l], Sigma[l, :, :], np.size(Label2[index2[1]]))
fig = plt.figure()
pic = fig.add_subplot(111, projection='3d')
colorr=('r','b','c','g')
for i in range(10000):
    if(Label2[i]==0):
        pic.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color=colorr[0])
    elif(Label2[i]==1):
        pic.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color=colorr[1])
    elif (Label2[i] == 2):
        pic.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color=colorr[2])
    elif (Label2[i] == 3):
        pic.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color=colorr[3])
pic.set_xlabel('X')
pic.set_ylabel('Y')
pic.set_zlabel('Z')
plt.title('the 10000 samples in Part 2')
plt.show()


eval0 = multivariate_normal(mean=mu[:,0], cov=Sigma[0,:,:]).pdf(x2) * q[0]
eval1 = multivariate_normal(mean=mu[:,1], cov=Sigma[1,:,:]).pdf(x2) * q[1]
eval2 = multivariate_normal(mean=mu[:,2], cov=Sigma[2,:,:]).pdf(x2) * q[2]
eval3 = multivariate_normal(mean=mu[:,3], cov=Sigma[3,:,:]).pdf(x2) * q[3]
index00 = np.where((eval0 > eval1) & (eval0 > eval2) & (eval0 > eval3))
index11 = np.where((eval1 > eval0) & (eval1 > eval2) & (eval1 > eval3))
index22 = np.where((eval2 > eval0) & (eval2 > eval1) & (eval2 > eval3))
index33 = np.where((eval3 > eval0) & (eval3 > eval1) & (eval3 > eval2))
Deci = np.zeros(N)
Deci[index00[0]] = 0
Deci[index11[0]] = 1
Deci[index22[0]] = 2
Deci[index33[0]] = 3
mis1=len(np.where((Label2==0) & (Deci != 0))[0])
mis2=len(np.where((Label2==1) & (Deci != 1))[0])
mis3=len(np.where((Label2==2) & (Deci != 2))[0])
mis4=len(np.where((Label2==3) & (Deci != 3))[0])
error_pro=(mis1+mis2+mis3+mis4)/10000
print('the error probablity of MAP: ' + str(error_pro))
fig = plt.figure()
picc = fig.add_subplot(111, projection='3d')

for i in range(10000):
    if (Label2[i] == Deci[i]):
        picc.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color='green', label='true')
    else:
        picc.scatter(x2[i, 0], x2[i, 1], x2[i, 2], color='red', label='false')
picc.set_xlabel('x0')
picc.set_ylabel('x1')
picc.set_zlabel('x2')
plt.title('Map result(green is correct)')
plt.show()


N = 100
x_test100 = np.zeros((N,3))
ran_test100 = np.random.rand(1, N)
Label_100  = np.zeros(N)
for l in range(4):
    index_100 = np.where((ran_test100 >= prior_range[l]) & (ran_test100 < prior_range[l + 1]))
    Label_100[index_100[1]] = l
    x_test100[index_100[1], :] = np.random.multivariate_normal(mu[:, l], Sigma[l, :, :], np.size(Label_100[index_100[1]]))


N = 1000
x_test1000 = np.zeros((N,3))
ran_test1000 = np.random.rand(1, N)
Label_1000  = np.zeros(N)
for l in range(4):
    index_1000 = np.where((ran_test1000 >= prior_range[l]) & (ran_test1000 < prior_range[l + 1]))
    Label_1000[index_1000[1]] = l
    x_test1000[index_1000[1], :] = np.random.multivariate_normal(mu[:, l], Sigma[l, :, :], np.size(Label_1000[index_1000[1]]))


N = 10000
x_test10000 = np.zeros((N,3))
ran_test10000 = np.random.rand(1, N)
Label_10000  = np.zeros(N)
for l in range(4):
    index_10000 = np.where((ran_test10000 >= prior_range[l]) & (ran_test10000 < prior_range[l + 1]))
    Label_10000[index_10000[1]] = l
    x_test10000[index_10000[1], :] = np.random.multivariate_normal(mu[:, l], Sigma[l, :, :], np.size(Label_10000[index_10000[1]]))


x_data = x_test10000
y_data = keras.utils.to_categorical(Label_10000, num_classes=4)
kf = KFold(n_splits=10)
kf_T = 0
ind= np.array([1,2,3,4,5,6,7,8,9,10])
acc = np.zeros((10, 10))
for train_index, test_index in kf.split(x_data):
    x_train, x_test = x_data[train_index], x_data[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]
    for i in range(10):
        model = Sequential()
        model.add(Dense(ind[i], activation='relu', input_dim=3))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy',optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True),metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=5, epochs=30, verbose=1)
        score = model.evaluate(x_test, y_test)
        acc[kf_T, i] = score[1]
    kf_T= kf_T + 1
Means_row = np.mean(acc, axis=1)
Best_order = Find_Max(Means_row, max(Means_row))

plt.plot(range(1, 11), Means_row)
plt.title('different Preceptron')
plt.xlabel('Perceptrons')
plt.ylabel('Right Probability')
plt.show()

y_test10000 = keras.utils.to_categorical(Label2, num_classes = 4)
model = Sequential()
model.add(Dense(ind[Best_order], activation='relu', input_dim=3))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True), metrics=['accuracy'])
model.fit(x_data, y_data, batch_size=5, epochs=50)
score = model.evaluate(x2, y_test10000)
print("The final accuracy: " + str(score[1]))
