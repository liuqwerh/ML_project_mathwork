import os,random
#os.environ["KERAS_BACKEND"] = "tensorflow"  #theano
import math
import numpy as np
import matplotlib.pyplot as plt
import pickle, random, sys
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data


# Load the dataset
#  from a certain local path
Xd = pickle.load(open("./RML2016.10a/RML2016.10a_dict.pkl",'rb'), encoding='latin1')
print("Dataset imported")
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])  # Xd[('QPSK', 2)].shape: (1000, 2, 128)
        for i in range(Xd[(mod,snr)].shape[0]):
            lbl.append((mod,snr))

X = np.vstack(X)  # X.shape: (220*1000, 2, 128)
# numpy.vstack(): Stack arrays in sequence vertically (row wise).

# For dataset RML2016.10a_dict, we should have data size 220000*2*128
print("Dataset formatted into shape: ",X.shape)  # (220000, 2, 128)
# print out the snrs and mods
print("Dataset with SNRs: ",snrs)
print("Dataset with Modulations: ",mods)
print("Data prepared")
# Dataset with SNRs:  [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
# Dataset with Modulations:  ['8PSK', 'AM-DSB', 'AM-SSB', 'BPSK', 'CPFSK', 'GFSK', 'PAM4', 'QAM16', 'QAM64', 'QPSK', 'WBFM']

# Partition the data
#  into training and test sets of the form we can train/test on
#  while keeping SNR and Mod labels handy for each
np.random.seed(2017)
n_examples = X.shape[0]
n_train = int(n_examples * 0.5) #110000
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
test_idx = list(set(range(0,n_examples))-set(train_idx))
X_train = X[train_idx] #print(X_train.shape, in_shp) # (110000, 2, 128) [2, 128]
X_test = X[test_idx]
#Xx = X.tolist()

#!!!
def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

#!!!
Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))
in_shp = list(X_train.shape[1:])   #print(X_train.shape, in_shp) # (110000, 2, 128) [2, 128]
classes = mods

LEARNING_RATE_BASE = 0.0007
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 107

global_step = tf.Variable(0,trainable = False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY,
                                           staircase=True)



#TENSORFLOW

#the size of each pitch
batch_size = 256
#calculate the number of pitches
#n_batch = mnist.train.num_examples // batch_size  110000/1024
n_batch = n_train // batch_size

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# def bias_variable(shape):
#      initial = tf.constant(0.1, shape=shape)
#      return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# def convv2d(x, W):
#     # stride [1, x_movement, y_movement, 1]
#     # Must have strides[0] = strides[3] = 1
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

#def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
#return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID'

xs=tf.placeholder(tf.float32,[None,2,128])/255
#xs = tf.placeholder(tf.float32, [None, 256])/255.   # 2x128
ys = tf.placeholder(tf.float32, [None, len(classes)]) #ys = tf.placeholder(tf.float32, [None, len(classes)])
keep_prob = tf.placeholder(tf.float32)
lr= tf.Variable(0.001, dtype=tf. float32)
x_signal = tf.reshape(xs, [-1, 2, 128, 1])    #[-1,1,2,128]

## conv1 layer ##
W_conv1 = weight_variable([1,4, 1,16]) # patch 1x4, in size 1, out size 64
#b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_signal, W_conv1))  # output size 2x128x64
b_conv1_drop = tf.nn.dropout(h_conv1, keep_prob)
## conv2 layer ##
W_conv2 = weight_variable([2,4, 16,32]) # patch 2x4, in size 64, out size 64
#b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(b_conv1_drop, W_conv2)) # output size 2x128x64
b_conv2_drop = tf.nn.dropout(h_conv2, keep_prob)
## conv3 layer ##
W_conv3 = weight_variable([1,8, 32,64]) # patch 5x5, in size 64, out size 128
#b_conv3 = bias_variable([128])
h_conv3 = tf.nn.relu(conv2d(b_conv2_drop, W_conv3))  # output size 2x128x128
b_conv3_drop = tf.nn.dropout(h_conv3, keep_prob)
## conv4 layer ##
W_conv4 = weight_variable([1,8, 64,128]) # patch 5x5, in size 128, out size 128
#b_conv4 = bias_variable([128])
h_conv4 = tf.nn.relu(conv2d(b_conv3_drop, W_conv4))  # output size 2x128x128
h_conv4_drop = tf.nn.dropout(h_conv4, keep_prob)

## fc1 layer ##
W_fc1 = weight_variable([2*128*128, 256])
#b_fc1 = bias_variable([256])
# [n_samples, 2, 128, 128] ->> [n_samples, 2*128*128]
h_conv4_flat = tf.reshape(h_conv4_drop, [-1, 2*128*128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv4_flat, W_fc1))
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 256
## fc2 layer ##
W_fc2 = weight_variable([256, len(classes)]) #W_fc2 = weight_variable([256, len(classes)])
#b_fc2 = bias_variable([len(classes)]) #b_fc2 = bias_variable([len(classes)])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2))


#loss= tf.nn.softmax_cross_entropy_with_logits(labels =ys,logits =prediction)
cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels =ys,logits =prediction))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy) #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(prediction, 1),tf.argmax(ys,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(60):
        #sess.run(tf.assign(lr, 0.001 * (0.95 ** epoch)))
        for i in range(n_batch):
            start = i * 256
            end =(i+1)*256
            #batch_xs,batch_ys = Xx.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={xs:X_train[start:end],ys:Y_train[start:end],keep_prob:0.7})
            #loss = sess.run(loss, feed_dict={xs: X_train[start:end], ys: Y_train[start:end], keep_prob: 0.7})
            for batch in range(n_batch):
                startt = batch* 256
                endd = (batch+1)*256
        TEST_ACC= sess.run(accuracy,feed_dict={xs:X_test[startt:endd],ys:Y_test[startt:endd],keep_prob:0.7})
        TRAIN_ACC = sess.run(accuracy, feed_dict={xs: X_train[startt:endd], ys: Y_train[startt:endd], keep_prob: 0.7})
        test_y1 = sess.run(prediction, feed_dict={xs: X_test[startt:endd],ys:Y_test[startt:endd],keep_prob: 0.7})
        test_y2 = sess.run(prediction, feed_dict={xs: X_train[startt:endd], ys: Y_train[startt:endd], keep_prob: 0.7})
        #test_y=tf.argmax(test_y1, 1)
        #test_Y_i_hat = sess.run(prediction, feed_dict={xs: test_X_i[startt:endd], keep_prob: 0.7})
        #test_y = sess.run(prediction, feed_dict={xs: X_test, keep_prob: 0.7})
        print("Iter" + str(epoch) + ", Testing Accuracy= " + str(TEST_ACC)+ ", Training Accuracy= " + str(TRAIN_ACC),"predoct="+str(test_y1))

'''
# Optional: show analysis graphs
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.show()
'''




conf=np.argmax(test_y2,1)
confnorm=np.argmax(test_y1,1)
 # test_y = sess.run(prediction, feed_dict={xs: X_test, keep_prob: 0.7})
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    confuse_martix=sess.run(tf.convert_to_tensor(tf.confusion_matrix(conf,confnorm)))
print(confuse_martix)




# Accuracy and confusion matrix for data with each SNR
acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]

    # estimate classes
    #test_Y_i_hat = model.predict(test_X_i)
    with tf.Session() as sess:
         sess.run(tf.global_variables_initializer())
         test_Y_i_hat = sess.run(prediction, feed_dict={xs: test_X_i, keep_prob: 0.7})
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])

    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1

    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])

    #plt.figure()
    #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("SNR: ",snr, " Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)


# Save results to a pickle file for plotting later
print(acc)
fd = open('results_cnn_d0.5.dat','wb')
pickle.dump(("CNN", 0.5, acc), fd)

# Plot accuracy curve
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("New Model Classification Accuracy on RadioML 2016.10 Alpha")
plt.show()


