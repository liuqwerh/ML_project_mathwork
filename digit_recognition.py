import numpy as np
import tensorflow as tf
import cv2


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
 
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
 
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
 
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')#maxpool 2*2
 
def network(x):
    x_image = tf.reshape(x, [-1,28,28,1]) #-1 means arbitrary
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  #conv1
    h_pool1 = max_pool(h_conv1)                               #max_pool1
 
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  #conv2
    h_pool2 = max_pool(h_conv2)                               #max_pool2
 
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) #fc1
    
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)               #dropout
 
    y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) #fc2 output
    return y_predict

    #       frame                 ROI
    # 1 (0,0),(280,280)     [0:280,0:280]
    # 2 (0,560),(280,840)   [560:840,0:280]
    # 3 (0,280),(280,560)   [280:560,0:280]
    # 5 (560,280),(840,560) [280:560,560:840]
    # 6 (280,280),(560,560) [280:560,280:560]
    # 7 (280,560),(560,840) [560:840,280:560]
    # 8 (280,0),(560,280)   [0:280,280:560]
    # 9 (560,560),(840,840) [560:840,560:840]
    # 0 (560,0),(840,280)   [0:280,560:840]

keep_prob = tf.placeholder("float")
W_conv1 = weight_variable([5, 5, 1, 32])#conv 5*5
b_conv1 = bias_variable([32])
W_conv2 = weight_variable([5, 5, 32, 64])#conv 5*5
b_conv2 = bias_variable([64])
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
sess=tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, "./model_save.ckpt") #WIN load model file must have ./ with tensorflow1.0

# capture = cv2.VideoCapture(1)
# ref,frame = capture.read()#camera
frame = cv2.imread('./all_rev_2.png')#white bg
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
ref,binary = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)#fixed threshold binarization
gaussian = cv2.GaussianBlur(binary, (3, 3), 0)
circles1 = cv2.HoughCircles(gaussian, cv2.HOUGH_GRADIENT, 1, 800, param1=100, param2=24, minRadius=15, maxRadius=50)#detect 3 circle
#print(np.shape(circles1))
circles = circles1[0, :, :]
circles = np.int16(np.around(circles))#signed，abs,np.uint16(np.around(circles))
print(circles)
for i in circles[:]:
    cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 3)#cirecle，img，center，r，color，thick
    cv2.circle(frame, (i[0], i[1]), 2, (255, 0, 255), 10)#center draw

#calculate distance
dis_1 = abs(circles[0,0] - circles[1,0])
dis_2 = abs(circles[1,0] - circles[2,0])
dis_3 = abs(circles[0,0] - circles[2,0])
dis = max(dis_1, dis_2, dis_3) - circles[0,2]*2
ROIdis = int(dis/3)#simple ROI
#print(ROIdis)

#find left coordinate
start_x = min(circles[0,0], circles[1,0], circles[2,0])
start_y = min(circles[0,1], circles[1,1], circles[2,1])
radius = max(circles[0,2], circles[1,2], circles[2,2])
start_x = start_x + radius
start_y = start_y + radius
#print(start_x)
#print(start_y)
#print(radius)

#TF
cv2.bitwise_not(frame, frame)#black bg
i = 0
print('which number do you want:')
num = input()
num = int(num)
#print(num)

#num  1     2    3     5    6    7    8    9    0
#x1 = [0,    0,   0,   560, 280, 280, 280, 560, 560]
#y1 = [0,    560, 280, 280, 280, 560, 0,   560,   0]
#x2 = [280,  280, 280, 840, 560, 560, 560, 840, 840]
#y2 = [280,  840, 560, 560, 560, 840, 280, 840, 280]

#ROI frame
x1 = [0,        ROIdis, ROIdis*2,        0,   ROIdis, ROIdis*2,        0,   ROIdis, ROIdis*2]
y1 = [0,             0,        0,   ROIdis,   ROIdis,   ROIdis, ROIdis*2, ROIdis*2, ROIdis*2]
x2 = [ROIdis, ROIdis*2, ROIdis*3,   ROIdis, ROIdis*2, ROIdis*3,   ROIdis, ROIdis*2, ROIdis*3]
y2 = [ROIdis,   ROIdis,   ROIdis, ROIdis*2, ROIdis*2, ROIdis*2, ROIdis*3, ROIdis*3, ROIdis*3]

for i in range(9):
    roiImg = frame[start_y + y1[i] + 4:start_y + y2[i] - 4, start_x + x1[i] + 4:start_x + x2[i] - 4]#y1:y2,x1:x2
    #roiImg = frame[start_y + y1[i]:start_y + y2[i], start_x + x1[i]:start_x + x2[i]]#y1:y2,x1:x2
    img = cv2.resize(roiImg,(28,28))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_img = img.astype(np.float32)

    netoutput = network(np_img)
    predictions = sess.run(netoutput,feed_dict={keep_prob: 0.5})

    predicts=predictions.tolist() #tensorflow output is numpy.ndarray like [[0 0 0 0]]
    label=predicts[0]
    result=label.index(max(label))
    #print(result)

    if result==num:
        print("show")
        print("Any key to exit!")
        cv2.bitwise_not(frame, frame)#white bg
        cv2.rectangle(frame,(start_x + x1[i] + 4,start_y + y1[i] + 4),(start_x + x2[i] - 4,start_y + y2[i] -4 ),(0,0,255),2)#img，start(x1,y1)，end(x2,y2)，color，thick
        cv2.imshow("img", frame)
        cv2.waitKey()
        break
        
cv2.destroyAllWindows()

