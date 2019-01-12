# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 03:34:45 2019

@author: priya
"""
import cv2                 
import numpy as np         
import os                  
from random import shuffle 
from tqdm import tqdm     
TRAIN_DIR ='F:/dogs-cats-images/dog vs cat/dataset/training_set/train'
TEST_DIR = 'F:/dogs-cats-images/dog vs cat/dataset/test_set/test'

IMG_SIZE =50
LR = 1e-3

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

def label_img (img):
    word_label=img.split('.')[0]
    if word_label =='cat':return [1,0]
    if word_label=='dog':return [0,1]

def create_train_data():
    training_data=[]
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label=label_img(img)
        path=os.path.join(TRAIN_DIR,img)
        path2=path.split('\\')
        path=path2[0]+'/'+path2[1]
        path2=str(path)
        img=cv2.imread('{}'.format(path2),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
        else:
            print(" train image not loaded")
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

train_data=create_train_data()

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        path2=path.split('\\')
        path=path2[0]+'/'+path2[1]
        path2=str(path)
        
        
        img_num = img.split('.')[0]
        img=cv2.imread('{}'.format(path2),cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img), img_num])
        else:
            print(" testing image not loaded")
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=3, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
