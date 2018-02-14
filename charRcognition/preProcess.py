'''
Created on 2018/01/24

@author: hanbing.cheng
'''

import os
import cv2
import numpy as np
import pandas as pd
from pandas import DataFrame
from charRcognition.featureExtraction import getFeatures
from charRcognition.featureExtraction import featuresDetector
from common.utils import asColumnMatrix
ROOT_PATH = '../dataset/Char_Image/'

def split_char_dataset():
    #split image for char_image
    
    data = pd.read_csv(ROOT_PATH + 'Char_Index.txt', sep = "\t", header = 0)
    #print (data.category.value_counts(ascending=True))
    labels = data.category.unique()
    train_data = DataFrame(columns = ['id', 'category', 'image'])
    test_data = DataFrame(columns = ['id', 'category', 'image'])
    for (_, label) in enumerate(labels):
        df = data[data.category == label]
        df= df.reset_index(drop=True)
        train_num =  int(df.shape[0] * 0.7)
        print ("train_num={}".format(train_num))
        print (df[0:train_num].shape)
        train_data = train_data.append(df[0:train_num])
        test_data = test_data.append(df[train_num:])
    
    train_data= train_data.reset_index(drop=True)
    train_data.to_csv(ROOT_PATH + 'train/Char_Index.txt', sep='\t', encoding='utf-8')
    for i in range(train_data.shape[0]):
        src = ROOT_PATH + ' ' + train_data.image[i]
        target = ROOT_PATH + 'train/' +  train_data.image[i]
        os.rename(src, target)
    
    test_data= test_data.reset_index(drop=True)
    test_data.to_csv(ROOT_PATH + 'test/Char_Index.txt', sep='\t', encoding='utf-8')
    for i in range(test_data.shape[0]):
        src = ROOT_PATH + ' ' + test_data.image[i]
        target = ROOT_PATH + 'test/' +  test_data.image[i]
        os.rename(src, target)

   
def getImageData(directory):
    
    data  = []
    label = []
    image_infos = pd.read_csv(directory + 'Char_Index.txt', sep = "\t", header = 0)
    for i in range(image_infos.shape[0]):
        img = cv2.imread(os.path.join(directory, image_infos.image[i]))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # resize to () for svm
        img = cv2.resize(img,(20,40),interpolation=cv2.INTER_CUBIC)
        # resize to (10, 20) for knn and ann
        #img = cv2.resize(img,(10,20),interpolation=cv2.INTER_CUBIC)
        
        img = cv2.GaussianBlur(img, (5,5), 0)    
        
        #img = cv2.inpaint(img,img,3,cv2.INPAINT_TELEA)
        #ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)         
        #ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_TRIANGLE)
        #ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
        
        # get features from image
        features = getFeatures(featuresDetector.HOG, img)
        data.append(features) 
        label.append(image_infos.category[i])
    
    data = np.array(data)
    data = data.reshape(data.shape[0], data.shape[1])    
    return (data, np.array(label, dtype = np.int))

def preprocessing(mode):
    if mode == 'train':
        # get train image 
        data, label = getImageData(os.path.join(ROOT_PATH, 'train/'))
        return data, label
    else:    
        data, label = getImageData(os.path.join(ROOT_PATH, 'test/'))
        return data, label
