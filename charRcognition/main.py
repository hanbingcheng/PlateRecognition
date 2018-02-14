'''
Created on 2018/01/24

@author: teikanhei
'''
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from charRcognition.preProcess import preprocessing
from charRcognition.charSvmModel import CharSvmModel
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from charRcognition.featureExtraction import featuresDetector
from datashape.coretypes import char
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

char_dic = {10: "京", 11: "渝" , 12: "鄂" ,
                20: "0", 22: "2",  25 :  "5" , 26 : "6", 28: "8",
                30:  "A", 31: "B", 32: "C", 33: "D", 34: "Q"}

def getChar(category):
    return char_dic.get(category)
    
         
if __name__ == "__main__":
    
    print ("start...")
    # train
    train_data, train_label =  preprocessing('train')  
    print (train_data.shape)
    print (train_label.shape)
    print ("data process finished")
    # use SVM classfier
    clf =  CharSvmModel()
    
    # use kNN Classifier
    #clf = neighbors.KNeighborsClassifier()  
    
    # use ANN Classifier
    #clf = MLPClassifier(hidden_layer_sizes=(train_data.shape[1],100,13),max_iter=500)
    # use AdaBoostClassifier 
    #clf = AdaBoostClassifier(SVC(probability=True, kernel='linear'), n_estimators=5, learning_rate=1.0, algorithm='SAMME')
    #dt_stump=DecisionTreeClassifier(max_depth=8,min_samples_leaf=1)
    #clf = AdaBoostClassifier(dt_stump, n_estimators=10, learning_rate=1.0, algorithm='SAMME.R')
    
    clf.fit(train_data, train_label)
    print ("train finished")
    # test
    test_data, test_label = preprocessing('test') 
    prediction = clf.predict(test_data)#
    print ("prediction finished")
    score = accuracy_score(test_label, prediction)
    print ('accuracy_score: {:.5f}'.format(score))
    target_names = [str(val) for val in np.unique(test_label)]
    print(classification_report(test_label, prediction, target_names = target_names))
    for i in range(test_label.shape[0]):
        if test_label[i] != prediction[i]:
            print ("index:{}, true:{}, prediction:{}".format(i, getChar(test_label[i]), getChar(prediction[i])))

