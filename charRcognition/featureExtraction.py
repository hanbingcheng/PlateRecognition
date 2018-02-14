'''
Created on 2018/01/26

@author: hanbing.cheng
'''
import cv2
from enum import Enum

class featuresDetector(Enum):
    Agast = "Agast"
    HOG = "HOG"
    AKAZE = "AKAZE"
    BRISK = "BRISK"
    FAST = "FAST"
    KAZE = "KAZE"
    MSER = "MSER"
    ORB = "ORB"
    SIFT = "SIFT"
    SimpleBlob = "SimpleBlob"
    
    def __str__(self):
        return self.value

def getFeatures(mode, img):
    if mode == featuresDetector.HOG:
        return getFeaturesByHog(img)
        
    else: 
        getFeaturesByDetector(mode, img)
    
def getFeaturesByHog(img):
    # set parameter for hog
    #parameter for hogdescriptor

    winSize = (10, 20)
    blockSize = (4, 4)
    blockStride = (2, 2)
    cellSize = (2, 2)    
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    return hog.compute(img)
    
def getFeaturesByDetector(mode,img):
    
    detector = cv2.SimpleBlobDetector_create()
    
    if mode == featuresDetector.Agast:
        detector = cv2.AgastFeatureDetector_create()
    elif mode == featuresDetector.AKAZE:
        detector = cv2.AKAZE_create()
    elif mode == featuresDetector.BRISK:
        detector = cv2.BRISK_create()
    elif mode == featuresDetector.FAST:
        detector = cv2.FastFeatureDetector_create()
    elif mode == featuresDetector.KAZE:
        detector = cv2.KAZE_create()
    elif mode == featuresDetector.MSER:
        detector = cv2.MSER_create()
    elif mode == featuresDetector.ORB:
        detector = cv2.ORB_create()
    elif mode == featuresDetector.SIFT:
        detector = cv2.xfeatures2d.SIFT_create()
    elif mode == featuresDetector.SimpleBlob:
        detector = cv2.SimpleBlobDetector_create()
    
    keypoints = detector.detect(img)
    descriptors = detector.compute(img, keypoints)
    
    return descriptors









def getFeaturesByFAST(img):
    
    detector = cv2.FastFeatureDetector_create()
    keypoints = detector.detect(img)
    descriptors = detector.compute(img, keypoints)
    
    return descriptors

def getFeaturesByAgastFeatureDetector(img):
    
    
    keypoints = detector.detect(img)
    descriptors = detector.compute(img, keypoints)
    
    return descriptors
    
    