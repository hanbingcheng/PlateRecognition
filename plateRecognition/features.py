'''
Created on 2018/02/04

@author: teikanhei
'''

import cv2
import glob
from common import Configuration
import numpy as np
from skimage.feature import hog
from common import imageTools
import mahotas.thresholding

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8),cellsPerBlock = (3, 3), visualise=False, normalize = False): 
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.visualise = visualise
        self.normalize = normalize

    def describe(self, image):
        hist , hog_image= hog(image,
                            orientations = self.orienations,
                            pixels_per_cell = self.pixelsPerCell,
                            cells_per_block = self.cellsPerBlock,
                            visualise= self.visualise)
                            # normalise = self.normalize)  

        return hist, hog_image
    

class CreateFeatures():    

    def __init__(self):
        self.conf = Configuration.get_datamodel_storage_path()
        self.train_path = [self.conf['Train_data1_dir'],self.conf['Train_data2_dir']]
    
    def hog_feature(self, image_orig):
        # print ('CreateFeatures!! hog_feature')
        #hog= a_HOG_computation.HOG(orientations=18, pixelsPerCell=(9,9), cellsPerBlock=(3,3), visualise=True, normalize=True) 
        #hog= a_HOG_computation.HOG(orientations=9, pixelsPerCell=(6,6), cellsPerBlock=(1,1), visualise=True, normalize=True)     
        hog = HOG(orientations=9, pixelsPerCell=(9,9), cellsPerBlock=(3,3), visualise=True, normalize=True)
        # Convert the coloured image into grayscale image
        image_resized = imageTools.resize(image_orig, width=140)
        image_gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
        # Do thresholding
        image_thresh = image_gray
        T = mahotas.thresholding.otsu(image_gray) # will find an optimal value of T from the image

        # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is white
        image_thresh[image_thresh > T] = 255    
        # This goes pixel by pixel if the pixel value of the thresh is greater than the optimal value then the color is Black
        image_thresh[image_thresh < T] = 0
        image_thresh = cv2.bitwise_not(image_thresh)

        # Perform erotion (Morphological operation)
        kernel = np.ones((2,2),np.uint8)
        image_eroded = cv2.erode(image_thresh,kernel,iterations = 1)

        # We resize the numberplate Since we are using only cars we would resize it to 60*120 pixels
        image_resized = imageTools.resize(image_eroded,width=90, height=30)
        # We use HOG for licence plate detection
        hist, hog_image = hog.describe(image_resized)  
        return hist,hog_image


    def create_features(self, path):    # creates feature-set for all the images in the directory, One image is a row of the feature_matrix
        feature_arr = []
        label_arr = []
        for folder in path:
            #print folder
            for file in glob.glob(folder):
                image_orig = cv2.imread(file)
                hist,hog_image = self.hog_feature(image_orig)
                feature_arr.append(hist)
                # We prepare the class for the photos that are actual licence plate as 1 and non licence plate as 0
                if folder == path[0]:  # path 0
                    label_arr.append(1)
                else:
                    label_arr.append(0)
        return feature_arr, np.array(label_arr)

    def create_training_feature(self, store=None):
        feature_train, labels_train = self.create_features(self.train_path)     # Prepare the training dataset
        self.features = np.array(feature_train, dtype="float64")
        self.labels = np.array(labels_train)
        print (self.features.shape, self.labels.shape)

        if store:
            print ("Plate_feature_dir", self.conf['Plate_feature_dir'])
            print ("Plate_labels_dir", self.conf['Plate_labels_dir'])
            self.store_feature_matrix(self.conf['Plate_feature_dir'], self.conf['Plate_labels_dir'])

    def store_feature_matrix(self, feature_dir, label_dir):
        # We shall load the feature_train set in the disk to let our crossvalidation and classification data use the array and predict there class
        np.savetxt(feature_dir, self.features, delimiter=",")
        np.savetxt(label_dir, self.labels, delimiter=",")

if __name__ == "__main__":
   
    CreateFeatures().create_training_feature(store = True)

    