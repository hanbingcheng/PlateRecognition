'''
Created on 2018/02/10

@author: hanbing.cheng
'''
import os
import cv2
from common import configuration
from plateRecognition.plate_color_locate import PlateColorLocate
from plateRecognition.plate_sobel_locate import PlateSobelLocate
from plateRecognition.plate_mser_locate import PlateMserLoacte

class PlateLocate:
    
    def __init__(self):
        self.debug = configuration.get_debug_mode()
        
        
    def plateLocate(self, srcImage):
        all_result_Plates = []
        
        colorLocater = PlateColorLocate(self.debug)
        color_plates = colorLocater.plateLocate(srcImage)
        if len(color_plates) > 0:
            all_result_Plates.extend(color_plates)
        
        sobelLocater = PlateSobelLocate(self.debug)
        sobel_plates = sobelLocater.plateLocate(srcImage)
        if len(sobel_plates) > 0:
            all_result_Plates.extend(sobel_plates)
            
        mserLocater = PlateMserLoacte(self.debug)
        mser_plates = mserLocater.plateLocate(srcImage)
        if len(mser_plates) > 0:
            all_result_Plates.extend(mser_plates)

        return all_result_Plates

if __name__ == "__main__":
    image_dir= '../dataset/Plate_Image/'
    plate = PlateLocate()
    count = 0
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") : #704004834828
            #if count < 3 :
            #    count = count + 1
            #    continue    
                
            print (filename)
            fileFullPath = os.path.join(image_dir,filename)
            img = cv2.imread(fileFullPath)
            #imageTools.imshow("img", img)
            
            locater = PlateLocate()
            all_result_Plates = locater.plateLocate(img)