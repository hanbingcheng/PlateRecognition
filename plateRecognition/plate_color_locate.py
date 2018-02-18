'''
Created on 2018/02/16

@author: hanbing.cheng
'''
import os
import numpy as np
import cv2
from common import imageTools
from common import constants
from common.constants import COLOR
from plateRecognition.rotationRect import RotationRect
from plateRecognition.plate_locate_common import PlateLocateCommon

class PlateColorLocate:
    
    def __init__(self, debug):
        self.debug = debug
        self.common = PlateLocateCommon(debug)
        
    def colorSearch(self, srcImage, color):
        # width is important to the final results;
        color_morph_width = 10
        color_morph_height = 2
    
        match_grey = imageTools.colorMatch(srcImage, color, False)
        if self.debug:
            imageTools.imshow("match_grey", match_grey)
        
        _,src_threshold = cv2.threshold(match_grey, 0, 255,
            cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        if self.debug:
            imageTools.imshow("src_threshold", src_threshold)
        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (color_morph_width,                                                    
                                                             color_morph_height))
        src_threshold = cv2.morphologyEx(src_threshold, cv2.MORPH_CLOSE, element)
    
        if self.debug:
            imageTools.imshow("threshold", src_threshold);
    
        out = src_threshold.copy()
    
        _,contours,_ = cv2.findContours(src_threshold,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)  # all pixels of each contours
        outRects = []
        testImage = srcImage.copy()
        mask = np.zeros((srcImage.shape[0], srcImage.shape[1], 1), dtype=np.uint8)
        for itc in contours:
            # 中心(x,y), (宽,高), 旋转 角度
            rotationrect = cv2.minAreaRect(itc)
            rect = RotationRect(rotationrect)
            
            if self.common.verifySizes(rect):
                outRects.append(rotationrect)
                if self.debug:
                    cv2.drawContours(mask, [itc], -1, (255, 255, 255), -1)
                    targetContour = cv2.bitwise_and(testImage, testImage, mask=mask)
                    imageTools.imshow("target contour", targetContour)
              
        return (out, outRects)
    

    def plateLocate(self, srcImage):
        candPlates = []
        srcImage_clone = srcImage.copy()
        
        src_b_blue,rects_color_blue = self.colorSearch(srcImage, COLOR.BLUE)
        plates_blue = self.common.deskew(srcImage, src_b_blue, rects_color_blue, True, COLOR.BLUE)
        candPlates.extend(plates_blue)
        
        src_b_yellow, rects_color_yellow = self.colorSearch(srcImage_clone, COLOR.YELLOW)
        plates_yellow = self.common.deskew(srcImage_clone, src_b_yellow, rects_color_yellow, True, COLOR.YELLOW)
        #if self.debug:
        print ("num of plates form blue", len(plates_blue))
        print ("num of plates from yellow", len(plates_yellow))
        
        candPlates.extend(plates_yellow)
        
        return candPlates 
    
if __name__ == "__main__":
    image_dir= '../dataset/Plate_Image/'
    locater = PlateColorLocate(True )
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") : #704004834828
            print (filename)
            fileFullPath = os.path.join(image_dir,filename)
            img = cv2.imread(fileFullPath)
            imageTools.imshow("img", img)
            locater.plateLocate(img) 