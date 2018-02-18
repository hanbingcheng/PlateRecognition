'''
Created on 2018/02/16

@author: hanbing.cheng
'''
import os
import numpy as np
import cv2
import math
from common import imageTools
from common import constants
from common import imageTools
from plateRecognition.rotationRect import RotationRect
from plateRecognition.plate import Plate
from plateRecognition.plate_locate_common import PlateLocateCommon
class PlateSobelLocate:
    
    def __init__(self, debug):
        self.debug = debug
        self.common = PlateLocateCommon(debug)
        self.m_GaussianBlurSize = constants.DEFAULT_GAUSSIANBLUR_SIZE
        self.m_MorphSizeWidth = constants.DEFAULT_MORPH_SIZE_WIDTH
        self.m_MorphSizeHeight = constants.DEFAULT_MORPH_SIZE_HEIGHT
        
    def sobelOper(self, srcImage, blurSize, morphW, morphH):          
        mat_blur = cv2.GaussianBlur(srcImage, (blurSize, blurSize), 0, 0, cv2.BORDER_DEFAULT)
        if self.debug:
            imageTools.imshow("mat_blur" , mat_blur)
        
        if mat_blur.ndim == 3:
            mat_gray = cv2.cvtColor(mat_blur, cv2.COLOR_RGB2GRAY);
        else:
            mat_gray = mat_blur
        if self.debug:
            imageTools.imshow("mat_gray" , mat_gray)
        
        scale = constants.SOBEL_SCALE
        delta = constants.SOBEL_DELTA
        ddepth = constants.SOBEL_DDEPTH

        #grad_x = cv2.Sobel(mat_gray, ddepth, 1, 0, 3, scale, delta, cv2.BORDER_DEFAULT)
        grad_x = cv2.Sobel(mat_gray, ddepth, 1, 0, 3 )
        # 将RealSense提取的深度图片进行显示时，由于是16位图片，想将图片转化成为8位图形进行显示
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        if self.debug:
            imageTools.imshow("abs_grad_x" , abs_grad_x)
        grad = cv2.addWeighted(abs_grad_x, constants.SOBEL_X_WEIGHT, 0, 0, 0)
        _, mat_threshold = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
        if self.debug:
            imageTools.imshow("mat_threshold" , mat_threshold)
        
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (morphW, morphH))
        mat_threshold = cv2.morphologyEx(mat_threshold, cv2.MORPH_CLOSE, element)
        if self.debug:
            imageTools.imshow("mat_threshold after close" , mat_threshold)
        
        return mat_threshold
    
    def sobelFrtSearch(self, srcImage) :
        src_threshold = self.sobelOper(srcImage, self.m_GaussianBlurSize, 
                            self.m_MorphSizeWidth, self.m_MorphSizeHeight)
        if self.debug:
            imageTools.imshow("src_threshold" , src_threshold)
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
                isFormRect, safeBoundRect = imageTools.calcSafeRect(rotationrect, srcImage)  
                if not isFormRect:
                    continue   
                if self.debug:
                    cv2.drawContours(mask, [itc], -1, (255, 255, 255), -1)
                    targetContour = cv2.bitwise_and(testImage, testImage, mask=mask)
                    imageTools.imshow("target contour", targetContour)

                outRects.append(safeBoundRect)
                              
        return outRects
    
    def sobelSecSearchPart(self, bound, x, y):
        outRects = [  ]
        bound_threshold = self.sobelOper(bound, 3, 6, 2)
        tempBoundThread = imageTools.clearLiuDingOnly(bound_threshold)
 
        flg, posLeft, posRight = imageTools.bFindLeftRightBound(tempBoundThread)
        if flg:
            # find left and right bounds to repair
            if posRight != 0 and posLeft != 0 and posLeft < posRight:
                row = int(bound_threshold.shape[0] * 0.5)
                start = posLeft + int(bound_threshold.shape[0] * 0.1)
                bound_threshold[row, start: posRight - 4] = 255
                
            if self.debug:
                imageTools.imshow("bound_threshold", bound_threshold)
        
            # remove the left and right boundaries
            bound_threshold[:, posLeft] = 0
            bound_threshold[:, posRight] = 0
            
            if self.debug:
                imageTools.imshow("bound_threshold", bound_threshold)
                
            _,contours,_ = cv2.findContours(bound_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for itc in contours:
                # 中心(x,y), (宽,高), 旋转 角度
                rotationrect = cv2.minAreaRect(itc)
                rect = RotationRect(rotationrect)
                
                if self.common.verifySizes(rect):
                    refcenter = (rect.center_x + x,  rect.center_y + y)
                    refroi = (refcenter, rect.size, rect.angle)
                    outRects.append(refroi)
           
        return outRects
    
    def sobelSecSearch(self,  bound, x, y) :
        bound_threshold = self.sobelOper(bound, 3, 10, 3)
        if self.debug:
            imageTools.imshow("bound_threshold", bound_threshold)
            
        _,contours,_ = cv2.findContours(bound_threshold,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)  # all pixels of each contours
        outRects = []
        for itc in contours:
            # 中心(x,y), (宽,高), 旋转 角度
            rotationrect = cv2.minAreaRect(itc)
            rect = RotationRect(rotationrect)
            
            if self.common.verifySizes(rect):
                refcenter = (rect.center_x + x,  rect.center_y + y)
                refroi = (refcenter, rect.size, rect.angle)
                outRects.append(refroi)
    
        return outRects 
    
    def plateLocate(self, srcImage):
        bound_rects = self.sobelFrtSearch(srcImage)
        bound_rects_part = []
        # enlarge area 
        for bound_rect in bound_rects:
            x, y, width, height = bound_rect
            rRatio = width / height
            
            if 1.0 < rRatio < 3.0 and  height < 120 :
                itemRect_x = x - height * (4 - rRatio)
                if itemRect_x < 0 :
                    itemRect_x = 0
                    
                itemRect_width = width + height * 2 * (4 - rRatio)
                if itemRect_width + itemRect_x >= srcImage.shape[1] :
                    itemRect_width = srcImage.shape[1] - itemRect_x
            
                    itemRect_y = y - height * 0.08
                    itemRect_height = height * 1.16
                    itemRect = (itemRect_x, itemRect_y, itemRect_width, itemRect_height)
                    bound_rects_part.append(itemRect)
        
        # second processing to split one
        rects_sobel_all = []
        for bound_rect in bound_rects_part:
            bound_rect_x, bound_rect_y, bound_rect_width, bound_rect_height = bound_rect
            x = bound_rect_x if bound_rect_x > 0 else 0
            y = bound_rect_y if bound_rect_y > 0 else 0
            width = bound_rect_width if x + bound_rect_width < srcImage.shape[1] else srcImage.shape[1] - x
            height = bound_rect_height if y + bound_rect_height       < srcImage.shape[0] else srcImage.shape[0] - y
            
            bound_mat = imageTools.crop(srcImage, x, y, width, height)
            rects_sobel = self.sobelSecSearchPart(bound_mat, x, y)
            if len(rects_sobel) > 0:
                rects_sobel_all.extend(rects_sobel)
             
        candPlates = []      
        for bound_rect in bound_rects:
            bound_rect_x, bound_rect_y, bound_rect_width, bound_rect_height = bound_rect
            x = bound_rect_x if bound_rect_x > 0 else 0
            y = bound_rect_y if bound_rect_y > 0 else 0
            width = bound_rect_width if x + bound_rect_width < srcImage.shape[1] else srcImage.shape[1] - x
            height = bound_rect_height if y + bound_rect_height < srcImage.shape[0] else srcImage.shape[0] - y
            bound_mat = imageTools.crop(srcImage, x, y, width, height)
            rects_sobel = self.sobelSecSearch(bound_mat, x, y)
            if len(rects_sobel) > 0:
                rects_sobel_all.extend(rects_sobel)
                
                
        src_b = self.sobelOper(srcImage, 3, 10, 3)
        plates = self.common.deskew(srcImage, src_b, rects_sobel_all)
        if len(plates) > 0:
            candPlates.extend(plates)
    
        return  candPlates
    
if __name__ == "__main__":
    image_dir= '../dataset/Plate_Image/'
    locater = PlateSobelLocate(True)
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") : #704004834828
            print (filename)
            fileFullPath = os.path.join(image_dir,filename)
            img = cv2.imread(fileFullPath)
            imageTools.imshow("img", img)
            locater.plateLocate(img)   