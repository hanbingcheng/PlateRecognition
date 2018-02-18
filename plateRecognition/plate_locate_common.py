'''
Created on 2018/02/17

@author: hanbing.cheng
'''
import numpy as np
import cv2
import math
from plateRecognition.rotationRect import RotationRect
from common import imageTools
from common import constants
from common.constants import COLOR
from plateRecognition.plate import Plate

class PlateLocateCommon:
    
    def __init__(self, debug):
        self.debug = debug
        self.m_error = constants.DEFAULT_ERROR
        self.m_aspect = constants.DEFAULT_ASPECT
        self.m_verifyMin = constants.DEFAULT_VERIFY_MIN
        self.m_verifyMax = constants.DEFAULT_VERIFY_MAX
        self.m_angle = constants.DEFAULT_ANGLE
        
        
    def deskew(self, srcImage, src_b, inRects, useDeteleArea = False, color = COLOR.UNKNOWN):
        outPlate = []
        for roi_rect in inRects:
            rect = RotationRect(roi_rect)
            roi_ratio = rect.width / rect.height
            roi_angle = rect.angle
            roi_rect_size = rect.size
            if roi_ratio < 1:
                roi_angle = 90 + roi_angle
                rect.width, rect.height = rect.height, rect.width
        
            # m_angle=60
            if (roi_angle - self.m_angle) < 0 and (roi_angle + self.m_angle) > 0 :
                isFormRect, safeBoundRect = imageTools.calcSafeRect(roi_rect, srcImage)  
                if not isFormRect:
                    continue   
                (x,y, w, h) = safeBoundRect
                bound_mat = imageTools.crop(srcImage, x, y, w, h)
                bound_mat_b = imageTools.crop(src_b, x, y, w, h)
        
                if self.debug:
                    imageTools.imshow("bound_mat", bound_mat)
                    imageTools.imshow("bound_mat_b", bound_mat_b)
    
                roi_ref_center = (rect.center_x - safeBoundRect[0], rect.center_y - safeBoundRect[1])
        
                deskew_mat = bound_mat
                if (roi_angle - 5 < 0 and roi_angle + 5 > 0) or 90.0 == roi_angle or -90.0 == roi_angle :
                    deskew_mat = bound_mat
                else:
                    flg, rotated_mat =  self.rotation(bound_mat, roi_rect_size, roi_ref_center, roi_angle)
                    if not flg:
                        continue
            
                    flg, rotated_mat_b = self.rotation(bound_mat_b, roi_rect_size, roi_ref_center, roi_angle)
                    if not flg:
                        continue
            
                    # we need affine for rotatioed image
                    if self.debug:
                        imageTools.imshow("1roated_mat",rotated_mat);
                        imageTools.imshow("rotated_mat_b",rotated_mat_b);
                    flg, roi_slope = self.isdeflection(rotated_mat_b, roi_angle)
                    if flg:
                        deskew_mat = self.affine(rotated_mat, roi_slope)
                    else:
                        deskew_mat = rotated_mat
                
                plate_size = (constants.HEIGHT, constants.WIDTH)
            
                #haitungaga add，affect 25% to full recognition.
                if useDeteleArea:
                    deskew_mat = self.deleteNotArea(deskew_mat, color)
            
                if deskew_mat.shape[1] * 1.0 / deskew_mat.shape[0] > 2.3 and deskew_mat.shape[1] * 1.0 / deskew_mat.shape[0]       < 6:
                    if deskew_mat.shape[1] >= constants.WIDTH or deskew_mat.shape[0] >= constants.HEIGHT:
                        plate_mat = cv2.resize(deskew_mat, plate_size, 0, 0, cv2.INTER_AREA)
                    else:
                        plate_mat = cv2.resize(deskew_mat, plate_size, 0, 0, cv2.INTER_CUBIC)
            
                    plate = Plate()
                    plate.setPlatePos(roi_rect)
                    plate.setPlateMat(plate_mat)
                    if color != COLOR.UNKNOWN:
                        plate.setPlateColor(color)
                    outPlate.append(plate)
        return outPlate
        
    def rotation(self, img, rect_size, center, angle):
        
        in_large = None
        if img.ndim < 3:
            in_large = np.zeros((int(img.shape[0] * 1.5), int(img.shape[1] * 1.5)), np.uint8)
        else:
            in_large = np.zeros((int(img.shape[0] * 1.5), int(img.shape[1] * 1.5), img.shape[2]), np.uint8)
        
        x = in_large.shape[1] / 2 - center[0] if in_large.shape[1] / 2 - center[0] > 0 else 0
        x = int(x)
        y = in_large.shape[0] / 2 - center[1] if in_large.shape[0] / 2 - center[1] > 0 else 0
        y = int(y)
        
        width = img.shape[1] if x + img.shape[1] < in_large.shape[1] else in_large.shape[1] - x
        height = img.shape[0] if y + img.shape[0] < in_large.shape[0] else in_large.shape[0] - y    
    
        if width != img.shape[1] or height != img.shape[0]:
            return (False, None)
    
        imageRoi =  imageTools.crop(in_large, x, y, width, height)
        imageRoi = cv2.addWeighted(imageRoi, 0, img, 1, 0)
    
        #center_diff = (img.shape[1] / 2, img.shape[0] / 2)
        new_center = (in_large.shape[1] // 2, in_large.shape[0] // 2)
    
        rot_mat = cv2.getRotationMatrix2D(new_center, angle, 1)
        
        if self.debug:
            imageTools.imshow("in_copy", in_large)
    
        mat_rotated = cv2.warpAffine(in_large, rot_mat, (in_large.shape[1], in_large.shape[0]),
             cv2.INTER_CUBIC)
    
        if self.debug:
            imageTools.imshow("mat_rotated", mat_rotated)
           
        img_crop = cv2.getRectSubPix(mat_rotated, (rect_size[0], rect_size[1]), new_center)
       
        if self.debug:
            imageTools.imshow("img_crop", img_crop)
    
        return (True, img_crop)

    def affine(self, img, slope):
        if self.debug:
            imageTools.imshow("img", img)
    
        height = img.shape[0]
        width = img.shape[1]
        xiff = abs(slope) * height
        plTri = []
        dstTri = []
        if slope > 0:
            # right, new position is xiff/2
            plTri.append([0, 0])
            plTri.append([width - xiff - 1, 0])
            plTri.append([0 + xiff, height - 1])
           
            dstTri.append([xiff / 2, 0])
            dstTri.append([width - 1 - xiff // 2, 0])
            dstTri.append([xiff / 2, height - 1])
        else:
            # left, new position is -xiff/2
            plTri.append([0 + xiff, 0])
            plTri.append([width - 1, 0])
            plTri.append([0, height - 1])
        
            dstTri.append([xiff / 2, 0])
            dstTri.append([width - 1 - xiff + xiff // 2, 0])
            dstTri.append([xiff / 2, height - 1])
        
        plTri = np.float32(plTri)
        dstTri =  np.float32(dstTri)
        warp_mat = cv2.getAffineTransform(plTri, dstTri)
        
        affine_size = (height, width)
        
        if  img.shape[0] > constants.HEIGHT or img.shape[1] > constants.WIDTH:
            affine_mat = cv2.warpAffine(img, warp_mat, affine_size, cv2.INTER_AREA)
        else:
            affine_mat = cv2.warpAffine(img, warp_mat, affine_size, cv2.INTER_CUBIC);
        
        return affine_mat

    def isdeflection(self, img, angle):
        if self.debug:
            imageTools.imshow("img", img)
      
        nRows = img.shape[0]
        nCols = img.shape[1]
        
        #assert(in.channels() == 1)
        comp_index = [0, 0, 0]
        len  = [0, 0, 0]
        
        comp_index[0] = nRows // 4
        comp_index[1] = nRows // 4 * 2
        comp_index[2] = nRows // 4 * 3
    
        for i in  range(4):
            index = comp_index[i]
            p = img[index]
        
            j = 0
            value = 0
            while 0 == value and j < nCols:
                value = int(p[j])
                j = j + 1
        
            len[i] = j
        
            maxlen = max(len[2], len[0])
            minlen = min(len[2], len[0])
            difflen = abs(len[2] - len[0])
            PI = 3.14159265
            g = math.tan(angle * PI / 180.0)
        
            if maxlen - len[1] > nCols / 32 or len[1] - minlen > nCols / 32:
                slope_can_1 = (len[2] - len[0]) / comp_index[1]
                slope_can_2 = (len[1] - len[0]) / comp_index[0]
                slope_can_3 = (len[2] - len[1]) / comp_index[0]
    
                slope = slope_can_1 if abs(slope_can_1 - g) <= abs(slope_can_2 - g) else slope_can_2
    
                return (True, slope)
            else:
                slope = 0
            
        return (False, slope)
   
    def deleteNotArea(self, inmat, color = COLOR.UNKNOWN):
        input_grey = cv2.cvtColor(inmat, cv2.COLOR_BGR2GRAY)
        
        w = inmat.shape[1]
        h = inmat.shape[0]
        tmpMat =  imageTools.crop(inmat, h * 0.1, w * 0.15, w * 0.7, h * 0.7)
        plateType = COLOR.UNKNOWN
        if COLOR.UNKNOWN == color:
            plateType = imageTools.getPlateType(tmpMat, True)
        else:
            plateType = color
        
        img_threshold = None
        if COLOR.BLUE == plateType:
            img_threshold = input_grey.copy()
            tmp = imageTools.crop(input_grey, w * 0.15, h * 0.15, w * 0.7, h * 0.7)
            if self.debug:
                imageTools.imshow("tmp", tmp)
        
            threadHoldV = imageTools.thresholdOtsu(tmp)
            _, img_threshold = cv2.threshold(input_grey, threadHoldV, 255, cv2.THRESH_BINARY)
            
            if self.debug:
                imageTools.imshow("img_threshold", img_threshold)
        
        elif COLOR.YELLOW == plateType:
            img_threshold = input_grey.copy(   )
            tmp = imageTools.crop(input_grey, w * 0.1, h * 0.1, w * 0.8, h * 0.8)
            threadHoldV = imageTools.thresholdOtsu(tmp)
        
            _,img_threshold = cv2.threshold(input_grey, threadHoldV, 255,cv2.THRESH_BINARY_INV)
            
            if self.debug:
                imageTools.imshow("img_threshold", img_threshold)
    
        else:
            _, img_threshold = cv2.threshold(input_grey, 10, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY);
        
      
        top = 0
        bottom = img_threshold.shape[0] - 1
        flg, img_threshold = imageTools.clearLiuDing(img_threshold)
        
        flg, posLeft, posRight = imageTools.bFindLeftRightBound1(img_threshold)
        if flg:
            inmat = imageTools.crop(inmat,posLeft, top, w - posLeft, bottom - top)
            if self.debug:
                imageTools.imshow("inmat", inmat)
           
        return inmat

    def verifySizes(self, rect):
        error = self.m_error
        # China car plate size: 136 * 36, aspect 4
        aspect = self.m_aspect
    
        # Set a min and max area. All other patchs are discarded
        # minimum area
        min = 80 * 20 * self.m_verifyMin 
        # maximum area
        max = 80 * 20 * self.m_verifyMax  
        # Get only patchs that match to a respect ratio.
        rmin = aspect - aspect * error
        rmax = aspect + aspect * error
        
        if rect.width < 1 or rect.height < 1:
            return False
        
        area = rect.height * rect.width
        r = rect.width / rect.height
        if r < 1:
            r = rect.height / rect.width
    
        if (area < min or area > max) or (r < rmin or r > rmax):
            return False
        else:
            return True
    