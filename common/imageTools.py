'''
Created on 2018/01/27

@author: hanbing.cheng
'''
import numpy as np
import cv2
import math
from common.constants import COLOR

def translate(image, x, y):
    M=np.float32([[1,0,x],[0,1,y]])
    # M is defined as the floating point array because cv2 expects the matrix to be in floting point array
    # The first array [1,0,x] indicates the  number of pixels to shift right, A negative x would shift x pixels left
    # The second array [0,1,y] indicates the number of pixels to shift down, A negative y would shift the image y pixels up
    shifted=cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    # The first argument to the warpAffine function is the image
    # The second argument to the warpAffine function is the array by which the image has to shift
    # The third argument is the dimension in (height and width), by default the image is stored in (weidth * height) but opencv takes dimension in (height * weidth) format
    return shifted

def rotate(image, angle, center = None, scale=1.0):
    # The arguments are :
    '''
    1. The image
    2. the angle by which you want to rotate
    3. The center from which you want to rotate, The default is None, when None then we define the center
    4. scale=1.0 by default, It states that
    '''
    (h,w)= image.shape[:2]

    # When the center is None then we define the center from which the rotation is to be done
    if center is None:
        center=(w/2, h/2)

    M=cv2.getRotationMatrix2D(center, angle, scale)
    rotated= cv2.warpAffine(image, M, (w,h))

    return rotated


def resize(image, width= None, height = None, inter=cv2.INTER_AREA):
    dim=None
    (h,w)= image.shape[:2]  # numpy array stores images in (height, width) array, but cv2 uses images in order (width, height) order

    if width is None and height is None:  # when no resizing occur
        return image

    if width is None:       # when resized height is passed and width is not then we calculate the aspect ratio of the weidth
        r= height / float(h)
        dim=(int(w * r), height)      # height is the resized hieght
    elif height is None:    # When resized width is passed and hieght is not then we calculate the aspect ratio for the height
        r= width / float(w)
        dim=(width , int(h * r))
    else:                   # when both width and height ratio are provided
        dim=(width, height)

    resized= cv2.resize(image, dim, interpolation=inter)
    # the third argument hold an algorithm in cv2 defined to resize the image
    # we can also use other algorithm like cv2.INTER_LINEAR, cv2.INTER_CUBIC, and cv2.INTER_NEAREST.

    return resized

def imwrite(filePath, img):
    cv2.imwrite(filePath, img)

def imshow(title, img):
    cv2.imshow(title, img)
    cv2.waitKey()
    cv2.destroyWindow(title)
    
def colorMatch(srcImage, color, adaptive_minsv):

    # if use adaptive_minsv
    # min value of s and v is adaptive to h
    max_sv = 255
    minref_sv = 64
    minabs_sv = 95
    # H range of blue 
    min_blue = 100
    max_blue = 140

    # H range of yellow
    min_yellow = 15
    max_yellow = 40

    #H range of white
    min_white = 0
    max_white = 30

    #convert to HSV space
    src_hsv = cv2.cvtColor(srcImage, cv2.COLOR_BGR2HSV)

    hsvSplit = cv2.split(src_hsv)
    hsvSplit[2] = cv2.equalizeHist(hsvSplit[2])
    src_hsv = cv2.merge(hsvSplit)

    # match to find the color
    min_h = 0
    max_h = 0

    if color == COLOR.BLUE:
        min_h = min_blue
        max_h = max_blue
    elif color == COLOR.YELLOW:
        min_h = min_yellow
        max_h = max_yellow
    elif color == COLOR.WHITE:
        min_h = min_white
        max_h = max_white

    diff_h = (max_h - min_h) / 2
    avg_h = min_h + diff_h

    #channels = 1 if len(src_hsv.shape) == 1 else len(src_hsv.shape)
    nRows = src_hsv.shape[0]

    # consider multi channel image
    nCols = src_hsv.shape[1] #* channels
    #if src_hsv.isContinuous():
    #    nCols = nCols * nRows
    #    nRows = 1

    s_all = 0
    v_all = 0
    count = 0

    for i in range(0, nRows-1):
        p = src_hsv[i]
        for j in range(0, nCols-1):
            H = p[j][0]      # 0-180
            S = p[j][1] # 0-255
            V = p[j][2]  # 0-255
            
            s_all += S
            v_all += V
            count= count + 1

            colorMatched = False

            if H > min_h and H < max_h:
                Hdiff = 0
                if H > avg_h:
                    Hdiff = H - avg_h
                else:
                    Hdiff = avg_h - H
    
                Hdiff_p = Hdiff / diff_h
    
                min_sv = 0
                if adaptive_minsv:
                    min_sv = minref_sv - minref_sv / 2 * (1 - Hdiff_p)  
                        # inref_sv - minref_sv / 2 * (1 - Hdiff_p)
                else:
                    min_sv = minabs_sv
    
                if (S > min_sv and S < max_sv) and (V > min_sv and V < max_sv):
                    colorMatched = True
            
            if colorMatched:
                src_hsv[i][j][0] = 0
                src_hsv[i][j][1] = 0
                src_hsv[i][j][2] = 255
            else:
                src_hsv[i][j][0] = 0
                src_hsv[i][j][1] = 0
                src_hsv[i][j][2] = 0
        
        # get the final binary
        hsvSplit_done = cv2.split(src_hsv)
        src_grey = hsvSplit_done[2]

        return src_grey

def getBoundRect(box):
    
    p1,p2,p3,p4 = box[0], box[1], box[2], box[3]
    p1_x, p1_y = p1[0], p1[1]
    p2_x, p2_y = p2[0], p2[1]
    p3_x, p3_y = p3[0], p3[1]
    p4_x, p4_y = p4[0], p4[1]
    
    x = min(p1_x, p2_x, p3_x, p4_x)
    y = min(p1_y, p2_y, p3_y, p4_y)
    width = abs(p1_x - p3_x)
    height = abs(p1_y - p3_y)
    
    return x, y, width, height
    
#  calc safe Rect
#  if not exit, return false
def calcSafeRect(roi_rect, srcImage):
    box = cv2.boxPoints(roi_rect)
    #roi_rect.boundingRect()
    # while the x, y in cv2 is different in numpy
    boudRect_x, boudRect_y, boudRect_width, boudRect_height = getBoundRect(box)
    
    x = boudRect_x if boudRect_x > 0 else 0
    y = boudRect_y if boudRect_y > 0 else 0

    br_x = (boudRect_x + boudRect_width - 1) if boudRect_x + boudRect_width < srcImage.shape[1] else srcImage.shape[1] - 1
    br_y = (boudRect_y + boudRect_height - 1) if boudRect_y + boudRect_height < srcImage.shape[0] else srcImage.shape[0] - 1
    
    roi_width = br_x - x
    roi_height = br_y - y

    if roi_width <= 0 or roi_height <= 0:
        return (False, None)

    #  a new rect not out the range of mat
    x = int(math.ceil(x))
    y = int(math.ceil(y))
    w = int(math.ceil(roi_width))
    h = int(math.ceil(roi_height))
    safeBoundRect = (x, y, w, h)

    return (True, safeBoundRect)


def calcSafeRectROI(roi_rect, width, height):
    boudRect = roi_rect.boundingRect();

    x = boudRect.x if boudRect.x > 0 else 0
    y = boudRect.y if boudRect.y > 0 else 0

    br_x = (boudRect.x + boudRect.width - 1) if (boudRect.x + boudRect.width) < width else (width - 1)
    br_y = (boudRect.y + boudRect.height - 1) if boudRect.y + boudRect.height < height else (height - 1)

    roi_width = br_x - x
    roi_height = br_y - y

    if roi_width <= 0 or roi_height <= 0:
        return (False, None)

    
    #  a new rect not out the range of mat
    x = int(math.ceil(x))
    y = int(math.ceil(y))
    w = int(math.ceil(roi_width))
    h = int(math.ceil(roi_height))
    safeBoundRect = (x, y, w, h)

    return (True, safeBoundRect)


def crop(image, x, y, w, h):
    # change x and y since the x,y in opencv is different in nupmy
    x1 = int(math.ceil(y))
    y1 = int(math.ceil(x))
    x2 = x1 + int(math.ceil(h))
    y2 = y1 + int(math.ceil(w))
        
    if x2 > image.shape[0]:
        x2 = image.shape[0]
    
    if y2 > image.shape[1]:
        y2 = image.shape[1]
    
    return image[x1 : x2, y1 : y2]

def plateColorJudge(src, color, adaptive_minsv):

    thresh = 0.45
    src_gray = colorMatch(src, color, adaptive_minsv)

    percent = cv2.countNonZero(src_gray) / (src_gray.shape[0] * src_gray.shape[1])
    
    #print ("percent:", percent)

    if percent > thresh:
        return True
    else:
        return False
  
def getPlateType(src, adaptive_minsv):
    max_percent = 0
    max_color = COLOR.UNKNOWN

    if plateColorJudge(src, COLOR.BLUE, adaptive_minsv):
        print ("BLUE")
        return COLOR.BLUE;
    elif plateColorJudge(src, COLOR.YELLOW, adaptive_minsv):
        print ("YELLOW")
        return COLOR.YELLOW;
    elif plateColorJudge(src, COLOR.WHITE, adaptive_minsv):
        print ("WHITE")
        return COLOR.WHITE
    else:
        print ("OTHER")
        return COLOR.BLUE

def clearLiuDingOnly(img):
    x = 7
    jump = np.zeros((1, img.shape[0]), np.int)
    for i in range(img.shape[0]):
        jumpCount = 0
        whiteCount = 0
        for j in range(img.shape[1] - 1):
            if img[i, j] != img[i, j + 1]:
                jumpCount = jumpCount + 1
    
            if img[i, j] == 255:
                whiteCount = whiteCount + 1
                
        jump[0, i] = jumpCount

    for i in range(img.shape[0]):
        if jump[0, i] <= x:
            for j in range(img.shape[1]):
                img[i, j] = 0
    return img

def clearLiuDing(img):
    fJump = []
    whiteCount = 0
    x = 7
    #imshow("img", img)
    jump = np.zeros((1, img.shape[0]), np.float32)
    for i in range(img.shape[0]):
        jumpCount = 0

        for j in range(img.shape[1] - 1):
            if img[i, j] != img[i, j + 1]:
                jumpCount = jumpCount + 1
    
            if img[i, j] == 255:
                whiteCount = whiteCount + 1

        jump[0, i] = jumpCount

    iCount = 0
    for i in range(img.shape[0]):
        fJump.append(jump[0, i]);
        if 16 <= jump[0, i] <= 45:
            #jump condition
            iCount = iCount + 1

    # if not is not plate
    if iCount * 1.0 / img.shape[0] <= 0.40 : 
        return (False, img)

    if whiteCount * 1.0 / img.size < 0.15 or whiteCount * 1.0 / img.size > 0.50:
        return (False, img)

    for i in range(img.shape[0]):
        if jump[0, i] <= x:
            for j in range(img.shape[1]):
                img[i, j] = 0

    return (True, img)

def thresholdOtsu(mat):
    # histogram
    histogram = cv2.calcHist([mat],[0],None,[256],[0,256])

    # normalize histogram
    size = mat.size;
    for i in range(256): 
        histogram[i] = histogram[i] / size;

    # average pixel value
    avgValue = 0
    for i in range(256): 
        avgValue = avgValue + i * histogram[i]

    thresholdV = 0
    maxVariance = 0
    w = 0 
    u = 0
    for i in range(256): 
        w = w + histogram[i]
        u = u + i * histogram[i]
        
        t = avgValue * w - u
        variance = t * t / (w * (1 - w))
        if variance > maxVariance:
            maxVariance = variance
            thresholdV = i

    return thresholdV

def histeq(img):
    out = np.zeros(img.size)
    if img.channels() == 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsvSplit = cv2.split(hsv)
        hsvSplit[2] = cv2.equalizeHist(hsvSplit[2])
        hsv = cv2.merge(hsvSplit)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    elif img.channels() == 1:
        out = cv2.equalizeHist(img)
        
    return out


def bFindLeftRightBound1(bound_threshold):
    posLeft = 0
    posRight = 0
    span = round(bound_threshold.shape[0] * 0.2)

    for i in range(bound_threshold.shape[1] - span - 1):
        whiteCount = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i + span, 1):
                if bound_threshold[k, l] == 255:
                    whiteCount = whiteCount + 1
              
        if whiteCount * 1.0 / (span * bound_threshold.shape[0]) > 0.15:
            posLeft = i
            break

    span = int(bound_threshold.shape[0] * 0.2)

    for i in range(bound_threshold.shape[1] - 1, span, -2):
        whiteCount = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i,i - span, -1):
                if bound_threshold[k, l] == 255:
                    whiteCount = whiteCount + 1

        if whiteCount * 1.0 / (span * bound_threshold.shape[0]) > 0.06:
            posRight = i;
            if posRight + 5 < bound_threshold.shape[1]:
                posRight = posRight + 5;
            else:
                posRight = bound_threshold.shape[1] - 1
    
            break
    if posLeft < posRight:
        return (True, posLeft, posRight)
  
    return (False, posLeft, posRight)


def bFindLeftRightBound(bound_threshold):
    posLeft = 0
    posRight = 0
    span = round(bound_threshold.shape[0] * 0.2)

    for i in range(0, bound_threshold.shape[1] - span - 1, 2):
        whiteCount = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i, i + span, 1):
                if bound_threshold[k, l] == 255:
                    whiteCount = whiteCount + 1
              
        if whiteCount  / (span * bound_threshold.shape[0]) > 0.36:
            posLeft = i
            break
        
    span = int(bound_threshold.shape[0] * 0.2)

    for i in range(bound_threshold.shape[1] - 1, span, -2):
        whiteCount = 0
        for k in range(bound_threshold.shape[0]):
            for l in range(i,i - span, -1):
                if bound_threshold[k, l] == 255:
                    whiteCount = whiteCount + 1

        if whiteCount  / (span * bound_threshold.shape[0]) > 0.26:
            posRight = i;
            break
        
    if posLeft < posRight:
        return (True, posLeft, posRight)
  
    return (False, posLeft, posRight)