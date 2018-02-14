'''
Created on 2018/02/10

@author: hanbing.cheng
'''
from enum import Enum
import cv2

class COLOR(Enum):
    BLUE = "Blue"
    YELLOW = "Yellow"
    WHITE = "White"
    UNKNOWN = "UnKnown"
    
    def __str__(self):
        return self.value

DEBUG = True

DEFAULT_GAUSSIANBLUR_SIZE = 5
SOBEL_SCALE = 1
SOBEL_DELTA = 0
SOBEL_DDEPTH = cv2.CV_16S
SOBEL_X_WEIGHT = 1
SOBEL_Y_WEIGHT = 0
DEFAULT_MORPH_SIZE_WIDTH = 17
DEFAULT_MORPH_SIZE_HEIGHT = 3

DEFAULT_ERROR = 0.9
DEFAULT_ASPECT = 3.75
    
WIDTH = 136
HEIGHT = 36

DEFAULT_VERIFY_MIN = 1
DEFAULT_VERIFY_MAX = 100 # 24

DEFAULT_ANGLE = 60
