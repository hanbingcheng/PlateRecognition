'''
Created on 2018/02/12

@author: hanbing.cheng
'''

class RotationRect:
    def __init__(self, opencv_rect):
        
        # opencv_rect
        # 中心(x,y), (宽,高), 旋转角度）
        self.center = opencv_rect[0]
        self.center_x = self.center[0]
        self.center_y = self.center[1]
        size = opencv_rect[1]
        self.width = int(size[0])
        self.height = int(size[1])
        self.size = (self.width, self.height)
        self.angle = opencv_rect[2]
        

        
