'''
Created on 2018/02/11

@author: hanbing.cheng
'''

from common.constants import COLOR

class Plate:
    
    def __init__(self, other = None):
        if other == None:
            self.__m_plateMat = None
            self.__m_chineseMat = None
            self.__m_chineseKey = None
            self.__m_score = -1
            self.__m_platePos = None
            self.__m_plateStr = ""
            self.__m_locateType = None
            self.__m_plateColor = COLOR.UNKNOWN
            self.__m_line = None
            self.__m_leftPoint = None
            self.__m_rightPoint = None
            self.__m_mergeCharRect = None
            self.__m_maxCharRect = None
            self.__m_scale = None
            self.__m_distVec = []
            self.__m_mserCharVec = []
            self.__m_reutCharVec = []
            self.__m_ostuLevel = None
            
        else:
            self.__m_plateMat = other.m_plateMat
            self.__m_chineseMat = other.m_chineseMat
            self.__m_chineseKey = other.m_chineseKey
            self.__m_score = other.m_score
            self.__m_platePos = other.m_platePos
            self.__m_plateStr = other.m_plateStr
            self.__m_locateType = other.m_locateType
            self.__m_plateColor = other.m_plateColor
            self.__m_line = other.m_line
            self.__m_leftPoint = other.m_leftPoint
            self.__m_rightPoint = other.m_rightPoint
            self.__m_mergeCharRect = other.m_mergeCharRect
            self.__m_maxCharRect = other.m_maxCharRect
            self.__m_scale = other.m_scale
            self.__m_distVec = other.m_distVec
            self.__m_mserCharVec = other.m_mserCharVec
            self.__m_reutCharVec = other.m_reutCharVec
            self.__m_ostuLevel = other.m_ostuLevel

    def __eq__(self, other):
        if self == other:
            return True
        else:
            return False
        
    def setPlateMat(self, param):
        self.__m_plateMat = param
        
    def getPlateMat(self):
        return self.__m_plateMat

    def setChineseMat(self, param):
        self.__m_chineseMat = param
        
    def getChineseMat(self):
        return self.__m_chineseMat

    def setChineseKey(self, param):
        self.__m_chineseKey = param
        
    def  getChineseKey(self):
        return self.__m_chineseKey

    def setPlatePos(self, param):
        self.__m_platePos = param
        
    def  getPlatePos(self):
        return self.__m_platePos

    def setPlateStr(self, param):
        self.__m_plateStr = param
         
    def  getPlateStr(self):
        return self.__m_plateStr

    def setPlateLocateType(self, param):
        self.__m_locateType = param
        
    def  getPlateLocateType(self):
        return self.__m_locateType

    def setPlateColor(self, param):
        self.__m_plateColor = param
        
    def  getPlateColor(self):
        return self.__m_plateColor

    def setPlateScale(self, param):
        self.__m_scale = param
        
    def  getPlateScale(self):
        return self.__m_scale

    def setPlateScore(self, param):
        self.__m_score = param
        
    def  getPlateScore(self):
        return self.__m_score

    def setPlateLine(self, param):
        self.__m_line = param
        
    def  getPlateLine(self):
        return self.__m_line

    def setPlateLeftPoint(self, param): 
        self.__m_leftPoint = param
        
    def  getPlateLeftPoint(self):
        return self.__m_leftPoint

    def setPlateRightPoint(self, param):
        self.__m_rightPoint = param
        
    def  getPlateRightPoint(self):
        return self.__m_rightPoint

    def setPlateMergeCharRect(self, param):
        self.__m_mergeCharRect = param
        
    def  getPlateMergeCharRect(self):
        return self.__m_mergeCharRect

    def setPlateMaxCharRect(self, param):
        self.__m_maxCharRect = param
        
    def  getPlateMaxCharRect(self):
        return self.__m_maxCharRect

    def setPlatDistVec(self, param):
        self.__m_distVec = param
        
    def  getPlateDistVec(self):
        return self.__m_distVec

    def setOstuLevel(self, param):
        self.__m_ostuLevel = param
    
    def  getOstuLevel(self):
        return self.__m_ostuLevel

    def setMserCharacter(self, param):
        self.__m_mserCharVec = param
    
    def addMserCharacter(self, param):
        self.__m_mserCharVec.append(param)
        
    def  getCopyOfMserCharacters(self):
        return self.__m_mserCharVec

    def setReutCharacter(self, param):
        self.__m_reutCharVec = param
        
    def addReutCharacter(self, param):
        self.__m_reutCharVec.append(param)
        
    def  getCopyOfReutCharacters(self):
        return self.__m_reutCharVec

    def __lt__(self, plate):
        return self.__m_score < plate.m_score   