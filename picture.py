# -*- coding: utf-8 -*-
import cv2
import numpy as np
import math
import sys
from numpy.linalg import norm
import globallist

def calibrate(img):
    ''' 摄像头矫正
    input: 摄像头扭曲图像
    return: 单通道矫正图像
    '''
    coe = np.load("coeff.npz")
    mapx = coe["arr_2"]
    mapy = coe["arr_3"]
    dst = cv2.remap(img,mapx,mapy,cv2.INTER_CUBIC)
    return dst

def getLine(img,w):
    ''' 找到五线谱中的线,w是纵向区分两条线的最短距离,这是摄像头自动检测五线
    input: 矫正的图像
    return: 五线谱中线的位置,r和theta值
    '''
    figure = []
    blurimg = cv2.GaussianBlur(img,(3,3),0) #高斯平滑处理原图像降噪   
    canny = cv2.Canny(blurimg, 50, 150)
    lines = cv2.HoughLines(canny,1,np.pi/180,100)
    lines = np.squeeze(lines)
    if lines.shape != ():
        for i in range(0,len(lines)):
            figure.append([lines[i][0],lines[i][1]])
            count = len(figure)
            if count == 2:
                if math.fabs(figure[1][0] - figure[0][0]) < w:
                    del figure[1]
            j = 0
            if count > 2:
                while(j <= count-2):
                    if math.fabs(figure[count-1][0] - figure[j][0]) < w:
                        del figure[count-1]
                        break
                    else:
                        j = j + 1
        figure.sort()
        if len(figure) != 5: #设默认值，检测不到五条线
            print("use the default line")
            figure = []
            figure = [[100],[140],[180],[220],[260]]
            return figure
    else:
        figure = [[100],[140],[180],[220],[260]]
        return figure


def manugetLine(img):
    '''手动标记五线谱中的五条线,然后保存在line.txt文件中
        input:一张矫正的五线谱图片
        return:五线谱中线的位置
    '''
    figure = []
    def getExtremity(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img,(x,y),3,(255,255,0),-1)
            iy = int(y)
            figure.append([iy])
    cv2.namedWindow('manugetLine')
    cv2.setMouseCallback('manugetLine',getExtremity)
    while(1):
        cv2.imshow("manugetLine",img)
        if cv2.waitKey(1) == ord('q'):
            break
    figure.sort()
    if len(figure) == 5:
        fp = open("line.txt",'w')
        for i in range(0,5):
            strl = []
            strl = str(figure[i][0]) + "\n"
            fp.write(strl)
        fp.close()
    return figure

def lined():
    ''' 使用保存在line.txt中的五线的信息
    return : 五线谱中线的位置
    '''
    figure = []
    fp = open("line.txt",'r')
    lines = fp.readlines()
    fp.close()
    for line in lines:
        figure.append([int(line)])
    return figure


def hog(img):
    '''提取图像的方向梯度直方图，用作SVM训练的特征
    input:60*40的图像
    return:特征值，16*4*5*3=960个特征。
    '''
    SamplesList = []
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16
    bin = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = []
    mag_cells = []
    Sample = []
    Samples = []
    for y in range(0,60,10):
        for x in range(0,40,10):
            bin_cells.append(bin[y:y+10,x:x+10])
            mag_cells.append(mag[y:y+10,x:x+10])
    for j in range(0,5,1):
        for i in range(0,3):
            bin_block = bin_cells[j*4+i],bin_cells[j*4+i+1],bin_cells[j*4+i+4],bin_cells[j*4+i+5]
            mag_block = mag_cells[j*4+i],mag_cells[j*4+i+1],mag_cells[j*4+i+4],mag_cells[j*4+i+5]
            hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_block, mag_block)]
            hist = np.hstack(hists)
            hist = hist.astype("float64")
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps 
            Samples.append(hist)
    Sample = np.hstack(Samples)
    SamplesList.append(np.array(Sample))
    return np.float32(SamplesList)

def getY(coor):
    '''得到音符的纵向坐标信息，获取音标信息。do,re,mi,fa,sol,la,si。
        input:音符的位置的纵坐标值
        return:y
    '''
    i = 0
    yList = []
    count = len(coor)
    while(i < count):
        y = coor[i][1]
        for j in range(0,4):
            if y > j and y <j + 120:
                yList.append(j)
        i = i + 1
    return yList




class Picture():
    '''音符图片发现，处理，输出的类,用的是opencv-python (3.3.0.10),python 3.6.3 '''
    def __init__(self,img):
        self.img = img  #原始图片
        self.binaryimg = None #这是二值化或者自适应二值化得到的图
        self.erodimg = None #腐蚀得到的图像
        self.contours = None #音符的边框信息
        self.contimg = None #把轮廓画在这张图上
        self.rectimg = None #画最小矩形的图片
        self.sizimg = None #标准大小的音符图片
        self.coor = None #音符左下角的坐标
        self.model = cv2.ml.SVM_load("bestsvm.dat")  #这是SVM向量机训练的结果
        self.coe = np.load("coeff.npz") #这是摄像头矫正参数
        self.mapx = self.coe["arr_2"]
        self.mapy = self.coe["arr_3"]
        self.figure = None #五线谱中线的信息,r和theta
        self.info = None #音符的位置和形状
        self.beginFlag = False #开始信号
    # def calibrate(self):
    #     ''' 摄像头矫正
    #     input: 摄像头扭曲图像
    #     return: 三通道矫正图像
    #     '''
    #     if self.img is None:
    #         print("img is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
    #         exit()
    #     else:
    #         self.cimg = cv2.remap(self.img,self.mapx,self.mapy,cv2.INTER_CUBIC)
    #         return self.cimg
    def Update(self,img):
        '''获取最新图片,这个图片已经经过矫正了'''
        if img is None:
            print("img is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            self.img = calibrate(img)
            return self.img
    def ConvBinary(self,adaptflag=0):
        '''将图像转化为二值化图片'''
        if self.img is None:
            print("img is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            grayimg = cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY)
            blurimg = cv2.medianBlur(grayimg,5)
            if adaptflag == 0:
                self.binaryimg = cv2.threshold(blurimg,threshold,255,cv2.THRESH_BINARY)
            else:
                self.binaryimg = cv2.adaptiveThreshold(blurimg, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
            return self.binaryimg
    def Erode(self,shape=cv2.MORPH_RECT,ksize=(9, 9)):
        '''腐蚀处理'''
        if self.binaryimg is None:
            print("img is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            kernel = cv2.getStructuringElement(shape,ksize) 
            self.erodeimg = cv2.erode(self.binaryimg,kernel)
            return self.erodeimg
    def Contours(self,mode=cv2.RETR_TREE,method=cv2.CHAIN_APPROX_SIMPLE,contCoe=(0.8,0.005)):
        ''' 获取音符轮廓'''
        if self.erodeimg is None:
            print("erodeimg is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            contImg,self.contours,hierarchy = cv2.findContours(self.erodeimg,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            i = 0
            count = len(self.contours)
            while(i<count):
                area = math.fabs(cv2.contourArea(self.contours[i]))
                if area >= self.img.shape[0]*self.img.shape[1]*contCoe[0] or area <= self.img.shape[0]*self.img.shape[1]*contCoe[1]:
                    del self.contours[i]
                    count = count - 1 
                    i = i - 1
                i = i + 1
            self.contimg = self.img.copy()
            self.contimg[:] = 255
            cv2.drawContours(self.contimg,self.contours,-1,(125,125,125),1)
            return self.contours,self.contimg
    def GetRect(self):
        '''得到每个轮廓最小矩形'''
        if self.contimg is None or self.contours is None:
            print("contoursImg is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            self.coor = []
            self.rectimg = self.contimg.copy()
            i = 0
            count = len(self.contours)
            while(i<count):
                rect = cv2.minAreaRect(self.contours[i])
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(self.rectimg,[box], 0, (255,0,0), 2)
                origin = ()
                circle = ()
                for j in range(1,3):
                    if box[j][0] < box[j-1][0] and box[j][1] > box[j+1][1]:
                        origin = origin + (box[j][0],box[j][1])
                if box[3][0] < box[2][0] and box[3][1] > box[0][1]:
                    origin = origin + (box[3][0],box[3][1])
                if box[0][0] < box[3][0] and box[0][1] > box[1][1]:
                    origin = origin + (box[0][0],box[0][1])
                if len(origin) == 4:
                    k = float((origin[3]-origin[1]))/(origin[2]-origin[0])
                    k = math.atan(k)/3.1415926*180
                    if k <= 45:
                        if origin[0] < origin[2]:
                            circle = (origin[0],origin[1])
                        else:
                            circle = (origin[2],origin[3])
                    if k > 45:
                        if origin[1] > origin[3]:
                            circle = (origin[0],origin[1])
                        else:
                            circle = (origin[2],origin[3])
                else:
                    circle = origin
                    k = 90
                if k != 90:
                    rows,cols,channel = self.rectimg.shape
                    rad = int(math.sqrt((rect[1][1])**2 + (rect[1][0])**2)/2)+2
                    x1 = round(rect[0][0])-rad
                    x2 = round(rect[0][0])+rad
                    y1 = round(rect[0][1])-rad
                    y2 = round(rect[0][1])+rad
                    if x1 <= 0:
                        x1 = 0
                    if x2 >= cols:
                        x2 = cols
                    if y1 <= 0:
                        y1 = 0
                    if y2 >= rows:
                        y2 = rows
                    roiimg = self.img[int(y1):int(y2),int(x1):int(x2)]
                    row,col,chan = roiimg.shape
                    if row > 0 and col > 0: 
                        rotMat = cv2.getRotationMatrix2D((col/2,row/2),k,1)
                        modimg = cv2.warpAffine(roiimg,rotMat,(col,row))
                        reaimg = modimg[int(row/2-rect[1][0]/2):int(row/2+rect[1][0]/2),int(col/2-rect[1][1]/2):int(col/2+rect[1][1]/2)]
                        if reaimg.shape[0] > 0 and reaimg.shape[1] > 0:
                            if reaimg.shape[0] < reaimg.shape[1]:
                                reaimg = np.rot90(reaimg,3)
                            self.sizimg = cv2.resize(reaimg,(40,60),interpolation=cv2.INTER_CUBIC)
                if k == 90:
                    reaimg = self.img[int(rect[0][1]-rect[1][1]/2):int(rect[0][1]+rect[1][1]/2),int(rect[0][0]-rect[1][0]/2):int(rect[0][0]+rect[1][0]/2)] 
                    if reaimg.shape[0] > 0 and reaimg.shape[1] > 0:
                        if reaimg.shape[0] < reaimg.shape[1]:
                            reaimg = np.rot90(reaimg,3)
                        self.sizimg = cv2.resize(reaimg,(40,60),interpolation=cv2.INTER_CUBIC)
                if self.sizimg is not None:
                    graysizimg = cv2.cvtColor(self.sizimg,cv2.COLOR_BGR2GRAY)
                    # cv2.imshow("graysizing",graysizimg)
                    sample = hog(graysizimg)
                    label = self.model.predict(sample)[1].ravel()
                    label = int(label)
                    self.coor.append([circle[0],circle[1],label])
                cv2.circle(self.rectimg, circle, 5, (0,0,255), -1)
                i = i + 1
            self.coor.sort()
            return self.coor
    def Getline(self):
        '''获取五线谱中线的信息,两条线的距离自己设置'''
        caimg = calibrate(self.img)
        # self.figure = getLine(caimg,30) #这是自动获取五线谱中的线
        # self.figure = manugetLine(caimg)#这是手动获取五线谱中的线
        self.figure = lined() #这是获取保存在line.txt文件中的线
        if len(self.figure) != 5:
            print("figure is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            return self.figure
    def GetList(self):
        '''得到音符的信息，它的左下角位置坐标，它的音符形状'''
        if self.coor is None:
            print("coor is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
            exit()
        else:
            if len(self.figure) != 5:
                print("figure is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
                exit()
            else:
                self.info = []
                for i in range(0,len(self.coor)):
                    # if self.coor[i][1] > self.figure[0][0]-10 and self.coor[i][1] <= self.figure[0][0]+10:
                    #     self.info.append(5.5)
                    # if self.coor[i][1] > self.figure[1][0]-10 and self.coor[i][1] <= self.figure[1][0]+10:
                    #     self.info.append(4.5)
                    # if self.coor[i][1] > self.figure[2][0]-10 and self.coor[i][1] <= self.figure[2][0]+10:
                    #     self.info.append(3.5)
                    # if self.coor[i][1] > self.figure[3][0]-10 and self.coor[i][1] <= self.figure[3][0]+10:
                    #     self.info.append(2.5)
                    # if self.coor[i][1] > self.figure[4][0]-10 and self.coor[i][1] <= self.figure[4][0]+10:
                    #     self.info.append(1.5)
                    # if self.coor[i][1] > self.figure[0][0]+10 and self.coor[i][1] <= self.figure[1][0]-10:
                    #     self.info.append(5)
                    # if self.coor[i][1] > self.figure[1][0]+10 and self.coor[i][1] <= self.figure[2][0]-10:
                    #     self.info.append(4)
                    # if self.coor[i][1] > self.figure[2][0]+10 and self.coor[i][1] <= self.figure[3][0]-10:
                    #     self.info.append(3)
                    # if self.coor[i][1] > self.figure[3][0]+10 and self.coor[i][1] <= self.figure[4][0]-10:
                    #     self.info.append(2)
                    # if self.coor[i][1] > self.figure[4][0]+10 and self.coor[i][1] <= self.figure[4][0]+30:
                    #     self.info.append(1)
                    if self.coor[i][1] > self.figure[4][0]+10 and self.coor[i][1] < self.figure[4][0]+30:
                        self.info.append((1,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[4][0]-10 and self.coor[i][1] < self.figure[4][0]+10:
                        self.info.append((1.5,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[3][0]+10 and self.coor[i][1] < self.figure[4][0]-10:
                        self.info.append((2,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[3][0]-10 and self.coor[i][1] < self.figure[3][0]+10:
                        self.info.append((2.5,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[2][0]+10 and self.coor[i][1] < self.figure[3][0]-10:
                        self.info.append((3,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[2][0]-10 and self.coor[i][1] < self.figure[2][0]+10:
                        self.info.append((3.5,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[1][0]+10 and self.coor[i][1] < self.figure[2][0]-10:
                        self.info.append((4,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[1][0]-10 and self.coor[i][1] < self.figure[1][0]+10:
                        self.info.append((4.5,self.coor[i][2]))
                    if self.coor[i][1] > self.figure[0][0]+10 and self.coor[i][1] < self.figure[1][0]-10:
                        self.info.append((5,self.coor[i][2]))
                return self.info

    def debugshow(self):
        '''调试时显示各个状态的图片'''
        if globallist.imgflag is True or globallist.debugflag is True:
            cv2.imshow("originimg",self.img)
        if globallist.binaryimgflag is True or globallist.debugflag is True:
            cv2.imshow("binaryimg",self.binaryimg)
        if globallist.erodimgflag is True or globallist.debugflag is True:
            cv2.imshow("erodimg",self.erodeimg)
        if globallist.contoursflag is True or globallist.debugflag is True:
            print("contours:")
            print(self.contours)
        if globallist.contimgflag is True or globallist.debugflag is True:
            cv2.imshow("contimg",self.contimg)
        if globallist.rectimgflag is True or globallist.debugflag is True:
            cv2.imshow("rectimg",self.rectimg)
        if globallist.coorflag is True or globallist.debugflag is True:
            print("coor:")
            print(self.coor)
        if globallist.figureflag is True or globallist.debugflag is True:
            print("figure:")
            print(self.figure)
        if globallist.infoflag is True or globallist.debugflag is True:
            print("info:")
            print(self.info)
    def Beginmark(self):
        img = self.img[globallist.begincoor[1]-20:globallist.begincoor[1]+20,globallist.begincoor[0]-20:globallist.begincoor[0]+20]
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        count = 0
        for i in range(0,40):
            for j in range(0,40):
                if img[i,j] < 128: #越黑，灰度值越低
                    count = count + 1
        if count/1600 > 0.5:
            self.beginFlag = True
        else:
            self.beginFlag = False




        


             


            


            

        

    


