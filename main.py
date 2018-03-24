# -*- coding: utf-8 -*-
import time
import pygame
from EasyMIDI import EasyMIDI,Track,Note,Chord,RomanChord
import cv2
import numpy as np
import math
import sys
import os
import time
import threading
from picture import *
import globallist
#参数
scale = 1  #这是摄像头原始图像缩放倍数
yinlist=[]  #音符的信息
endFlag = False
playFlag = False
class pictureTrack(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global yinlist
        global playFlag
        global endFlag
        cap = cv2.VideoCapture(1)
        for a in range(0,3):  #先读取三帧
            ret,Img = cap.read()
        if ret is True:
            Img = cv2.resize(Img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            note = Picture(Img)
            note.Getline()  #这是唯一一次找五线谱的线
        else:
            print("where is the camera")
        while(ret is True):
            key = cv2.waitKey(1)
            note.ConvBinary(adaptflag=1)
            cv2.circle(note.img, (globallist.begincoor[0],globallist.begincoor[1]), 30, (0,255,0), 1)
            note.Erode()
            note.Contours()
            note.GetRect()
            note.GetList()
            yinlist = note.info
            note.Beginmark()
            playFlag = note.beginFlag
            note.debugshow()  #需要调试时运行这个程序
            ret,Img = cap.read()
            Img = cv2.resize(Img,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
            note.Update(Img)
            # note.Getline()  #这边有个巨大注意事项：第一帧得不到五线谱的形状，得从第二帧开始，摄像头才读取五线谱,这个放到上面去了      
            if key == 27:
                break
        cap.release()
        endFlag = True
        cv2.destroyAllWindows()


# class soundTrack(threading.Thread):
#     def __init__(self):
#         threading.Thread.__init__(self)
#     def run(self):
#         global yinlist
#         global endFlag
#         song = Noteout(message=yinlist,playFlag=True)
#         while(endFlag is False):
#             if playFlag is True and yinlist is not None:
#                 i = 0
#                 while(i < len(yinlist)):
#                     midAdd = "example.mid"
#                     song.updateTrack(message=yinlist[i],writeFlag=True,writeAdd=midAdd)
#                     print(yinlist[i])
#                     song.performance(fileAdd=midAdd)
#                     i = i + 1
#             else :
#                 time.sleep(1)
class soundTrack(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        global yinlist
        global playFlag
        global endFlag
        while(endFlag is False):
            if  yinlist != [] and playFlag is True:
                name = []
                vol = 100
                easyMIDI = EasyMIDI()
                track = Track("acoustic grand piano")
                for item in yinlist:
                    dur = []
                    octave = []
                    if item[0] == 1:
                        octave = 1
                    if item[0] == 1.5:
                        octave = 2
                    if item[0] == 2:
                        octave = 3
                    if item[0] == 2.5:
                        octave = 4
                    if item[0] == 3:
                        octave = 5
                    if item[0] == 3.5:
                        octave = 6
                    if item[0] == 4:
                        octave = 7
                    if item[1] == 0:
                        dur = 1/2
                    if item[1] == 1:
                        dur = 1/4
                    if item[1] == 2:
                        dur = 1/8
                    if item[1] == 3:
                        name = "F"
                    if item[1] == 4:
                        name = "G"
                    if item[1] == 10:
                        name = "C"
                    if name == []:  #如果没有谱号
                        print("name of Note is None!",sys._getframe().f_lineno,sys._getframe().f_code.co_name)
                    else:
                        if dur != [] and octave != []:
                            pitch = Note(name,octave,dur,vol)
                            track.addNotes(pitch)
                easyMIDI.addTrack(track)
                easyMIDI.writeMIDI("example.mid")
                file = r'example.mid'
                pygame.mixer.init()
                print("play music")
                track = pygame.mixer.music.load(file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pass
                pygame.mixer.music.stop()
                os.remove("example.mid")
            else:
                time.sleep(1)



if __name__ == "__main__":
    thread1 = pictureTrack()
    thread2 = soundTrack()
    thread2.start()
    thread1.start()
