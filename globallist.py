# -*- coding: utf-8 -*-
debugflag = False #它是True时，所有flag全为True
imgflag = True  #原始图片
binaryimgflag = False  #这是二值化或者自适应二值化得到的图
erodimgflag = False #腐蚀得到的图像
contoursflag = False #音符边框信息
contimgflag = False #把轮廓画在这张图上
rectimgflag = True #画最小矩形
sizimgflag = False #标准大小的音符图片
coorflag = False #音符左下角坐标
figureflag = False #五线谱中线的信息，r和theta
infoflag = False #音符的位置和形状信息
begincoor = [315,380] #开始信号的标记位