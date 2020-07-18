#numpy包的学习
import numpy as np
import cv2 as cv
import time as time

def access_pixel(image):
    print(image.shape);
    HEIGHT=image.shape[0]
    Width=image.shape[1]
    Chancel=image.shape[2]
    print("height:%s,width:%s,chancel:%s"%(HEIGHT,Width,Chancel))
    for row in range(HEIGHT):
        for col in range(Width):
            for c in range(Chancel):
                pv=image[row,col,c]
                image[row,col,c]=255-pv
    cv.imshow('pixel',image)

def create_image():
    img=np.ones([400,400,3],np.double)
    cv.imshow('new image',img)

def inverse(image):
    dst=cv.bitwise_not(image)
    cv.imshow('照片取反',dst)

src =cv.imread('example.jpg')
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('input image', src)
# t1=cv.getTickCount() #大概估计的时间
start_time=time.time()
inverse(src)
end_time=time.time()
# t2=cv.getTickCount() 
# time=(t2-t1)/cv.getTickFrequency() 
print('time:%s ms'%((end_time-start_time)*1000))
cv.waitKey(0)
cv.destroyAllWindows()