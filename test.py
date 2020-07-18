import cv2 as cv
import numpy as np

def video_demo():
    capture = cv.VideoCapture(0)
    while True:
        ret,frame = capture.read()
        frame=cv.flip(frame,1)
        cv.imshow('video',frame)
        c = cv.waitKey(50)
        if c == 27:
            break

        



def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)


print("----------------分割线--------------------------------")
src =cv.imread('example.jpg')
cv.namedWindow('input image', cv.WINDOW_AUTOSIZE)
cv.imshow('input image', src)
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imwrite('example.bmp', gray)
# get_image_info(src)
video_demo()

cv.waitKey(0)
cv.destroyAllWindows()


