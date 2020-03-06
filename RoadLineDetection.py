import cv2
import numpy as np
import matplotlib.pylab as plt


def regionOfInt( img , vertices) :
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask , vertices , match_mask_color)
    masked_img  = cv2.bitwise_and(img , mask)
    return masked_img


def draw_line(img , lines):
    img = np.copy(img)
    line_image = np.zeros((img.shape[0] ,img.shape[1] ,img.shape[2]) , dtype= np.uint8)

    for line in lines:
        for x1, y1, x2 , y2 in line:
            cv2.line(line_image , (x1,y1) , (x2,y2) , (0 , 255 ,255) , 3)
 
    img = cv2.addWeighted(img , 0.8 , line_image , 1 , 0.0)
    return img

img = cv2.imread("lane.jpg")
img = cv2.cvtColor(img ,cv2.COLOR_BGR2RGB)

print(img.shape)
height = img.shape[0]
width = img.shape[1]

roi = [
    (0 , height) , (width/2 , height/2 ) , (width  , height)
]
gray = cv2.cvtColor(img , cv2.COLOR_RGB2GRAY)
cannyimg = cv2.Canny(gray , 100 , 260)
croped_image =  regionOfInt(cannyimg , np.array([roi] , np.int32),)

lines = cv2.HoughLinesP(croped_image ,rho= 6 , theta = np.pi/60 ,threshold =80 ,lines = np.array([]) , minLineLength= 40 , maxLineGap= 25)

image_line = draw_line(img , lines)

plt.imshow(image_line)
plt.show()
