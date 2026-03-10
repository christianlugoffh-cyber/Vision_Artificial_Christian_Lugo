import cv2
import numpy as np

img = cv2.imread("bailarina.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_,th = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)

size = np.size(th)
skel = np.zeros(th.shape,np.uint8)

element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

done = False
img_temp = th.copy()

while not done:
    eroded = cv2.erode(img_temp,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img_temp,temp)
    skel = cv2.bitwise_or(skel,temp)
    img_temp = eroded.copy()

    zeros = size - cv2.countNonZero(img_temp)
    if zeros==size:
        done = True

cv2.imshow("Original Bailarina",img)
cv2.imshow("Preprocesada Bailarina",th)
cv2.imshow("Esqueleto Bailarina",skel)

cv2.waitKey(0)
cv2.destroyAllWindows()