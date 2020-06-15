import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import matplotlib as plt
img = cv.imread('img1.jpg', 0)
# 图像归一化
fi = img / 255.0
# 伽马变换
gamma = 0.4
out = np.power(fi, gamma)
##解决显示过大的图像问题
#cv.nameWindow('demo',0)
#cv.resizeWindow('demo',600,500)
#cv2.namedWindow('test', cv2.WINDOW_NORMAL)
#cv2.resizeWindow('test', 1000, 1000)
#缩放显示
x, y = img.shape[0:2]
img = cv.resize(img, (int(y / 10), int(x / 10)))
cv.imshow("img", img)
#缩放显示
x, y = out.shape[0:2]
out = cv.resize(out, (int(y / 10), int(x / 10)))
cv.imshow("out", out)

cv.waitKey(0)
