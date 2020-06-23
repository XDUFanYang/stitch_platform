import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#import matplotlib as plt

img = cv.imread("jian11.jpg", 0)
# 图像归一化
fi = img / 255.0
# 伽马变换
gamma = 0.5
out = np.power(fi, gamma)
x, y = img.shape[0:2]
cv.imshow("img", img)
x, y = out.shape[0:2]
out = cv.resize(out, (int(y / 1), int(x / 1)))
cv.imwrite("jian11r.jpg", out*255)
cv.imshow("out_1", out)
cv.waitKey(0)
