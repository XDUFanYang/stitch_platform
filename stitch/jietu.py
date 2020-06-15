import cv2
import os.path
from tqdm import tqdm

file_path = r"C:\Users\12092\Desktop\PDF Expert\test1.mp4"
path_dir = os.listdir(file_path)  # 返回文件夹中的文件名
save_path = r"C:\Users\12092\Desktop\PDF Expert"
count = 1
name_count = 1
for allDir in tqdm(path_dir):
    video_path = file_path + allDir
    video = cv2.VideoCapture(video_path)  # 读入视频文件
    if video.isOpened():  # 判断是否正常打开
        rval, frame = video.read()
    else:
        rval = False

    timeF = 50  # 视频帧计数间隔频率

    while rval:  # 循环读取视频帧
        rval, frame = video.read()
        if (count % timeF == 0):  # 每隔timeF帧进行存储操作
            # cv2.imshow('pic',frame)
            cv2.imwrite(save_path + str(name_count) + '.jpg', frame)  # imwrite在py3中无法保存中文路径
            # cv2.imencode('.jpg', frame)[1].tofile(save_path + str(count) + '.jpg')  # 存储为图像
            # print('E:\Dataset\file\数据\image/' + '%06d' % c + '.jpg')
            name_count = name_count + 1
        count = count + 1
        cv2.waitKey(1)

video.release()
