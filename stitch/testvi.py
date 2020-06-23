import cv2

mp4 = cv2.VideoCapture("E:\\github\\stitch\\stitch\\video.mp4")  # 读取视频
is_opened = mp4.isOpened()  # 判断是否打开
print(is_opened)
fps = mp4.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
print(fps)
fps=fps*10
widght = mp4.get(cv2.CAP_PROP_FRAME_WIDTH)  # 获取视频的宽度
height = mp4.get(cv2.CAP_PROP_FRAME_HEIGHT)  # 获取视频的高度
print(str(widght) + "x" + str(height))
i = 0
while is_opened:
    if i == 10:  # 截取前10张图片
      break
    else:
      i+= 1
    (flag, frame) = mp4.read()  # 读取图片
    file_name = "iamge" + str(i) + ".jpg"
    print(file_name)
    if flag == True:
        cv2.imwrite(file_name, frame, [cv2.IMWRITE_JPEG_QUALITY])  # 保存图片

print("转换完成")
