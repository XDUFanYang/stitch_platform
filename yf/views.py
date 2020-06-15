from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# Create your views here.
from stitch import settings
from django.conf import settings  # 导入settings
import os  # 创建文件夹需要的包


import numpy as np
import imutils
import cv2


# import pinjie

def homeUI(request):
    return render(request, 'homeUI.html', {})

def fun2(request):
    return render(request, 'fun2.html', {})

def fun3(request):
    return render(request, 'fun3.html', {})

def aboutus(request):
    return render(request, 'aboutus.html', {})


def picshow(request):
    return render(request, 'picshow.html', {})


def pagenumerror(request):
        return render(request, 'pagenumerror.html', {})

class Stitcher:
    def __init__(self):
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):

        # 简单地检测关键点并从两个图像中提取局部不变量描述符SIFT并匹配
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)

        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 没有足够的匹配关键点，返回空
        if M is None:
            return None

        # 应用透视变换将图像缝合在一起
        (matches, H, status) = M
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检查是否应该可视化关键点匹配
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            return (result, vis)

        # 返回缝合图像
        return result

    def detectAndDescribe(self, image):
        """
        :param image:
        :return: 特征描述点
        """
        descriptor = cv2.xfeatures2d.SIFT_create()
        (kps, features) = descriptor.detectAndCompute(image, None)
        kps = np.float32([kp.pt for kp in kps])
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 计算两组点之间的单应性需要  至少初始的四组匹配。
        if len(matches) > 4:
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
            return (matches, H, status)
        return None

    # 可视化
    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 仅当关键点成功匹配时才处理匹配
            if s == 1:
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
        return vis


def uploadtxt(request):
    if request.method == 'GET':
        # img = PictureModel.objects.get(id=18)
        return render(request, 'homeUI.html')
    else:
        # 需要从表单input中，获取上传的文件对象(图片)
        # pic = request.FILES.get('picture')
        pic = request.FILES.getlist('picture')
        if len(pic)!=2:
            return render(request, 'pagenumerror.html')
        print(pic[0].name)
        print(pic[1].name)

        # 1. 创建Model对象，保存图片路径到数据库
        # (这里先不写)
        # model = PictureModel()
        # model.pic_url = pic.name
        # model.save()
        # 2. 开始处理图片，将图片写入到指定目录。(/static/media/images/)
        # 拼接图片路径
        url0 = settings.MEDIA_ROOT + 'images/' + pic[0].name
        url1 = settings.MEDIA_ROOT + 'images/' + pic[1].name
        with open(url0, 'wb') as f0:
            # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
            for data in pic[0].chunks():
                f0.write(data)

        with open(url1, 'wb') as f1:
            # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
            for data in pic[1].chunks():
                f1.write(data)

        imageA = cv2.imread(url0)
        imageB = cv2.imread(url1)

        imageA = imutils.resize(imageA, width=400)
        imageB = imutils.resize(imageB, width=400)
        # 将图像缝合在一起以创建全 景图
        stitcher = Stitcher()
        (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)


        cv2.imwrite('E:/stitch/static/media/images/stitch_result.jpg', result)

        picpath = 'http://127.0.0.1:8000/media/images/stitch_result.jpg'

        #imwrite()路径问题也已经解决

        #return render(request, 'login.html', {'img_href': picpath})

        #return HttpResponse('图片上传成功')
        return render(request, 'picshow.html', {'path': picpath})


def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def MSR(img, scales):
    weight = 1 / 3.0
    scales_size = len(scales)
    h, w = img.shape[:2]
    log_R = np.zeros((h, w), dtype=np.float32)

    for i in range(scales_size):
        img = replaceZeroes(img)
        L_blur = cv2.GaussianBlur(img, (scales[i], scales[i]), 0)
        L_blur = replaceZeroes(L_blur)
        dst_Img = cv2.log(img/255.0)
        dst_Lblur = cv2.log(L_blur/255.0)
        dst_Ixl = cv2.multiply(dst_Img, dst_Lblur)
        log_R += weight * cv2.subtract(dst_Img, dst_Ixl)

    dst_R = cv2.normalize(log_R,None, 0, 255, cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8

def uploadtxt1(request):
    if request.method == 'GET':
        # img = PictureModel.objects.get(id=18)
        return render(request, 'homeUI.html')
    else:
        # 需要从表单input中，获取上传的文件对象(图片)
        # pic = request.FILES.get('picture')
        pic = request.FILES.getlist('picture')
        #print(pic)
        if len(pic)!=1:
            return render(request, 'pagenumerror.html')
        print(pic[0].name)

        # 1. 创建Model对象，保存图片路径到数据库
        # (这里先不写)
        # model = PictureModel()
        # model.pic_url = pic.name
        # model.save()
        # 2. 开始处理图片，将图片写入到指定目录。(/static/media/images/)
        # 拼接图片路径
        url0 = settings.MEDIA_ROOT + 'images/' + pic[0].name
        #url1 = settings.MEDIA_ROOT + 'images/' + pic[1].name
        with open(url0, 'wb') as f0:
            # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
            for data in pic[0].chunks():
                f0.write(data)

        #with open(url1, 'wb') as f1:
            # pic.chunks()循环读取图片内容，每次只从本地磁盘读取一部分图片内容，加载到内存中，并将这一部分内容写入到目录下，写完以后，内存清空；下一次再从本地磁盘读取一部分数据放入内存。就是为了节省内存空间。
        #    for data in pic[1].chunks():
        #        f1.write(data)

        imageA = cv2.imread(url0)
        #imageB = cv2.imread(url1)

        #imageA = imutils.resize(imageA, width=400)
        #imageB = imutils.resize(imageB, width=400)
        # 将图像缝合在一起以创建全 景图
        #stitcher = Stitcher()
        #(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
        scales = [15, 101, 301]  # [3,5,9]  #看不出效果有什么差别
        src_img = cv2.imread(url0)
        x, y = src_img.shape[0:2]
        src_img = cv2.resize(src_img, (int(y / 5), int(x / 5)))
        b_gray, g_gray, r_gray = cv2.split(src_img)
        b_gray = MSR(b_gray, scales)
        g_gray = MSR(g_gray, scales)
        r_gray = MSR(r_gray, scales)
        result = cv2.merge([b_gray, g_gray, r_gray])


        cv2.imwrite('E:/stitch/static/media/images/result1.jpg', result)

        picpath = 'http://127.0.0.1:8000/media/images/result1.jpg'

        #imwrite()路径问题也已经解决

        #return render(request, 'login.html', {'img_href': picpath})

        #return HttpResponse('图片上传成功')
        return render(request, 'picshow.html', {'path': picpath})

