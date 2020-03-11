from django.shortcuts import render
from django.http import HttpResponse, JsonResponse

# Create your views here.
from yf_stitch import settings
from django.conf import settings       #导入settings
import os                   #创建文件夹需要的包
from yf_demo1 import models

import numpy as np
import imutils
import cv2

#import pinjie

def home(request):
    return render(request, 'home.html', {})


def pic_show(request):
    return render(request, 'pic_show.html', {})



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
        # return render(request, 'index.html', {'img': img})
        return render(request, 'home.html')
    else:
        # 需要从表单input中，获取上传的文件对象(图片)
        #pic = request.FILES.get('picture')
        pic = request.FILES.getlist('picture')
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

        cv2.imwrite('./index/static/images/stitch_result.jpg', result)
        
        picpath='http://127.0.0.1:8000/index/static/images/stitch_result.jpg'
        
        #imwrite()路径问题也已经解决
        cv2.waitKey(2000)
        return render(request, 'login.html', {'img_href': picpath})
        
        #return HttpResponse('图片上传成功')
        # return render(request, 'pic_show.html', {'path': url})

def index(request):
    if request.method == 'POST':
        img = request.FILES.get('img')
        path = settings.MEDIA_ROOT1
        file = '图片存储文件夹'
        pic_path = path + '/' + file
        print(pic_path)
        isExists = os.path.exists(pic_path)  			# 路径存在则返回true，不存在则返回false
        if isExists:
            print("目录已存在")
        else:
            os.mkdir(pic_path)
            print("创建成功")
        img_url = pic_path + '/' + img.name
        print(img_url)
        with open(img_url, 'wb') as f:                #将图片以二进制的形式写入
            for data in img.chunks():
                f.write(data)
        pic_data = 'http://127.0.0.1:8000/media' + '/' + 'file' + img.name
        #将路径转化一下，转为href的形式，然后存入数据库，发到后端
        models.imginfo.objects.create(img=pic_data)                     #生成一条数据
        img_href = models.imginfo.objects.filter(id=1)[0]
        #将id为1的数据取出

        print('pic_data路径:'+pic_data)
        return render(request, 'login.html', {'img_href': pic_data})
    return render(request, 'index.html')


def login(request):
        return render(request, 'login.html')
