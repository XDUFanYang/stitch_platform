from django.conf.urls import url
from django.urls import path, re_path
from . import views
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path('', views.home, name="home"),
    path('uploadtxt/',views.uploadtxt,name="uploadtxt"),
    path('pic_show/', views.pic_show, name='pic_show'),
    path('index/', views.index),
    url(r'^index/static/(?P<path>.*)$', serve, {'document_root':'F:\yf_stitch\index\static'})
    #这句意思是将访问的图片href由“http://127.0.0.1:8888/media/图片存储文件夹/字母哥.jpg”转为本地访问D:\workspace\upload_pic\media的形式
]