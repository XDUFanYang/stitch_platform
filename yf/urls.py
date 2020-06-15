from django.conf.urls import url
from django.urls import path, re_path
from django.conf import settings
from . import views
from django.views.static import serve

urlpatterns = [

    path('', views.homeUI,name="homeUI"),
    path('uploadtxt/', views.uploadtxt, name="uploadtxt"),
    path('uploadtxt1/', views.uploadtxt1, name="uploadtxt1"),
    path('picshow/', views.picshow, name='picshow'),
    path('aboutus/', views.aboutus,name="aboutus"),
    path('fun2/', views.fun2,name="fun2"),
    path('fun3/', views.fun3,name="fun3"),
    path('pagenumerror/', views.pagenumerror,name="pagenumerror"),
    re_path('media/(?P<path>.*)$', serve, {'document_root': r'E:\stitch\static\media'}),
    #这句意思是将访问的图片href由“http://127.0.0.1:8888/media/图片存储文件夹/字母哥.jpg”转为本地访问D:\workspace\upload_pic\media的形式

]
