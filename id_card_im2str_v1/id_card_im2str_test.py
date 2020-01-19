# -*- coding: utf-8 -*-
# 摘自 https://blog.csdn.net/qq_37674858/article/details/80497563


import pytesseract
import cv2  # pip install opencv-python
import matplotlib.pyplot as plt
import dlib  # pip install xxx.whl
import matplotlib.patches as mpatches
from skimage import io, draw, transform, color
import numpy as np
import pandas as pd
import re


img_path = '../data/id_1_2.jpg'
shape_predictor_path = '../dlib_file/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
image = io.imread(img_path)
plt.figure()
ax = plt.subplot(111)
ax.imshow(image)
plt.show()

face_num = 0  # 识别到的人脸个数
for i in range(4):  # 尝试识别图片中的人脸4次
    plt.figure()
    ax = plt.subplot(111)
    ax.imshow(image)
    plt.show()

    dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果
    face_num = len(dets)
    if face_num > 0:
        break
    else:
        # 将图片逆时针旋转90度再识别一次
        image = transform.rotate(image, 90.0, clip=False)  # 旋转后图片会被切割？？？？？？？背景只能是黑色的？？？？？
        image = np.uint8(image * 255)
if face_num < 1:
    raise Exception('未识别到人脸')


# 将识别的图像可视化
plt.figure()
ax = plt.subplot(111)
ax.imshow(image)
plt.axis("off")
for i, face in enumerate(dets):
    # 在图片中标注人脸，并显示
    left = face.left()
    top = face.top()
    right = face.right()
    bottom = face.bottom()
    rect = mpatches.Rectangle((left, bottom), right - left, top - bottom,
                              fill=False, edgecolor='red', linewidth=1)
    ax.add_patch(rect)
plt.show()

predictor = dlib.shape_predictor(shape_predictor_path)
detected_landmarks = predictor(image, dets[0]).parts()
landmarks = np.array([[p.x, p.y] for p in detected_landmarks])

# 可视化脸部特征点
plt.figure()
ax = plt.subplot(111)
ax.imshow(image)
plt.axis("off")
# plt.plot(landmarks[0:4, 0], landmarks[0:4, 1], 'ro')  # 画特征点
plt.plot(landmarks[:, 0], landmarks[:, 1], 'ro')  # 画特征点
for ii in np.arange(len(landmarks)):
    # plt.text(landmarks[ii, 0] - 10, landmarks[ii, 1] - 15, ii)  # 原点在右下角  # (x-10, y-15)会让点向左上角移动
    plt.text(landmarks[ii, 0] - 2, landmarks[ii, 1] - 2, ii)  # 原点在右下角  # (x-2, y-2)会让点向左上角移动
plt.show()

# 将眼睛位置可视化
plt.figure()
ax = plt.subplot(111)
ax.imshow(image)
plt.axis("off")
# plt.plot(landmarks[0:4, 0], landmarks[0:4, 1], 'ro')  # 画特征点  # 使用5个特征点时，眼睛的索引
plt.plot(landmarks[36:48, 0], landmarks[36:48, 1], 'ro')  # 画特征点  # 使用68个特征点时，眼睛的索引
# for ii in np.arange(4):
for ii in np.arange(36, 48):
    plt.text(landmarks[ii, 0] - 12, landmarks[ii, 1] - 0, ii, fontSize=8)  # 原点在右下角  # (x-10, y-15)会让点向左上角移动
plt.show()


# 计算眼睛的倾斜角度,逆时针角度
def twopointcor(point1, point2):
    """point1 = (x1,y1),point2 = (x2,y2)"""
    deltxy = point2 - point1
    corner = np.arctan(deltxy[1] / deltxy[0]) * 180 / np.pi  # 角度
    return corner


# 计算多个角度求均值
# corner10 = twopointcor(landmarks[1, :], landmarks[0, :])
# corner23 = twopointcor(landmarks[3, :], landmarks[2, :])
# corner20 = twopointcor(landmarks[2, :], landmarks[0, :])
# corner = np.mean([corner10, corner23, corner20])
# print(corner10)
# print(corner23)
# print(corner20)
# print(corner)

# 计算多个角度求均值
corner_42_45 = twopointcor(landmarks[42, :], landmarks[45, :])
corner_39_36 = twopointcor(landmarks[39, :], landmarks[36, :])
corner_36_45 = twopointcor(landmarks[36, :], landmarks[45, :])
corner = np.mean([corner_42_45, corner_39_36, corner_36_45])
print('corner_42_45：', corner_42_45)
print('corner_39_36：', corner_39_36)
print('corner_36_45：', corner_36_45)
print('corner：', corner)


# 计算图像的身份证倾斜的角度
def IDcorner(landmarks):
    """landmarks:检测的人脸5个特征点
       经过测试使用第0个和第2个特征点计算角度较合适
    """
    # corner20 = twopointcor(landmarks[2, :], landmarks[0, :])
    # corner = np.mean([corner20])
    corner_36_45 = twopointcor(landmarks[36, :], landmarks[45, :])
    corner = np.mean([corner_36_45])
    return corner


corner = IDcorner(landmarks)
print('corner_36_45：', corner)


# 将照片转正
def rotateIdcard(image):
    "image :需要处理的图像"
    # 使用dlib.get_frontal_face_detector识别人脸
    detector = dlib.get_frontal_face_detector()
    dets = detector(image, 2)  # 使用detector进行人脸检测 dets为返回的结果
    # 检测人脸的眼睛所在位置
    predictor = dlib.shape_predictor(shape_predictor_path)
    detected_landmarks = predictor(image, dets[0]).parts()
    landmarks = np.array([[p.x, p.y] for p in detected_landmarks])
    corner = IDcorner(landmarks)
    # 旋转后的图像
    image2 = transform.rotate(image, corner, clip=False)
    image2 = np.uint8(image2 * 255)
    # 旋转后人脸位置
    det = detector(image2, 2)
    return image2, det


# 转正身份证：
image = io.imread(img_path)
image2, dets = rotateIdcard(image)

# 可视化修正后的结果
plt.figure()
ax = plt.subplot(111)
ax.imshow(image2)
plt.axis("off")
# 在图片中标注人脸，并显示
left = dets[0].left()
top = dets[0].top()
right = dets[0].right()
bottom = dets[0].bottom()
rect = mpatches.Rectangle((left, bottom), (right - left), (top - bottom),
                          fill=False, edgecolor='red', linewidth=1)
ax.add_patch(rect)

# 照片的位置（不怎么精确）
width = right - left
high = top - bottom
left2 = np.uint(left - 0.5 * width)
bottom2 = np.uint(bottom + 0.5 * width)
rect = mpatches.Rectangle((left2, bottom2), 1.8 * width, 2.2 * high,
                          fill=False, edgecolor='blue', linewidth=1)
ax.add_patch(rect)
plt.show()

# 身份证上人的照片
top2 = np.uint(bottom2 + 2.2 * high)
right2 = np.uint(left2 + 1.8 * width)
image3 = image2[top2:bottom2, left2:right2, :]
plt.imshow(image3)
plt.axis("off")
plt.show()
# cv2.imshow('image3', image3)
# cv2.waitKey()

# # 对图像进行处理，转化为灰度图像 --> 二值图像
# imagegray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
# # cv2.imshow('imagegray', imagegray)
# # cv2.waitKey()
# retval, imagebin = cv2.threshold(imagegray, 120, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# # 将照片去除
# imagebin[0:bottom2, left2:-1] = 255
# # 高斯双边滤波
# img_bilateralFilter = cv2.bilateralFilter(imagebin, 40, 75, 75)
#
# # cv2.imshow('img_bilateralFilter', img_bilateralFilter)
# # cv2.waitKey()
# plt.imshow(img_bilateralFilter, cmap=plt.cm.gray)
# plt.axis("off")
# plt.show()

# 对图像进行处理，转化为灰度图像 --> 二值图像
# 灰度处理
imagegray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('imagegray', imagegray)
# 二值化
retval, imagebin = cv2.threshold(imagegray, 50, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
# 自适应二值化
# imagebin = cv2.adaptiveThreshold(imagegray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
# # 开运算
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # getStructuringElement 可以方便的生成一个矩阵（kernel）
# im_open = cv2.morphologyEx(imagebin, cv2.MORPH_OPEN, kernel)
# # 自适应二值化
# imagebin = cv2.adaptiveThreshold(im_open, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
# 高斯模糊
kernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
im_blur = cv2.filter2D(imagebin, -1, kernel)
# 自适应二值化
imagebin = cv2.adaptiveThreshold(im_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 1)
# 将照片去除
imagebin[0:bottom2, left2:-1] = 255
# 高斯双边滤波
img_bilateralFilter = cv2.bilateralFilter(imagebin, 40, 100, 100)  # 高斯双边滤波

plt.imshow(img_bilateralFilter, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
# cv2.namedWindow("img_bilateralFilter", cv2.WINDOW_NORMAL)
# cv2.imshow('img_bilateralFilter', img_bilateralFilter)
# cv2.waitKey(0)


# 可以通过pytesseract库来查看检测效果，但是结果并不是很好
text = pytesseract.image_to_string(imagebin, lang='chi_sim')
print(text)


# 对识别结果处理
textlist = text.split("\n")
textdf = pd.DataFrame({"text": textlist})
textdf["textlen"] = textdf.text.apply(len)
# 去除长度<=1的行
textdf = textdf[textdf.textlen > 1].reset_index(drop=True)


# 提取相应的信息
print("姓名:", textdf.text[0])
print("=====================")
print("性别:", textdf.text[1].split(" ")[0])
print("=====================")
print("民族:", textdf.text[1].split(" ")[-1])
print("=====================")
yearnum = textdf.text[2].split(" ")[0]  # 提取数字
yearnum = re.findall("\d+", yearnum)[0]
print("出生年:", yearnum)
print("=====================")
# monthnum = textdf.text[2].split(" ")[1]  # 提取数字
# monthnum = re.findall("\d+", monthnum)[0]
# print("出生月:", monthnum)
# print("=====================")
# daynum = textdf.text[2].split(" ")[2]  ## 提取数字
# daynum = re.findall("\d+", daynum)[0]
# print("出生日:", daynum)
# print("=====================")
IDnum = textdf.text.values[-1]
if (len(IDnum) > 18):  # 去除不必要的空格
    IDnum = IDnum.replace(" ", "")
print("公民身份证号:", IDnum)
print("=====================")
# 获取地址，因为地址可能会是多行
desstext = textdf.text.values[3:(textdf.shape[0] - 1)]
print("地址:", "".join(desstext))
print("=====================")
pass
