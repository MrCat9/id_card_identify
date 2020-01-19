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


# 计算眼睛的倾斜角度,逆时针角度
def twopointcor(point1, point2):
    """point1 = (x1,y1),point2 = (x2,y2)"""
    deltxy = point2 - point1
    corner = np.arctan(deltxy[1] / deltxy[0]) * 180 / np.pi  # 角度
    return corner


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


# 将照片转正
def rotateIdcard(image):
    "image :需要处理的图像"
    # 使用dlib.get_frontal_face_detector识别人脸
    detector = dlib.get_frontal_face_detector()

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
            image = transform.rotate(image, 90.0, clip=True)  # 旋转后图片会被切割？？？？？？？背景只能是黑色的？？？？？
            image = np.uint8(image * 255)
    if face_num < 1:
        raise Exception('未识别到人脸')

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


def Idcard_im2str(image, threshod=120):
    # 转正身份证：
    image2, dets = rotateIdcard(image)
    # 提取照片的头像
    # 在图片中标注人脸，并显示
    left = dets[0].left()
    top = dets[0].top()
    right = dets[0].right()
    bottom = dets[0].bottom()
    # 照片的位置（不怎么精确）
    width = right - left
    high = top - bottom
    left2 = np.uint(left - 0.3 * width)
    bottom2 = np.uint(bottom + 0.4 * width)
    # 身份证上人的照片
    top2 = np.uint(bottom2 + 1.8 * high)
    right2 = np.uint(left2 + 1.6 * width)
    # [(left2,bottom2),(top2,right2)]
    rectangle = [(left2, bottom2), (top2, right2)]
    imageperson = image2[top2:bottom2, left2:right2, :]
    # 对图像进行处理，转化为灰度图像=>二值图像
    imagegray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    retval, imagebin = cv2.threshold(imagegray, threshod, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 将照片去除
    imagebin[0:bottom2, left2:-1] = 255
    # 通过pytesseract库来查看检测效果，但是结果并不是很好
    text = pytesseract.image_to_string(imagebin, lang='chi_sim')
    textlist = text.split("\n")
    textdf = pd.DataFrame({"text": textlist})
    textdf["textlen"] = textdf.text.apply(len)
    # 去除长度《＝1的行
    textdf = textdf[textdf.textlen > 1].reset_index(drop=True)
    return image2, dets, rectangle, imagebin, textdf


if __name__ == '__main__':
    img_path = '../data/id_1.jpg'
    # img_path = '../data/id_3.jpg'
    shape_predictor_path = '../dlib_file/shape_predictor_68_face_landmarks.dat'

    # 识别身份证的信息
    image = io.imread(img_path)
    image2, dets, rectangle, imagebin, textdf = Idcard_im2str(image, threshod=120)

    # # 提取相应的信息
    # print("姓名:", textdf.text[0])
    # print("=====================")
    # print("性别:", textdf.text[1].split(" ")[0])
    # print("=====================")
    # print("民族:", textdf.text[1].split(" ")[-1])
    # print("=====================")
    # yearnum = textdf.text[2].split(" ")[0]  # 提取数字
    # yearnum = re.findall("\d+", yearnum)[0]
    # print("出生年:", yearnum)
    # print("=====================")
    # monthnum = textdf.text[2].split(" ")[1]  # 提取数字
    # monthnum = re.findall("\d+", monthnum)[0]
    # print("出生月:", monthnum)
    # print("=====================")
    # daynum = textdf.text[2].split(" ")[2]  # 提取数字
    # daynum = re.findall("\d+", daynum)[0]
    # print("出生日:", daynum)
    # print("=====================")
    # IDnum = textdf.text.values[-1]
    # if (len(IDnum) > 18):  # 去除不必要的空格
    #     IDnum = IDnum.replace(" ", "")
    # print("公民身份证号:", IDnum)
    # print("=====================")
    # # 获取地址，因为地址可能会是多行
    # desstext = textdf.text.values[3:(textdf.shape[0] - 1)]
    # print("地址:", "".join(desstext))
    # print("=====================")

    # 对识别的信息进行可视化查看
    plt.figure(figsize=(12, 8))
    # 原始图像
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.axis("off")
    # 修正后图像
    ax = plt.subplot(2, 2, 2)
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

    # 照片的位置（不怎么精确）rectangle = [(left2,bottom2),(top2,right2)]
    width = rectangle[1][1] - rectangle[0][0]
    high = rectangle[1][0] - rectangle[0][1]
    left2 = rectangle[0][0]
    bottom2 = rectangle[0][1]
    rect = mpatches.Rectangle((left2, bottom2), width, high,
                              fill=False, edgecolor='blue', linewidth=1)
    ax.add_patch(rect)

    # 显示人的头像
    plt.subplot(2, 2, 3)
    # 身份证上人的照片
    top2 = bottom2 + high
    right2 = left2 + width
    image3 = image2[top2:bottom2, left2:right2, :]
    plt.imshow(image3)
    plt.axis("off")
    # 显示而值化图像
    plt.subplot(2, 2, 4)
    plt.imshow(imagebin, cmap=plt.cm.gray)
    plt.axis("off")
    plt.show()
