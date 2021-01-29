import cv2
import struct_info as sInfo
from PIL import Image, ImageDraw, ImageFont
import numpy as np


fontpath = "/home/ucloudy/FaceTest/face_expression/platech.ttf"
ft = ImageFont.truetype(fontpath, 15,encoding="utf-8")
def adjustHWTo4(frame):
    '''
    :param frame: (HxWx4) from cv2.imread
    :return: (H'xW'x4)
    '''
    height, width = frame.shape[0], frame.shape[1]
    adjust_width = width - (width % 4)
    adjust_height = height - (height % 4)
    frame = cv2.resize(frame, (adjust_width, adjust_height))
    return frame


def featureToDict(face_feature):
    face_feature_dict = {}
    face_feature_dict['feature'] = face_feature.feature
    face_feature_dict['featureSize'] = face_feature.featureSize
    return face_feature_dict


def dictTofeature(face_feature_dict):
    faceFeature = sInfo.ASF_FaceFeature()
    faceFeature.feature = face_feature_dict['feature']
    faceFeature.featureSize = face_feature_dict['featureSize']
    return faceFeature


fontpath = "/home/ucloudy/FaceTest/face_expression/platech.ttf"
ft = ImageFont.truetype(fontpath, 15,encoding="utf-8")
def draw(img, nameLabel, expressLabel, box):
    x1 = box.left
    y1 = box.top
    x2 = box.right
    y2 = box.bottom
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # cv2.putText(img, nameLabel, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    allLabel = nameLabel + expressLabel
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw_img = ImageDraw.Draw(img_pil)
    draw_img.text((x1, y1), allLabel, font=ft, fill=(255, 255, 255))
    img_res = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_res


def draw_coor(img, nameLabel, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, nameLabel, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)




def mrectToRect(mrect):
    top = mrect.top
    left = mrect.left
    right = mrect.right
    bottom = mrect.bottom
    return [left, top, right, bottom]


def mrectToList(mrect):
    res = []
    for rect in mrect:
        res.append(mrectToRect(rect))
    return res


def isBigFace(mrect, num):
    x1 = mrect.left
    y1 = mrect.top
    x2 = mrect.right
    y2 = mrect.bottom
    minSide = min((x2 - x1), (y2 - y1))
    return True if minSide > num else False


