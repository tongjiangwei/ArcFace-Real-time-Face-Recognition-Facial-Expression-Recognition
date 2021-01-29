import os
import cv2
import time
import json
import utils
import torch
import faceEngineInit
import numpy as np
import struct_info as sInfo
import transforms as transforms
from models import VGG
from PIL import Image, ImageFont
from skimage.transform import resize
from multiprocessing.dummy import Queue

# ================== FACE_ENGINE INIT ===========================
face_engine = faceEngineInit.face_engine
face_engine_2 = faceEngineInit.face_engine_2
MOK = faceEngineInit.MOK

# =================== READ CAMERA ==============
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)  # ~30

# =================== QUEUE ====================
frame_buffer_size = 100
max_face_number = 20
max_frame_rate = 25
frame_queue = Queue(frame_buffer_size)
upstream_queue = Queue(frame_buffer_size)
detected_face_queue = Queue()
result_queue = Queue(max_face_number)
expression_queue = Queue()
expression_faceID_queue = Queue()

# ================= DEFINITION =====================
similarityThreshold = 0.7
currentFaceID = []
faceIdToName = {}
faceLib = {}
expressionDict = {}

# ===================== LOAD FER MODEL =================
net = VGG('VGG19')
checkpoint = torch.load('FER2013_VGG19/expression_recognition_model.t7')
net.load_state_dict(checkpoint['net'])
net.cuda()
net.eval()

# =================== RECORD START TIME ================
stat_time = time.time()

# =================== FER INPUT_SIZE ADJUSTMENT ===========
cut_size = 44
transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

# ================== EMOTION CATEGORY ================
EMOTIONS = ['生气', '厌恶', '害怕', '开心', '难过', '惊讶', '平静']

# ================== DISPLAY FONT =================
fontpath = "./platech.ttf"
ft = ImageFont.truetype(fontpath, 15, encoding="utf-8")


def logfacelib():
    print('开始加载人脸库')
    start_time = time.time()
    with open('facelibrary/names.json', 'r') as f:
        names = json.load(f)
    imgdir = 'facelibrary/images'
    global faceLib
    for name in names:
        imgfile = os.path.join(imgdir, '{}.jpg'.format(name))
        img = cv2.imread(imgfile)
        res, detectedFaces = face_engine.ASFDetectFaces(img)
        if res == MOK:
            single_detected_face = sInfo.ASF_SingleFaceInfo()
            single_detected_face.faceRect = detectedFaces.faceRect[0]
            single_detected_face.faceOrient = detectedFaces.faceOrient[0]
            res, face_feature = face_engine.ASFFaceFeatureExtract(img, single_detected_face)
            faceLib[name] = face_feature
            if res != MOK:
                print('ASFFaceFeatureExtract {} fail: {}'.format(name, res))
        else:
            print('ASFDetectFaces {} fail: {}'.format(name, res))
    print('人脸库信息：', faceLib)
    totaltime = time.time() - start_time
    print('人脸库加载完毕！耗时 {}s'.format(totaltime))


def videoplay_loop():
    while True:
        faceDetectInfo = upstream_queue.get()
        frame, faceRectList, faceOrientList, faceIDList, faceNum = \
            faceDetectInfo[0], faceDetectInfo[1], faceDetectInfo[2], faceDetectInfo[3], faceDetectInfo[4]
        for i in range(faceNum):
            singleFaceInfo = sInfo.ASF_SingleFaceInfo()
            singleFaceInfo.faceRect = faceRectList[i]
            singleFaceInfo.faceOrient = faceOrientList[i]
            nameLabel = ' '
            if faceIdToName and faceIDList[i] in faceIdToName:
                nameLabel = faceIdToName[faceIDList[i]]
            expressLabel = ' '
            if expressionDict and faceIDList[i] in expressionDict:
                expressLabel = expressionDict[faceIDList[i]]
            # print("expressLabel",expressLabel)
            box = singleFaceInfo.faceRect
            # print('box', box)
            frame = utils.draw(frame, nameLabel, expressLabel, box)

        cv2.imshow('cap', frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord(" "):
            cv2.waitKey(0)
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def recognition_loop():
    # =================== compare face ===================
    while True:
        faceDetectInfo, newFaceID = detected_face_queue.get()
        frame, faceRectList, faceOrientList, faceIDList, faceNum = \
            faceDetectInfo[0], faceDetectInfo[1], faceDetectInfo[2], faceDetectInfo[3], faceDetectInfo[4]

        for id in newFaceID:
            index = faceIDList.index(id)
            singleFaceInfo = sInfo.ASF_SingleFaceInfo()
            singleFaceInfo.faceRect = faceRectList[index]
            singleFaceInfo.faceOrient = faceOrientList[index]

            newPersonName = 'xxx'

            res, face_feature = face_engine_2.ASFFaceFeatureExtract(frame, singleFaceInfo)

            # ============ compare with the faces in the facelib =====================
            for name, face_feature_compare in faceLib.items():
                res, score = face_engine_2.ASFFaceFeatureCompare(face_feature, face_feature_compare)
                if score >= similarityThreshold:
                    newPersonName = name
                    break

            global faceIdToName
            faceIdToName[id] = newPersonName
            # print(faceIdToName)

        # delete expired faceID
        for k in list(faceIdToName.keys()):
            if k not in faceIDList:
                faceIdToName.pop(k)


def detection_loop():
    # =================== detect face ====================
    def faceDetected(frame):
        res, detectedFaces = face_engine.ASFDetectFaces(frame)
        if res == MOK:
            multi_detected_face = sInfo.ASF_MultiFaceInfo()
            multi_detected_face.faceNum = detectedFaces.faceNum
            multi_detected_face.faceRect = detectedFaces.faceRect
            multi_detected_face.faceOrient = detectedFaces.faceOrient
            multi_detected_face.faceID = detectedFaces.faceID
            faceRectList = [multi_detected_face.faceRect[i] for i in range(multi_detected_face.faceNum)]  # multi-face list
            faceIDList = [multi_detected_face.faceID[i] for i in range(multi_detected_face.faceNum)]
            faceOrientList = [multi_detected_face.faceOrient[i] for i in range(multi_detected_face.faceNum)]
            faceNum = multi_detected_face.faceNum
            return faceRectList, faceOrientList, faceIDList, faceNum
        else:
            print('ASFDetectFaces failed, error code is: ', res)

    count = 0
    while True:
        frame = frame_queue.get()
        count += 1
        faceRectList, faceOrientList, faceIDList, faceNum = faceDetected(frame)
        faceDetectInfo = [frame, faceRectList, faceOrientList, faceIDList, faceNum]
        print('DetectInfo', faceRectList, faceOrientList, faceIDList, faceNum)
        upstream_queue.put(faceDetectInfo)
        if expression_queue.empty():
            print('expression_queue is empty')
            expression_queue.put((frame, faceRectList, faceIDList, faceNum))

        newFaceID = []
        global currentFaceID
        # print(currentFaceID)
        for i in range(faceNum):
            if len(currentFaceID) == 0 or faceIDList[i] > currentFaceID[-1]:
                # ========= the frame has a new face which is needed to be recognized =======
                if utils.isBigFace(faceRectList[i], 25):
                    newFaceID.append(faceIDList[i])
        if count == 60:
            newFaceID = faceIDList
            count = 0
            global faceIdToName
            faceIdToName = {}
            print('========================= 人脸识别刷新 =========================', count, faceIdToName)
        currentFaceID = faceIDList
        if len(newFaceID) != 0:
            # print('put recognition')
            detected_face_queue.put((faceDetectInfo, newFaceID))


def expression_loop():

    while True:
        frame, faceRectList, faceIDList, faceNum = expression_queue.get()
        print('faceRectList', faceRectList)
        print('faceIDList', faceIDList)
        rectList = utils.mrectToList(faceRectList)
        height, width = frame.shape[0], frame.shape[1]
        for i in range(faceNum):
            face = rectList[i]
            faceID = faceIDList[i]
            x1, y1, x2, y2 = max(0, face[0]), max(0, face[1]), min(width - 1, face[2]), min(height - 1, face[3])
            if min((y2-y1), (x2-x1)) > 25:
                raw_img = frame[y1:y2, x1:x2]
                gray = np.dot(raw_img[..., :3], [0.299, 0.587, 0.114])
                gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)
                img = gray[:, :, np.newaxis]
                img = np.concatenate((img, img, img), axis=2)
                img = Image.fromarray(img)
                inputs = transform_test(img)
                ncrops, c, h, w = np.shape(inputs)
                inputs = inputs.view(-1, c, h, w)
                inputs = inputs.cuda()
                with torch.no_grad():
                    outputs = net(inputs)
                    outputs_avg = outputs.view(ncrops, -1).mean(0)
                    _, predicted = torch.max(outputs_avg.data, 0)
                    label = EMOTIONS[int(predicted.cpu().numpy())]
                global expressionDict
                expressionDict[faceID] = label
        # delete expired faceID
        for k in list(expressionDict.keys()):
            if k not in faceIDList:
                expressionDict.pop(k)


def camera_loop(cap):
    # rate = 1 / 28
    rate = 1 / fps
    while True:
        start_time = time.time()
        flag, frame_source = cap.read()
        if flag:
            frame = utils.adjustHWTo4(frame_source)
            frame_queue.put(frame)
        res_time = rate - (time.time() - start_time)
        if res_time > 0:
            time.sleep(res_time)
