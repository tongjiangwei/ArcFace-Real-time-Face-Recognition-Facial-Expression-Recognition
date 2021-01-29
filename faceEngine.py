import ctypes
import struct_info as sInfo
import lib_func as F


ASVL_PAF_RGB24_B8G8R8 = 0x201  # RGB24位图，bgr格式(opencv 读取图像的默认格式)
ASVL_PAF_GRAY = 0x701          # 灰度图

ASF_GENDER = 0x00000010  # 性别
ASF_FACE3DANGLE = 0x00000020  # 3D角度
ASF_LIVENESS = 0x00000080  # RGB活体
ASF_IR_LIVENESS = 0x00000400  # 红外活体
ASF_FACE_DETECT = 0x00000001  # 此处detect可以是tracking或者detection两个引擎之一，具体的选择由detect mode 确定
ASF_FACERECOGNITION = 0x00000004


class ArcFace():
    def __init__(self):
        self.Handle = ctypes.c_void_p()  # 引擎指针对象

    def ASFInitEngine(self, detectMode:int, detectFaceOrientPriority:int, detectFaceScaleVal:int,
                      detectFaceMaxNum:int, combinedMask:int):
        '''
        :param detectMode: VIDEO/IMAGE模式  VIDEO:连续帧   IMAGE:单张
        :param detectFaceOrientPriority:
        :param detectFaceScaleVal:
        :param detectFaceMaxNum:
        :param combinedMask: 需要启用的功
        :return: 状态码
        '''
        return F.ASFInitEngine(detectMode, detectFaceOrientPriority, detectFaceScaleVal,
                               detectFaceMaxNum, combinedMask, ctypes.byref(self.Handle))

    def ASFDetectFaces(self, frame):
        '''
        face detect
        :param frame: 原始图像：注意：图片宽度必须 为 4 的倍数
        :return: 状态码,人脸检测信息
        '''
        height, width = frame.shape[:2]
        detectedFaces = sInfo.ASF_MultiFaceInfo()
        res = F.ASFDetectFaces(self.Handle, int(width), int(height), ASVL_PAF_RGB24_B8G8R8,
                               frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), ctypes.byref(detectedFaces))
        return res, detectedFaces

    def ASFProcess_IR(self, frame, detectedFaces, processIRMask=ASF_IR_LIVENESS):
        '''
        该接口仅支持单人脸 IR 活体检测
        :param frame: 原始图像：注意：图片宽度必须 为 4 的倍数
        :param detectedFaces: 多人脸检测信息对象
        :param processsIRMask: 当前只有ASF_IR_LIVENESS 一种选择
                            注：检测的属性须在引擎初始化接口的 combinedMask 参数中启用
        :return:
        '''
        height, width = frame.shape[:2]
        res = F.ASFProcess_IR(self.Handle, int(width), int(height), ASVL_PAF_GRAY,
                              frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                              ctypes.byref(detectedFaces), processIRMask)
        return res

    def ASFFaceFeatureExtract(self, frame, singleFaceInfo:sInfo.ASF_SingleFaceInfo):
        """
        人脸特征提取
        :param frame: 原始图像：注意：图片宽度必须 为 4 的倍数
        :param singleFaceInfo: 单个人脸检测框信息
        :return: 状态码,人脸检测信息
        """
        height, width = frame.shape[:2]
        face_feature = sInfo.ASF_FaceFeature()
        # print('facefeature')
        res = F.ASFFaceFeatureExtract(self.Handle, int(width), int(height), ASVL_PAF_RGB24_B8G8R8,
                                      frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                                      singleFaceInfo, ctypes.byref(face_feature))
        # print('res', res)
        copy_face_feature = sInfo.ASF_FaceFeature()
        copy_face_feature.featureSize = face_feature.featureSize
        copy_face_feature.feature = F.malloc(face_feature.featureSize)
        F.memcpy(copy_face_feature.feature, face_feature.feature, face_feature.featureSize)
        return res, copy_face_feature

    def ASFFaceFeatureCompare(self, face_feature1:sInfo.ASF_FaceFeature, face_feature2:sInfo.ASF_FaceFeature):
        """
        人脸特征比较
        :param face_feature1:  特征对象1
        :param face_feature2: 特征对象2
        :return: 状态码，人脸得分
        """
        compare_score = ctypes.c_float()
        ret = F.ASFFaceFeatureCompare(self.Handle, face_feature1, face_feature2, ctypes.byref(compare_score))
        return ret, compare_score.value

    def ASFProcess(self, frame, detectedFaces:sInfo.ASF_MultiFaceInfo, combineMask:int):
        '''
        人脸信息检测（年龄/性别/人脸 3D 角度/rgb活体），最多支持 4 张人脸信息检测
        :param frame: 原始图像：注意：图片宽度必须 为 4 的倍数
        :param detectedFaces: 多人脸检测信息对象
        :param combineMask: 检测的属性（ASF_AGE、ASF_GENDER、ASF_FACE3DANGLE、ASF_LIVENESS），支持多选
                            注：检测的属性须在引擎初始化接口的 combinedMask 参 数中启用
        :return:
        '''
        height, width = frame.shape[:2]
        res = F.ASFProcess(self.Handle, int(width), int(height), ASVL_PAF_RGB24_B8G8R8,
                           frame.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                           ctypes.byref(detectedFaces), combineMask)
        return res

    def ASFGetFace3DAngle(self):
        """
        获取 3D 角度信息
        :return: 状态码，人脸3d角度信息
        """
        angleInfo = sInfo.ASF_Face3DAngle()
        res = F.ASFGetFace3DAngle(self.Handle, angleInfo)
        return res, angleInfo

    def ASFGetAge(self):
        """
        获取年龄信息
        :return:  :状态码，年龄信息
        """
        ageInfo = sInfo.ASF_AgeInfo()
        res = F.ASFGetAge(self.Handle, ctypes.byref(ageInfo))
        return res, ageInfo

    def ASFUninitEngine(self):
        """
        销毁引擎
        :return: 状态码
        """
        return F.ASFUninitEngine(self.Handle)