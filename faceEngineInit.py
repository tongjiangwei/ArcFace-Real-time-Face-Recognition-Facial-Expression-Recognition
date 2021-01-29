import ctypes
import lib_func as F
import struct_info as sInfo
import faceEngine

# ================= APPID & SDKKey ==========================
APPID = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
SDKKey = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

# ================== read .so file ===========================
dll_null = ctypes.CDLL(r'./lib/libarcsoft_face.so')
dll = ctypes.CDLL(r'./lib/libarcsoft_face_engine.so')

# ================== face engine parameters setting ==========
MOK = 0  # 函数调用成功状态码
MERR_ASF_ALREADY_ACTIVATED = 90114  # 已激活状态返回码

# 引擎功能code
ASF_DETECT_MODE_VIDEO = 0x00000000  # Video模式，一般用于多帧连续检测
ASF_DETECT_MODE_IMAGE = 0xFFFFFFFF  # Image模式，一般用于静态图的单次检测

ASF_FACE_DETECT = 0x00000001  # 此处detect可以是tracking或者detection两个引擎之一，具体的选择由detect mode 确定
ASF_FACERECOGNITION = 0x00000004  # 人脸特征
ASF_AGE = 0x00000008  # 年龄
ASF_GENDER = 0x00000010  # 性别
ASF_FACE3DANGLE = 0x00000020  # 3D角度
ASF_LIVENESS = 0x00000080  # RGB活体
ASF_IR_LIVENESS = 0x00000400  # 红外活体

# 人脸检测方向
ASF_OP_0_ONLY = 0x1  # 仅检测 0 度
ASF_OP_90_ONLY = 0x2  # 仅检测 90 度
ASF_OP_270_ONLY = 0x3  # 仅检测 270 度
ASF_OP_180_ONLY = 0x4  # 仅检测 180 度
ASF_OP_0_HIGHER_EXT = 0x5  # 全角度检测

# 在线激活
lib_ASFOnlineActivation = dll.ASFOnlineActivation


def ASFOnlineActivation(AppID, SDKey):
    """
    在线激活 SDK
    :param Appkey: 官网获取的 APPID
    :param SDKey: 官网获取的 SDKKEY
    :return: 状态码
    """
    res = lib_ASFOnlineActivation(AppID, SDKey)
    if res != 90114 and res != 0:
        print("激活失败!错误码:{}".format(res))
    return res


def ASFGetActiveFileInfo():
    """
    获取激活文件信息
    :return: 状态码， 激活文件信息
    """
    activeFileInfo = sInfo.ASF_ActiveFileInfo()
    return F.ASFGetActiveFileInfo(ctypes.byref(activeFileInfo)), activeFileInfo



res = ASFOnlineActivation(APPID, SDKKey)
if MOK != res and MERR_ASF_ALREADY_ACTIVATED != res:
    print("ASFActivation fail: {}".format(res))
else:
    print("ASFActivation sucess: {}".format(res))

# 获取激活文件信息
res, activeFileInfo = ASFGetActiveFileInfo()
if res != MOK:
    print('ASFGetActiveFileInf fail: {}'.format(res))
else:
    print(activeFileInfo)

print('ok')

# 实例化人脸引擎
face_engine = faceEngine.ArcFace()
face_engine_2 = faceEngine.ArcFace()

# 需要引擎开启的功能
# mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS
mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE

# Init人脸引擎1
res = face_engine.ASFInitEngine(ASF_DETECT_MODE_VIDEO, ASF_OP_0_ONLY, 30, 10, mask)
if res != MOK:
    print('ASFInitEngine 1 fail: {}'.format(res))
else:
    print('ASFInitEngine 1 success: {}'.format(res))


# Init人脸引擎2
res = face_engine_2.ASFInitEngine(ASF_DETECT_MODE_VIDEO, ASF_OP_0_ONLY, 30, 10, mask)
if res != MOK:
    print('ASFInitEngine 2 fail: {}'.format(res))
else:
    print('ASFInitEngine 2 success: {}'.format(res))