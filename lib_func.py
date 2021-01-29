import ctypes
from ctypes import *
from struct_info import *


dll_null = ctypes.CDLL(r'./lib/libarcsoft_face.so')
dll = ctypes.CDLL(r'./lib/libarcsoft_face_engine.so')
dllc = ctypes.CDLL('libc.so.6')


# 激活
ASFOnlineActivation = dll.ASFOnlineActivation
ASFOnlineActivation.restype = c_int32
ASFOnlineActivation.argtypes = (c_char_p, c_char_p)

# 获取激活文件信息
ASFGetActiveFileInfo = dll.ASFGetActiveFileInfo
ASFGetActiveFileInfo.restype = c_int32
ASFGetActiveFileInfo.argtypes = (POINTER(ASF_ActiveFileInfo), )

# 初始化
ASFInitEngine = dll.ASFInitEngine
ASFInitEngine.restype = c_int32
ASFInitEngine.argtypes = (c_long, c_int32, c_int32, c_int32, c_int32, POINTER(c_void_p))

# 人脸识别
ASFDetectFaces = dll.ASFDetectFaces
ASFDetectFaces.restype = c_int32
ASFDetectFaces.argtypes = (c_void_p, c_int32, c_int32, c_int32,
                           POINTER(c_ubyte), POINTER(ASF_MultiFaceInfo))

# 特征提取
ASFFaceFeatureExtract = dll.ASFFaceFeatureExtract
ASFFaceFeatureExtract.restype = c_int32
ASFFaceFeatureExtract.argtypes = (c_void_p, c_int32, c_int32, c_int32, POINTER(c_ubyte),
                                  POINTER(ASF_SingleFaceInfo), POINTER(ASF_FaceFeature))

# 特征比对
ASFFaceFeatureCompare = dll.ASFFaceFeatureCompare
ASFFaceFeatureCompare.restype = c_int32
ASFFaceFeatureCompare.argtypes = (c_void_p, POINTER(ASF_FaceFeature), POINTER(ASF_FaceFeature),
                                  POINTER(c_float))

# 获取3d角度信息
ASFGetFace3DAngle = dll.ASFGetFace3DAngle
ASFGetFace3DAngle.restype = c_int32
ASFGetFace3DAngle.argtypes = (c_void_p, POINTER(ASF_Face3DAngle))

# 获取年龄
ASFGetAge = dll.ASFGetAge
ASFGetAge.restype = c_int32
ASFGetAge.argtypes = (c_void_p, POINTER(ASF_AgeInfo))

# 销毁引擎
ASFUninitEngine = dll.ASFUninitEngine
ASFUninitEngine.argtypes = (c_void_p,)


malloc = dllc.malloc
malloc.restype = c_void_p
malloc.argtypes = (ctypes.c_void_p, )

free = dllc.free
free.restype = None
free.argtypes = (c_void_p, )

memcpy = dllc.memcpy
memcpy.restype = c_void_p
memcpy.argtypes = (c_void_p, c_void_p, c_size_t)

