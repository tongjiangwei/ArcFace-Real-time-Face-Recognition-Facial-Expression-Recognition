# ArcFace Real-time Face Recognition && Facial Expression Recognition
A working with real-time, multi-person and multi-threaded face and facial expression recognition.


## Introduction  ##
This is a python multi-threaded framework for real-time, multi-person face and facial expression recognition based on Pytorch and ArcFace SDK v3.0(C++). This repo includes face detecition, face tracking, face fearture extraction and face feature comparison.The input of the USB camera is 30 fps and the outputs can reach **28~30** fps, which meets the standard of real-time detection.In this work, the overall running speed depends on the face detection thread **without being affected by the face recognition process and the expression recognition process**.


## Demos ##
![Image text](https://github.com/crawfordfan/ArcFace-Real-time-Face-Recognition-Facial-Expression-Recognition/blob/main/figures/dehua.gif)
![Image text](https://github.com/crawfordfan/ArcFace-Real-time-Face-Recognition-Facial-Expression-Recognition/blob/main/figures/xue.gif)

## Dependencies ##
- python==3.6
- pytorch==1.7.1+cu110 (for NVIDIA GeForce RTX 3080)
- opencv-python==4.1.0.25
- numpy==1.19.5
- scikit-learn==0.23.2

## Quick Start ##
1. Firstly, get your APPID and SDKKey and fill in "faceEngineInit.py"
  ```
  # ==================== APPID & SDKKey ================
  APPID = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  SDKKey = b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
  ```
2. Download the pre-trained model from https://pan.baidu.com/s/11FX20KJhUpHVTsOyMfjCdA Code:T209 and then put it in the "FER2013_VGG19" folder

3. Download file:libarcsoft_face_engine.so from https://pan.baidu.com/s/1Ibmh9rQ1ch13xpfdWFY_GA Code:T407 and then put it in the "lib" folder

4. Select a face image as the face comparison picture, put it under the "facelibrary/images" folder, and name it with the name corresponding to the face, such as "yourName.jpg"
  ```
  ${POSE_ROOT}
   `-- facelibrary
       |-- images
       |   |-- dehua.jpg
       |   |-- xue.jpg
       |   |-- yourName.jpg
       `-- names.json
        
   ```
5. Add the name in "facelibrary/names.json"
   ```
   ['dehua', 'xue', 'yourName']
   ```
   To load multiple face images, repeat the above operation. If you need to load a large number of face images, you can write a program to batch process

6. Make sure the usb camera is installed correctly and run:
   ```
   python main.py
   ```

## Train your own models ##
- You can refer to https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch to train your own models and put them in the "FER2013_VGG19" folder
- Dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
Image Properties: 48 x 48 pixels (2304 bytes)
labels: 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
The training set consists of 28,709 examples. The public test set consists of 3,589 examples. The private test set consists of another 3,589 examples.

## About the author ##
- If our work is helpful to you, please give us a star
- Authors: Yifan Li, Xiaoyi Xia, Yue Wang, Ruyi Liu, Han Zhang, Fudi Geng
