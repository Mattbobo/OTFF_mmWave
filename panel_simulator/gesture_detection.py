import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PyQt5 import QtCore, QtWidgets
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import FeatureMapReceiver
import time
import pickle

# 加载均值和标准差
with open('normalization_params.pkl', 'rb') as f:
    params = pickle.load(f)
    mean = params['mean']
    std = params['std']

class GestureDetection(QtCore.QObject):
    gesture_detected = QtCore.pyqtSignal(str)  # Signal to emit detected gesture

    def __init__(self, gesture_names, model, R):
        super().__init__()
        self.gesture_names = gesture_names
        self.model = model
        self.R = R
        # self.input_shape = (32, 32, 32, 2)
        self.num_frames = 35
        self.frame_buffer = np.full((self.num_frames, 32, 32, 2), 0)
        self.confidence_threshold = 0.95

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.detect_gesture)
        self.timer.start(35)

    def detect_gesture(self):
        res = self.R.getResults()
        if res is None:
            return
        res = np.array(res)  # 将 res 转换为 NumPy 数组
        # print('before transpose: ' + str(res.shape))  # 将 res.shape 转换为字符串
        res = res.transpose((1, 2, 0)) #32 32 2
        # print('after transpose: ' + str(res.shape))
        # print('buffer shape' + str(self.frame_buffer.shape))

        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)  # 将帧缓冲区向前滚动
        self.frame_buffer[-1, ...] = res  # 将新帧放入缓冲区
        # current_buffer_fill = np.count_nonzero(self.frame_buffer != -1)

        data = preprocess_data(self.frame_buffer)
        predictions = self.model.predict(data)
        confidence = np.max(predictions)
        predicted_label = np.argmax(predictions)

        if confidence > self.confidence_threshold and predicted_label != 5 and predicted_label != 1 or confidence > 0.8 and predicted_label == 1:
            gesture_name = self.gesture_names[predicted_label]
            self.gesture_detected.emit(gesture_name)  # Emit detected gesture signal
            # 重置 frame_buffer 为 -1
            self.frame_buffer = np.full((self.num_frames, 32, 32, 2), 0)
        else:
            # self.gesture_detected.emit("No gesture")  # Emit "no gesture" signal
            print('')
            # self.frame_buffer = np.full((self.num_frames, 32, 32, 2), 0)

# 使用保存的均值和标准差进行标准化
def preprocess_data(data):
    #40 32 32 2
    data = np.transpose(data,(3, 1, 2, 0)) #2 32 32 40
    data = data.reshape((1, 2, 32, 32, 35))
    data = (data - mean) / std
    #2 x 32 x 32 x 500 depth,height,width,frame

    data = np.transpose(data, (0, 4, 2, 3, 1))
    return data