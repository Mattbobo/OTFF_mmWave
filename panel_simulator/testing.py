import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Flatten
from PyQt5 import QtCore, QtWidgets
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import FeatureMapReceiver
import time
import pickle
from PyQt5.QtGui import QFont  # 确保导入 QFont 类

# 加载均值和标准差
with open('normalization_params.pkl', 'rb') as f:
    params = pickle.load(f)
    mean = params['mean']
    std = params['std']

# 手势标签名称
gesture_names = ['FrontWash', 'RearWash', 'ChangeMode', 'VolumeUp', 'VolumeDown','No_gesture']

# 加载模型
model_path = 'C:/Users/matt1/Desktop/OTFF_Project/mmWave_gesture_model/trained_model/gesture_recognition_model.h5'
model = load_model(model_path)


class GestureRecognitionApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        # self.input_shape = (2, 32, 32, 32)  # 设置输入形状为 (channel, height, width, num_frames)
        self.num_frames = 35  # 定义每个样本的帧数
        self.frame_buffer = np.full((self.num_frames, 32, 32, 2), 0)  # 初始化帧缓冲区为 -1
        self.confidence_threshold = 0.95  # 调低置信度阈值以提高检测敏感度

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_gesture)
        self.timer.start(35)  # 每 35 毫秒读取一次数据

    def initUI(self):
        # 创建一个字体对象并设置字体大小
        font = QFont("Arial", 16)  # 设置字体为 Arial，大小为 16

        self.label = QtWidgets.QLabel('Detected gesture: None', self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setGeometry(QtCore.QRect(20, 20, 600, 50))
        self.label.setFont(font)  # 设置字体

        # 添加一个新的 QLabel 用于显示预测的概率
        self.probability_label = QtWidgets.QLabel('Probabilities:', self)
        self.probability_label.setAlignment(QtCore.Qt.AlignCenter)
        self.probability_label.setGeometry(QtCore.QRect(20, 110, 600, 300))
        self.probability_label.setFont(font)  # 设置字体

        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Gesture Recognition')
        self.show()

    def update_gesture(self):
        res = R.getResults()
        if res is None:
            return
        res = np.array(res)  # 将 res 转换为 NumPy 数组
        res = res.transpose((1, 2, 0))  # 32 32 2

        self.frame_buffer = np.roll(self.frame_buffer, -1, axis=0)  # 将帧缓冲区向前滚动
        self.frame_buffer[-1, ...] = res  # 将新帧放入缓冲区

        data = preprocess_data(self.frame_buffer)
        predictions = model.predict(data)
        confidence = np.max(predictions)
        predicted_label = np.argmax(predictions)

        # 将每个手势的概率格式化为百分比并生成字符串
        probabilities_text = "Probabilities:\n"
        for i, prediction in enumerate(predictions[0]):
            probabilities_text += f"{gesture_names[i]}: {prediction * 100:.2f}%\n"

        # 更新概率标签
        self.probability_label.setText(probabilities_text)

        # 只有当最高置信度超过阈值时，才更新检测到的手势
        if confidence > self.confidence_threshold and predicted_label != 5 or confidence > 0.8 and predicted_label == 1:
            gesture_name = gesture_names[predicted_label]
            self.label.setText(f'Detected gesture: {gesture_name}')
            print(f"Detected gesture: {gesture_name} with confidence {confidence}")

            # 重置 frame_buffer 为 -1
            self.frame_buffer = np.full((self.num_frames, 32, 32, 2), 0)
        else:
            # self.label.setText('Detected gesture: No gesture')
            print("No gesture detected.")


# 使用保存的均值和标准差进行标准化
def preprocess_data(data):
    #40 32 32 2
    data = np.transpose(data,(3, 1, 2, 0)) #2 32 32 40
    data = data.reshape((1, 2, 32, 32, 35))
    data = (data - mean) / std
    #2 x 32 x 32 x 500 depth,height,width,frame

    data = np.transpose(data, (0, 4, 2, 3, 1))
    return data


def connect():
    connect = ConnectDevice()
    connect.startUp()  # 连接到设备
    reset = ResetDevice()
    reset.startUp()  # 重置硬件寄存器


def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")  # 设置配置文件夹名称
    ksp = SettingProc()  # 设置过程对象，用于设置硬件 AI 和 RF 以接收数据
    ksp.startUp(SettingConfigs)  # 启动设置过程


def startLoop():
    global R
    R = FeatureMapReceiver(chirps=32)
    R.trigger(chirps=32)
    time.sleep(0.5)
    print("Starting Qt Application")
    app = QtWidgets.QApplication([])
    ex = GestureRecognitionApp()
    app.exec_()


def main():
    kgl.setLib()
    connect()  # 首先需要连接到设备
    startSetting()  # 其次需要设置配置
    startLoop()  # 启动循环以获取数据并识别手势


if __name__ == '__main__':
    main()
