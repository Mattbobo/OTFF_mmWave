import sys
from PyQt5 import QtWidgets
from gesture_detection import GestureDetection
from ui_renderer import UIRenderer

# 保留原有的所有库和全局变量的初始化
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from KKT_Module.ksoc_global import kgl
from KKT_Module.Configs import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc, ConnectDevice, ResetDevice
from KKT_Module.DataReceive.DataReciever import FeatureMapReceiver
import time

gesture_names = ['FrontWash', 'RearWash', 'ChangeMode', 'VolumeUp', 'VolumeDown', 'No_gesture']
model_path = 'C:/Users/matt1/Desktop/OTFF_Project/mmWave_gesture_model/trained_model/gesture_recognition_model.h5'
model = load_model(model_path)

def connect():
    connect = ConnectDevice()
    connect.startUp()
    reset = ResetDevice()
    reset.startUp()

def startSetting():
    SettingConfigs.setScriptDir("K60168-Test-00256-008-v0.0.8-20230717_60cm")
    ksp = SettingProc()
    ksp.startUp(SettingConfigs)

def main():
    kgl.setLib()
    connect()
    startSetting()

    app = QtWidgets.QApplication(sys.argv)

    # Initialize UI Renderer
    ui_renderer = UIRenderer()

    # Initialize Gesture Detection
    global R
    R = FeatureMapReceiver(chirps=32)
    R.trigger(chirps=32)
    gesture_detection = GestureDetection(gesture_names, model, R)
    gesture_detection.gesture_detected.connect(ui_renderer.update_gesture)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
