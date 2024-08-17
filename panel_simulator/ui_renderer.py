import sys
import pygame
from PyQt5 import QtWidgets, QtCore
from pygame.locals import *

class UIRenderer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption("Bidet Control Panel Simulation")
        self.font = pygame.font.Font("wt009.ttf", 24)  # 確保字體路徑正確
        self.mode_font = pygame.font.Font("wt009.ttf", 36)  # 模式字體大小

        # 狀態變量
        self.current_gesture = "No gesture"
        self.modes = ["水量", "水溫", "座溫"]
        self.current_mode = 0
        self.scales = [1, 1, 1]  # 各模式的刻度值
        self.keyboard_mode = False  # 指示是否啟用鍵盤控制模式

        # 前後洗淨指示燈的狀態與計時器
        self.front_indicator_color = (255, 0, 0)  # 紅色
        self.rear_indicator_color = (255, 0, 0)   # 紅色
        self.front_timer = None
        self.rear_timer = None

        # 按鈕亮燈效果的顏色與計時器
        self.front_button_color = (255, 0, 0)  # 初始顏色
        self.rear_button_color = (255, 165, 0)  # 初始顏色
        self.front_button_timer = None
        self.rear_button_timer = None

        # 左右調整刻度按鈕亮燈效果的顏色與計時器
        self.left_button_color = (128, 128, 128)  # 初始灰色
        self.right_button_color = (128, 128, 128)  # 初始灰色
        self.left_button_timer = None
        self.right_button_timer = None

        # 啟動定時器來週期性地更新畫面
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_pygame)
        self.timer.start(50)  # 每50毫秒更新一次畫面

    def initUI(self):
        self.setGeometry(300, 300, 600, 400)
        self.setWindowTitle('Bidet Control Panel Simulation')
        self.show()

    @QtCore.pyqtSlot(str)
    def update_gesture(self, gesture_name):
        if not self.keyboard_mode:  # 只有在非鍵盤模式下才會更新手勢
            self.current_gesture = gesture_name
            self.process_gesture(gesture_name)

    def process_gesture(self, gesture_name):
        if gesture_name == 'FrontWash':  # 前洗淨手勢
            if self.front_timer and self.front_timer.isActive():
                self.front_timer.stop()  # 偵測到手勢時停止並重置計時器
            self.front_indicator_color = (0, 255, 0)  # 綠色
            self.front_timer = QtCore.QTimer(self)
            self.front_timer.setSingleShot(True)
            self.front_timer.timeout.connect(self.reset_front_indicator)
            self.front_timer.start(5000)  # 5秒後恢復紅色

            # 觸發前洗淨按鈕亮燈效果
            self.front_button_color = (255, 100, 100)  # 亮紅色
            self.front_button_timer = QtCore.QTimer(self)
            self.front_button_timer.setSingleShot(True)
            self.front_button_timer.timeout.connect(self.reset_front_button_color)
            self.front_button_timer.start(200)  # 0.2秒後恢復原色

        elif gesture_name == 'RearWash':  # 後洗淨手勢
            if self.rear_timer and self.rear_timer.isActive():
                self.rear_timer.stop()  # 偵測到手勢時停止並重置計時器
            self.rear_indicator_color = (0, 255, 0)  # 綠色
            self.rear_timer = QtCore.QTimer(self)
            self.rear_timer.setSingleShot(True)
            self.rear_timer.timeout.connect(self.reset_rear_indicator)
            self.rear_timer.start(5000)  # 5秒後恢復紅色

            # 觸發後洗淨按鈕亮燈效果
            self.rear_button_color = (255, 200, 100)  # 亮橙色
            self.rear_button_timer = QtCore.QTimer(self)
            self.rear_button_timer.setSingleShot(True)
            self.rear_button_timer.timeout.connect(self.reset_rear_button_color)
            self.rear_button_timer.start(200)  # 0.2秒後恢復原色

        elif gesture_name == 'ChangeMode':
            self.current_mode = (self.current_mode + 1) % len(self.modes)  # 切換模式
        elif gesture_name == 'VolumeUp':
            self.scales[self.current_mode] = min(self.scales[self.current_mode] + 1, 5)  # 增加刻度

            # 觸發右按鈕亮燈效果
            self.right_button_color = (200, 200, 200)  # 亮灰色
            self.right_button_timer = QtCore.QTimer(self)
            self.right_button_timer.setSingleShot(True)
            self.right_button_timer.timeout.connect(self.reset_right_button_color)
            self.right_button_timer.start(200)  # 0.2秒後恢復原色

        elif gesture_name == 'VolumeDown':
            self.scales[self.current_mode] = max(self.scales[self.current_mode] - 1, 1)  # 減少刻度

            # 觸發左按鈕亮燈效果
            self.left_button_color = (200, 200, 200)  # 亮灰色
            self.left_button_timer = QtCore.QTimer(self)
            self.left_button_timer.setSingleShot(True)
            self.left_button_timer.timeout.connect(self.reset_left_button_color)
            self.left_button_timer.start(200)  # 0.2秒後恢復原色

    def reset_front_indicator(self):
        self.front_indicator_color = (255, 0, 0)  # 恢復紅色

    def reset_rear_indicator(self):
        self.rear_indicator_color = (255, 0, 0)  # 恢復紅色

    def reset_front_button_color(self):
        self.front_button_color = (255, 0, 0)  # 恢復原紅色

    def reset_rear_button_color(self):
        self.rear_button_color = (255, 165, 0)  # 恢復原橙色

    def reset_left_button_color(self):
        self.left_button_color = (128, 128, 128)  # 恢復原灰色

    def reset_right_button_color(self):
        self.right_button_color = (128, 128, 128)  # 恢復原灰色
    #
    # def handle_keyboard_input(self, key):
    #     if key == K_q:
    #         self.current_gesture = 'FrontWash'  # 模擬前洗淨手勢
    #         self.process_gesture('FrontWash')
    #     elif key == K_w:
    #         self.current_gesture = 'RearWash'  # 模擬後洗淨手勢
    #         self.process_gesture('RearWash')
    #     elif key == K_e:
    #         self.current_gesture = 'ChangeMode'  # 模擬模式切換手勢
    #         self.process_gesture('ChangeMode')
    #     elif key == K_LEFT:
    #         self.current_gesture = 'VolumeDown'  # 模擬減少刻度手勢
    #         self.process_gesture('VolumeDown')
    #     elif key == K_RIGHT:
    #         self.current_gesture = 'VolumeUp'  # 模擬增加刻度手勢
    #         self.process_gesture('VolumeUp')

    def update_pygame(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            # elif event.type == KEYDOWN:
            #     if event.key == K_o:
            #         self.keyboard_mode = not self.keyboard_mode  # 切換鍵盤模式
            #     if self.keyboard_mode:
            #         self.handle_keyboard_input(event.key)  # 處理鍵盤輸入

        # 設定背景顏色
        self.screen.fill((230, 230, 230))  # 淺灰色背景

        # 顯示左上角的模式（手勢偵測模式或鍵盤操作模式）
        # mode_text = "鍵盤操作模式" if self.keyboard_mode else "手勢偵測模式"
        # mode_text_surface = self.font.render(mode_text, True, (0, 0, 0))
        # self.screen.blit(mode_text_surface, (10, 10))

        # 繪製按鈕與邊框
        pygame.draw.circle(self.screen, (0, 0, 0), (150, 200), 55)  # 前洗淨按鈕的黑色邊框
        pygame.draw.circle(self.screen, self.front_button_color, (150, 200), 50)  # 前洗淨按鈕

        pygame.draw.circle(self.screen, (0, 0, 0), (300, 200), 55)  # 後洗淨按鈕的黑色邊框
        pygame.draw.circle(self.screen, self.rear_button_color, (300, 200), 50)  # 後洗淨按鈕

        front_text_surface = self.font.render('前洗淨', True, (0, 0, 0))
        rear_text_surface = self.font.render('後洗淨', True, (0, 0, 0))
        self.screen.blit(front_text_surface, (110, 190))
        self.screen.blit(rear_text_surface, (260, 190))

        # 在洗淨按鈕下方繪製小指示燈
        pygame.draw.circle(self.screen, self.front_indicator_color, (150, 290), 15)  # 前洗淨指示燈
        pygame.draw.circle(self.screen, self.rear_indicator_color, (300, 290), 15)   # 後洗淨指示燈

        # 顯示當前手勢
        gesture_text_surface = self.font.render(f"手势: {self.current_gesture}", True, (0, 0, 0))
        self.screen.blit(gesture_text_surface, (10, 360))

        # 顯示"模式"文字在框框外
        mode_label_surface = self.font.render("模式", True, (0, 0, 0))
        self.screen.blit(mode_label_surface, (400, 120))  # 框框上方的位置

        # 框框內顯示模式名稱
        pygame.draw.rect(self.screen, (173, 216, 230), (400, 160, 150, 75))  # 淺藍色背景的框框
        pygame.draw.rect(self.screen, (0, 0, 0), (400, 160, 150, 75), 2)  # 框框邊框
        mode_name_surface = self.mode_font.render(self.modes[self.current_mode], True, (0, 0, 0))
        self.screen.blit(mode_name_surface, (435, 180))  # 顯示模式名稱

        # 顯示與當前模式對應的刻度，以色塊表示
        current_scale = self.scales[self.current_mode]
        for i in range(5):
            color = (0, 128, 0) if i < current_scale else (211, 211, 211)  # 綠色表示填滿，淺灰色表示空
            pygame.draw.rect(self.screen, color, (400 + i * 30, 250, 25, 25))  # 刻度塊

        # 在刻度下方繪製左右按鈕（左右三角形）
        pygame.draw.polygon(self.screen, self.left_button_color, [(375, 290), (395, 280), (395, 300)])  # 左三角形
        pygame.draw.polygon(self.screen, self.right_button_color, [(570, 290), (550, 280), (550, 300)])  # 右三角形

        pygame.display.flip()

