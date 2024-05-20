from PySide6 import QtWidgets
from PySide6 import QtGui
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtCore import QObject, QThread, Signal, Slot, Qt
from ui_main_window import Ui_MainWindow
from ui_change_zone import Ui_Zone_changing
from personDetector import Detector
import numpy as np
import sys
import cv2
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import socket
import struct
import time


class VideoThread(QThread):
    change_pixmap_signal_1 = Signal(np.ndarray)
    change_pixmap_signal_2 = Signal(np.ndarray)

    def __init__(self, detector_1: Detector, detector_2: Detector, cap_1, cap_2, is_active_detector_1: bool, is_active_detector_2: bool, server_ip='192.168.0.105', server_port_1=8000, server_port_2=8001):
        super().__init__()
        self.detector_1 = detector_1
        self.detector_2 = detector_2
        self.cap_1 = cap_1
        self.cap_2 = cap_2
        self.is_active_detector_1 = is_active_detector_1
        self.is_active_detector_2 = is_active_detector_2
        self.server_ip = server_ip
        self.server_port_1 = server_port_1
        self.server_port_2 = server_port_2
        self.client_socket_1 = None
        self.client_socket_2 = None
        self.server_socket_1 = None
        self.server_socket_2 = None
        self.client_connected_1 = False
        self.client_connected_2 = False

    def set_active_detector_1(self, is_active: bool) -> None:
        self.is_active_detector_1 = is_active

    def set_active_detector_2(self, is_active: bool) -> None:
        self.is_active_detector_2 = is_active

    def run(self):
        self.server_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket_1.bind((self.server_ip, self.server_port_1))
        self.server_socket_1.listen(5)
        self.server_socket_1.settimeout(1)

        self.server_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket_2.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket_2.bind((self.server_ip, self.server_port_2))
        self.server_socket_2.listen(5)
        self.server_socket_2.settimeout(1)

        print(f"Listening for connections on {self.server_ip}:{self.server_port_1} and {self.server_ip}:{self.server_port_2}")

        while True:
            # Accept connections for the first stream
            try:
                self.client_socket_1, _ = self.server_socket_1.accept()
                self.client_connected_1 = True
                print("Client 1 connected")
            except socket.timeout:
                pass

            # Accept connections for the second stream
            try:
                self.client_socket_2, _ = self.server_socket_2.accept()
                self.client_connected_2 = True
                print("Client 2 connected")
            except socket.timeout:
                pass

            # Read and process frame from the first stream
            _, frame_1 = self.cap_1.read()
            frame_1 = cv2.rotate(frame_1, cv2.ROTATE_90_CLOCKWISE)
            if not self.is_active_detector_1:
                frame_1 = cv2.resize(frame_1, (1280, 720))
            else:
                _, frame_1 = self.detector_1.detect(frame_1)

            self.change_pixmap_signal_1.emit(frame_1)

            # Send the first stream to the client if connected
            if self.client_connected_1:
                try:
                    self.send_frame(self.client_socket_1, frame_1)
                except (BrokenPipeError, ConnectionResetError):
                    self.client_socket_1.close()
                    self.client_connected_1 = False
                    print("Client 1 disconnected")

            # Read and process frame from the second stream
                    _, frame_2 = self.cap_2.read()
                    frame_2 = cv2.rotate(frame_2, cv2.ROTATE_90_CLOCKWISE)
                    if not self.is_active_detector_2:
                        frame_2 = cv2.resize(frame_2, (1280, 720))
                    else:
                        _, frame_2 = self.detector_2.detect(frame_2)

                    self.change_pixmap_signal_2.emit(frame_2)

                    # Send the second stream to the client if connected
                    if self.client_connected_2:
                        try:
                            self.send_frame(self.client_socket_2, frame_2)
                        except (BrokenPipeError, ConnectionResetError):
                            self.client_socket_2.close()
                            self.client_connected_2 = False
                            print("Client 2 disconnected")

            def send_frame(self, client_socket, frame):
                _, buffer = cv2.imencode('.jpg', frame)
                data = buffer.tobytes()
                size = len(data)
                client_socket.sendall(struct.pack(">L", size) + data)

class HumanDetectorDesktopApp(QMainWindow):
    def __init__(self) -> None:
        super(HumanDetectorDesktopApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Initialize two detectors
        self.detector_1 = Detector(
            resolution=(1280, 720),
            zone=np.array([1000, 0, 500, 0, 500, 720, 1000, 720], dtype=int).reshape((4, 2))
        )
        self.detector_2 = Detector(
            resolution=(1280, 720),
            zone=np.array([1000, 0, 500, 0, 500, 720, 1000, 720], dtype=int).reshape((4, 2))
        )

        # Initialize two video captures via UDP
        self.cap_1 = cv2.VideoCapture('udp://192.168.0.105:40002', cv2.CAP_FFMPEG)
        self.cap_2 = cv2.VideoCapture('udp://192.168.0.105:40003', cv2.CAP_FFMPEG)

        # Waiting for the first video stream
        while not self.cap_1.isOpened():
            self.cap_1 = cv2.VideoCapture('udp://192.168.0.105:40002', cv2.CAP_FFMPEG)
            print("Waiting for video stream 1...")
            time.sleep(3)

        print("Video stream 1 received!")

        # Waiting for the second video stream
        while not self.cap_2.isOpened():
            self.cap_2 = cv2.VideoCapture('udp://192.168.0.105:40003', cv2.CAP_FFMPEG)
            print("Waiting for video stream 2...")
            time.sleep(3)

        print("Video stream 2 received!")

        self.sender_email = None

        self.ui.settings.triggered.connect(self.open_settings_window)

        self.video_stream_1 = self.ui.video_stream_1
        self.video_stream_2 = self.ui.video_stream_2

        self.is_active_detector_1 = True
        self.is_active_detector_2 = True

        self.ui.activate_people_detector_1.clicked.connect(self.activate_detector_button_1_clicked)
        self.ui.activate_people_detector_2.clicked.connect(self.activate_detector_button_2_clicked)

        self.thread = VideoThread(
            self.detector_1, self.detector_2,
            self.cap_1, self.cap_2,
            self.is_active_detector_1, self.is_active_detector_2
        )

        self.thread.change_pixmap_signal_1.connect(self.update_image_1)
        self.thread.change_pixmap_signal_2.connect(self.update_image_2)
        self.thread.start()

    def open_settings_window(self):
        self.new_window = QtWidgets.QDialog()
        self.ui_settings_window = Ui_Zone_changing()
        self.ui_settings_window.setupUi(self.new_window)
        self.ui_settings_window.btn_save_zone.clicked.connect(self.save_new_cords)

        self.ui_settings_window.btn_save_resolution.clicked.connect(self.save_new_resolution)
        self.ui_settings_window.btn_save_reciever.clicked.connect(self.save_reciever)
        self.ui_settings_window.btn_save_sender.clicked.connect(self.save_sender)

        self.new_window.show()

    def save_reciever(self):
        self.reciever_email = self.ui_settings_window.le_reciever_email.text()

    def save_sender(self):
        self.sender_email = self.ui_settings_window.le_sender_email.text()
        self.sender_pass = self.ui_settings_window.le_sender_pass.text()

    def save_new_resolution(self):
        self.resolution = map(int, self.ui_settings_window.le_resolution.text().split())

    def save_new_cords(self):
        cords = list(map(int, self.ui_settings_window.le_right_top_cords.text().split()))
        cords.extend(list(map(int, self.ui_settings_window.le_left_top_cords.text().split())))
        cords.extend(list(map(int, self.ui_settings_window.le_left_bottom_cords.text().split())))
        cords.extend(list(map(int, self.ui_settings_window.le_right_bottom_cords.text().split())))
        self.detector_1.change_zone(np.array(cords, dtype=int).reshape((4, 2)))
        self.detector_2.change_zone(np.array(cords, dtype=int).reshape((4, 2)))

    def activate_detector_button_1_clicked(self):
        if self.is_active_detector_1:
            self.is_active_detector_1 = False
            self.ui.activate_people_detector_1.setText('Выключить\nраспознавание людей\nна видео')
        else:
            self.is_active_detector_1 = True
            self.ui.activate_people_detector_1.setText('Включить\nраспознавание людей\nна видео')

        self.thread.set_active_detector_1(self.is_active_detector_1)

    def activate_detector_button_2_clicked(self):
        if self.is_active_detector_2:
            self.is_active_detector_2 = False
            self.ui.activate_people_detector_2.setText('Выключить\nраспознавание людей\nна видео')
        else:
            self.is_active_detector_2 = True
            self.ui.activate_people_detector_2.setText('Включить\nраспознавание людей\nна видео')

        self.thread.set_active_detector_2(self.is_active_detector_2)

    @Slot(np.ndarray)
    def update_image_1(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_stream_1.setPixmap(qt_img)

    @Slot(np.ndarray)
    def update_image_2(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_stream_2.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1280, 720, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = HumanDetectorDesktopApp()
    window.show()
    sys.exit(app.exec())