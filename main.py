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
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, detector: Detector, cap, is_active_detector: bool, server_ip='192.168.0.105', server_port=8000):
        super().__init__()
        self.detector = detector
        self.cap = cap
        self.is_active_detector = is_active_detector
        self.server_ip = server_ip
        self.server_port = server_port
        self.reciever_email = None
        self.sender = None
        self.server_socket = None
        self.client_socket = None
        self.client_connected = False

    def set_active_detector(self, is_active: bool) -> None:
        '''
        Set is_active_detector to is_active value
        '''
        self.is_active_detector = is_active

    def run(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.server_ip, self.server_port))
        self.server_socket.listen(5)
        self.server_socket.settimeout(1)  # Set timeout for accept()

        print(f"Listening for connection on {self.server_ip}:{self.server_port}")

        last_send = 0
        while True:
            try:
                self.client_socket, _ = self.server_socket.accept()
                self.client_connected = True
                print("Client connected")
            except socket.timeout:
                pass  # No client connected within the timeout period

            _, frame = self.cap.read()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if not self.is_active_detector:
                frame = cv2.resize(frame, (1280, 720))
            else:
                was_human_detected_in_zone, frame = self.detector.detect(frame)
                # if was_human_detected_in_zone:
                    # curr_time = time.time()
                    # if curr_time - last_send > 60 * 5 and
                    # add sending email and push notification to mobile app
            # print(self.is_active_detector)
            self.change_pixmap_signal.emit(frame)

            if self.client_connected:
                try:
                    self.send_frame(frame)
                except (BrokenPipeError, ConnectionResetError):
                    self.client_socket.close()
                    self.client_connected = False
                    print("Client disconnected")

    def send_frame(self, frame):
        _, buffer = cv2.imencode('.jpg', frame)
        data = buffer.tobytes()
        size = len(data)
        self.client_socket.sendall(struct.pack(">L", size) + data)


class HumanDetectorDesktopApp(QMainWindow):
    def __init__(self) -> None:
        '''
        args:
                video_stream_path: path to video stream, default = 0 (video stream from web camera)
        '''
        super(HumanDetectorDesktopApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.detector = Detector(resolution=(1280, 720), zone=np.array([1000, 0, 500, 0, 500, 720, 1000, 720], dtype=int).reshape((4, 2)))
        # Получаем видео через UDP
        self.cap = cv2.VideoCapture('udp://192.168.0.105:40002', cv2.CAP_FFMPEG)

        # Ожидание получения видеопотока
        while not self.cap.isOpened():
            self.cap = cv2.VideoCapture('udp://192.168.0.105:40002', cv2.CAP_FFMPEG)
            print("Ожидание получения видеопотока...")
            time.sleep(3)  # Задержка в 5 секунд

        print("Видеопоток получен!")

        self.sender_email = None

        self.ui.settings.triggered.connect(self.open_settings_window)

        self.video_stream = self.ui.video_stream
        self.is_active_detector = True

        self.ui.activate_people_detector.clicked.connect(self.activate_detector_button_clicked)

        self.thread = VideoThread(self.detector, self.cap, self.is_active_detector)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def log_in(self):
        '''
        connects to email server

        return: touple(sender email, smtp server)
        '''
        server = smtplib.SMTP_SSL("smtp.gmail.com",465)
        server.login(self.sender_email, self.sender_pass)# login to email
        return self.sender_email, server

    def send_email(self, to_email: str, image_path: str, from_email: str, server: smtplib.SMTP_SSL) -> None:
        '''
        sends email with text and image

        args:
            to email: reciever email
            image_path: path to image
            from email: sender email
            server: smtp server
        return:
            none
        '''
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = ""
        # Add in the message body
        message_body = ""
        # add image
        message.attach(MIMEText(message_body, 'plain'))
        with open(image_path, 'rb') as file:
            image = MIMEImage(file.read())
            image.add_header('Content-Disposition', 'attachment', filename='image.jpg')
            message.attach(image)
        server.sendmail(from_email, to_email, message.as_string())

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
        self.detector.change_zone(np.array(cords, dtype=int).reshape((4, 2)))

    def activate_detector_button_clicked(self):
        if self.is_active_detector:
            self.is_active_detector = False
            self.ui.activate_people_detector.setText('Выключить\nраспознавание людей\nна видео')
        else:
            self.is_active_detector = True
            self.ui.activate_people_detector.setText('Включить\nраспознавание людей\nна видео')

        self.thread.set_active_detector(self.is_active_detector)

    @Slot(np.ndarray)
    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_stream.setPixmap(qt_img)

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