# Doan nay import cac ham trong core 
# Gia code: 
# Khoi tao
# translator = ()
# model = ()
# image_magician = ()
# #
import sys
from PyQt6.QtWidgets import QApplication, QLabel#, QMainWindow
from PyQt6.QtCore import Qt, QTimer
from core.cap_MH.image_magician import Image_Magician
from core.cap_MH.image_magician import OverlayWindow, SelectionWindow
from core.dich_thuat.Model import TextDetector, TranslateEngine
from PyQt6.QtCore import Qt
# Khoi tao may cai class

class Main_App():

    def __init__(self):
        self.image_magician = Image_Magician()
        self.select_window = None
        self.draw_window = OverlayWindow(self.image_magician)
        self.text_detector = TextDetector()
        self.tran_engine = TranslateEngine()
        self.timer = QTimer()
        self.timer.timeout.connect(self.work_loop)
        self.active_roi = None

    def lua_chon(self,event):
        print('Nhấn A để bắt đầu làm việc, nhấn Esc để kết thúc')
        if event.key() == Qt.Key.Key_A:
            self.workflow() 
        if event.key() == Qt.Key.Key_Escape:
            QApplication.quit()
        
    def start_select(self):
        self.timer.stop()
        self.draw_window.hide()
        self.select_window = SelectionWindow()
        self.select_window.area_selected.connect(self.setup_toado)
        self.select_window.show()

    def setup_toado(self,roi):
        self.active_roi = roi
        self.draw_window.setGeometry(
        roi["left"],
            roi["top"],
            roi["width"],
            roi["height"]
)
        self.draw_window.show()
        self.timer.start(1000)

    def work_loop(self):
        if not self.active_roi:
            return
        new_h, img = self.image_magician.capture_image(self.active_roi)
        if new_h:
            res_boxes = self.text_detector.predict(img, conf = 0.4, classes = [3]
                                                   , verbose = False )
            self.image_magician.use_with_YOLO_res(res_boxes=res_boxes, 
                                                  img=img, 
                                                  translator=self.tran_engine)
            self.draw_window.update()

    def workflow(self):
        #1. Cap man hinh
        self.start_select()

class ControlPanel(QLabel):
    """Cửa sổ điều khiển nhỏ để bắt sự kiện bàn phím"""
    def __init__(self, main_app_logic):
        super().__init__("Nhấn 'A' để chọn vùng dịch\nNhấn 'Esc' để thoát")
        self.logic = main_app_logic
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.resize(250, 100)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

    def keyPressEvent(self, event):
        # Truyền event vào hàm lua_chon của Main_App
        self.logic.lua_chon(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 1. Khởi tạo logic nghiệp vụ
    main_logic = Main_App()
    
    # 2. Khởi tạo giao diện điều khiển và truyền logic vào
    panel = ControlPanel(main_logic)
    panel.show()
    
    sys.exit(app.exec())