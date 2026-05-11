import cv2
import numpy as np
import mss
from PIL import Image
import time
from PyQt6.QtWidgets import  QWidget # Làm giao diện Overlay , neu loi thi import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen #Qfont
from manga_ocr import MangaOcr
from PyQt6.QtCore import QRect, pyqtSignal
#from core.dich_thuat.Model import TranslateEngine as TE
class Image_Magician:
    def __init__(self):
        self.cache_database = {}  # Lưu {Hash_ảnh_box: "Văn bản đã dịch"}
        self.current_results = [] # Lưu tọa độ và text để vẽ lên màn hình
        self.last_screen_hash = None # Kiểm tra thay đổi toàn cục
        self.roi = (0,0,500,500)
        self.mocr = MangaOcr()
        #self.translator = TE()

    def make_image_hash(self,image): # Ảnh sau khi qua bước grayscale và các bước 
                                     # xử lí ảnh khác.
        pixels = np.array(image)
        diff = pixels[:, 1:] > pixels[:, :-1]
        hash_value = ''.join('1' if b else '0' for b in diff.flatten())
        return hash_value

    def capture_image(self,roi):
        with mss.mss() as sct:
            #while True:
                #1. Cap MH
                screenshot = sct.grab(roi)                
                img = np.array(screenshot)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR) #Tối thiểu nhất
                h = self.make_image_hash(img)
                new_h = (h != self.last_screen_hash)
                # if h == new_h:
                #     time.sleep(1)
                #     continue
                self.last_screen_hash = h
                return new_h,img


    def use_with_YOLO_res(self, res_boxes,img,translator):
        new_display_data = []
        for box in res_boxes[0].boxes:
            toa_do = box.xyxy[0].tolist()
            crop = self.crop_text(img,toa_do,pad=15)
            box_hash = self.make_image_hash(crop)
            if box_hash in self.cache_database:
                trans_text = self.cache_database[box_hash]
            else:
                og_text = self.mocr(Image.fromarray(crop))
                trans_text = translator.translate(og_text)
                self.cache_database[box_hash] = trans_text
            new_display_data.append({'rect':toa_do, 'text':trans_text})
        self.current_results = new_display_data
    
    def crop_text(self,img,toa_do,pad):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = map(int, toa_do)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img



class SelectionWindow(QWidget):
    # Signal này sẽ gửi tọa độ ROI về sau khi chọn xong
    area_selected = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        # Làm cửa sổ mờ che toàn màn hình để người dùng kéo
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowOpacity(0.3) # Độ mờ 30%
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.showFullScreen()

        self.start_point = None
        self.end_point = None
        self.is_selecting = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_point = event.pos()
            self.is_selecting = True

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.end_point = event.pos()
            self.update() # Gọi lại paintEvent để vẽ hình chữ nhật đang kéo

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = False

            rect = QRect(self.start_point, self.end_point).normalized()
            x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

            # ✅ bắt buộc: mss KHÔNG nhận w/h <= 0
            if w <= 0 or h <= 0:
                self.close()
                return

            roi = {
                "left": int(x),
                "top": int(y),
                "width": int(w),
                "height": int(h)
            }

            self.area_selected.emit(roi)
            self.close()

    def paintEvent(self, event):
        if self.start_point and self.end_point:
            painter = QPainter(self)
            painter.setPen(QPen(QColor(255, 0, 0), 2)) # Vẽ viền đỏ cho vùng đang chọn
            rect = QRect(self.start_point, self.end_point)
            painter.drawRect(rect)

# --- 3. TẦNG HIỂN THỊ (OVERLAY TRONG SUỐT) ---
class OverlayWindow(QWidget):
    """ Cửa sổ trong suốt đè lên màn hình """
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        # Thuộc tính cửa sổ: Trong suốt, luôn nằm trên, không viền
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def paintEvent(self, event):
        """ Vẽ chữ lên màn hình mỗi khi có dữ liệu mới """
        painter = QPainter(self)
        for item in self.controller.current_results:
            x, y, x2, y2 = item['rect']
            # Vẽ nền màu (Inpaint giả)
            painter.setBrush(QColor(255, 255, 255, 200)) # Trắng đục
            painter.drawRect(int(x), int(y), int(x2-x), int(y2-y))
            
            # Vẽ chữ Việt
            painter.setPen(QColor(0, 0, 0)) # Chữ đen
            painter.drawText(int(x), int(y), int(x2-x), int(y2-y), 
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, 
                             item['text'])


                
    


