
from ultralytics import YOLO
from deep_translator import GoogleTranslator
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
# core/dich_thuat/Model.py
# ↑ lên core
# ↑ lên AI_FOR_SCREEN_TRANSLATING

model_place = BASE_DIR / "train_AI" / "runs" / "best.pt"

class TextDetector:
    def __init__(self):
        self.AI_MODEL = YOLO(model=str(model_place))
    
    def predict(self, image, conf=0.4, classes=None, verbose=False):
            return self.AI_MODEL.predict(
                image,
                conf=conf,
                classes=classes,
                verbose=verbose
            )


class TranslateEngine:
    def __init__(self):
        self.translator = GoogleTranslator(source='ja', target='vi') #Về sau mặc định là ja sau

    def change_lang(self, source_lang: str, target_lang: str):
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)

    def translate(self, text: str) -> str:
        return self.translator.translate(text)
