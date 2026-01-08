import pyttsx3, os, uuid
from config.settings import Settings

class TTSService:
    def __init__(self):
        os.makedirs(Settings.TTS_DIR, exist_ok=True)
        self.engine = pyttsx3.init()

    def generate(self, text):
        file_id = str(uuid.uuid4())
        path = f"{Settings.TTS_DIR}/{file_id}.mp3"
        self.engine.save_to_file(text, path)
        self.engine.runAndWait()
        return path
