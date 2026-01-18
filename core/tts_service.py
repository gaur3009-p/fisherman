import os
import uuid
from gtts import gTTS
from config.settings import Settings


class TTSService:
    """
    English text → speech using gTTS.
    Python 3.12 compatible.
    """

    def __init__(self):
        os.makedirs(Settings.TTS_DIR, exist_ok=True)

    def generate(self, text: str) -> str:
        file_id = str(uuid.uuid4())
        output_path = os.path.join(Settings.TTS_DIR, f"{file_id}.mp3")

        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(output_path)

        return output_path
