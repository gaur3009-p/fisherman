import os
import uuid
from TTS.api import TTS
from config.settings import Settings

class TTSService:
    """
    English TTS using Coqui-TTS (CPU-only).
    Python 3.12 compatible.
    """

    def __init__(self):
        os.makedirs(Settings.TTS_DIR, exist_ok=True)

        self.tts = TTS(
            model_name="tts_models/en/ljspeech/tacotron2-DDC",
            gpu=False
        )

    def generate(self, text: str) -> str:
        file_id = str(uuid.uuid4())
        output_path = os.path.join(Settings.TTS_DIR, f"{file_id}.wav")

        self.tts.tts_to_file(
            text=text,
            file_path=output_path
        )

        return output_path
