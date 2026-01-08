import os, uuid, soundfile as sf
from config.settings import Settings

class AudioService:
    def __init__(self):
        os.makedirs(Settings.AUDIO_DIR, exist_ok=True)

    def save(self, audio_np, sample_rate):
        audio_id = str(uuid.uuid4())
        path = f"{Settings.AUDIO_DIR}/{audio_id}.wav"
        sf.write(path, audio_np, sample_rate)
        return audio_id, path
