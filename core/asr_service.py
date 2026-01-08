import whisper

class ASRService:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path):
        r = self.model.transcribe(audio_path)
        return r["text"], r["language"]
