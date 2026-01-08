import whisper

class TranslationService:
    def __init__(self):
        self.model = whisper.load_model("base")

    def to_english(self, audio_path):
        r = self.model.transcribe(audio_path, task="translate")
        return r["text"]
