import whisper


class TranslationService:
    def __init__(self):
        self.model = whisper.load_model("large-v3")

    def to_english(self, audio_path):
        """
        Force Whisper to translate to English.
        """
        result = self.model.transcribe(
            audio_path,
            task="translate",
            temperature=0.0
        )

        return result.get("text", "").strip()
