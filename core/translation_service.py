from core.whisper_service import WhisperService


class TranslationService:
    def __init__(self):
        self.model = WhisperService.get_model()

    def to_english(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            task="translate",
            temperature=0.0
        )
        return result.get("text", "").strip()
