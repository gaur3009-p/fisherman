from core.whisper_service import WhisperService


class ASRService:
    def __init__(self):
        self.model = WhisperService.get_model()

    def transcribe(self, audio_path):
        result = self.model.transcribe(
            audio_path,
            task="transcribe",
            temperature=0.0
        )
        return result.get("text", "").strip(), result.get("language", "unknown")
