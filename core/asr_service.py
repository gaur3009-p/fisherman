from core.whisper_service import WhisperService

class ASRService:
    def __init__(self):
        self.model = WhisperService.get_model()

    def transcribe(self, audio_path: str):
        """
        Transcribe Odia (and other Indian languages) using auto-detection.
        """
        segments, info = self.model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(seg.text.strip() for seg in segments)
        language = info.language  # may return 'hi', 'bn', etc.

        return text.strip(), language
