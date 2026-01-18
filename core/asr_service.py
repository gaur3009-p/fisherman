from core.whisper_service import WhisperService

class ASRService:
    def __init__(self):
        self.model = WhisperService.get_model()

    def transcribe(self, audio_path: str):
        segments, info = self.model.transcribe(
            audio_path,
            language="or",          
            beam_size=5,
            vad_filter=True
        )

        text = " ".join(seg.text.strip() for seg in segments)
        language = info.language

        return text.strip(), language
