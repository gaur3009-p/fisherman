import whisper


class ASRService:
    def __init__(self):
        # Use a stronger model for Indian languages
        self.model = whisper.load_model("large-v3")

    def transcribe(self, audio_path):
        """
        Accurate transcription for Indian languages.
        """
        result = self.model.transcribe(
            audio_path,
            task="transcribe",
            temperature=0.0,
            no_speech_threshold=0.3,
            logprob_threshold=-1.0
        )

        text = result.get("text", "").strip()
        language = result.get("language", "unknown")

        return text, language
