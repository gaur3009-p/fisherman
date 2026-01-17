import whisper
import torch


class WhisperService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = whisper.load_model(
                "medium",   # SAFE for Kaggle/Colab
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        return cls._model
