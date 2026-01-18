from faster_whisper import WhisperModel
import torch

class WhisperService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            cls._model = WhisperModel(
                "large-v3",
                device=device,
                compute_type=compute_type
            )

        return cls._model
