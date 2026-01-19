from faster_whisper import WhisperModel
import torch


class WhisperService:
    _model = None

    @classmethod
    def get_model(cls):
        if cls._model is None:
            if torch.cuda.is_available():
                # Kaggle-safe GPU configuration
                cls._model = WhisperModel(
                    "medium",              # ⬅️ NOT large
                    device="cuda",
                    compute_type="int8",   # ⬅️ HUGE memory saver
                    device_index=0
                )
            else:
                # CPU fallback
                cls._model = WhisperModel(
                    "medium",
                    device="cpu",
                    compute_type="int8"
                )

        return cls._model
