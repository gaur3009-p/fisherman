from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TranslationService:
    """
    Proper Odia → English translation using IndicTrans2.
    This replaces Whisper translation completely.
    """

    def __init__(self):
        self.model_name = "ai4bharat/indictrans2-od-en-dist-200M"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def to_english_from_text(self, odia_text: str) -> str:
        """
        Translate Odia text to English accurately.
        """
        inputs = self.tokenizer(
            odia_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5
            )

        translation = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return translation.strip()
