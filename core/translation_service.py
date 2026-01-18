from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TranslationService:
    """
    Odia → English translation using IndicTrans2 (AI4Bharat).
    BEST model for Indian languages.
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

        self.model.to("cpu")  # safe for Kaggle / Python 3.12

    def to_english_from_text(self, odia_text: str) -> str:
        inputs = self.tokenizer(
            odia_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()
