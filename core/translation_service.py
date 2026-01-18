from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class TranslationService:
    """
    Odia → English using NLLB (text-to-text).
    Python 3.12 safe.
    """

    def __init__(self):
        self.model_name = "facebook/nllb-200-distilled-600M"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=False
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name
        ).to("cpu")

    def to_english_from_text(self, odia_text: str) -> str:
        inputs = self.tokenizer(
            odia_text,
            return_tensors="pt",
            truncation=True
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id["eng_Latn"],
                max_length=128,
                num_beams=5
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()
