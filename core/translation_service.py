import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationService:
    """
    Odia → English using IndicTrans2 (AI4Bharat)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = "ai4bharat/indictrans2-indic-en-1B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to(self.device)

    def to_english_from_text(self, odia_text: str) -> str:
        inputs = self.tokenizer(
            odia_text,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256
            )

        translation = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return translation.strip()
