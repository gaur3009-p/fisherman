from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TranslationService:
    """
    Odia → English translation using NLLB (modern Transformers API).
    Python 3.12 compatible.
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

        # Set source language (Odia)
        self.tokenizer.src_lang = "ory_Orya"

        # Cache target language token ID (English)
        self.tgt_lang_token_id = self.tokenizer.convert_tokens_to_ids("eng_Latn")

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
                forced_bos_token_id=self.tgt_lang_token_id,
                max_length=128,
                num_beams=5
            )

        translation = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return translation.strip()
