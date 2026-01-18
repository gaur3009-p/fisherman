from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class TranslationService:
    """
    Odia → English using IndicTrans2 (public, supported model).
    """

    def __init__(self):
        self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True
        ).to("cpu")

        # IMPORTANT: set source language (Odia)
        self.tokenizer.src_lang = "ory_Orya"
        self.tgt_lang = "eng_Latn"

    def to_english_from_text(self, odia_text: str) -> str:
        inputs = self.tokenizer(
            odia_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
            )

        return self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        ).strip()
