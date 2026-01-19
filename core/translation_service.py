import torch
from transformers import AutoProcessor, SeamlessM4TModel


class TranslationService:
    """
    Odia → English using Meta SeamlessM4T-v2
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "facebook/seamless-m4t-v2-large"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4TModel.from_pretrained(
            self.model_name
        ).to(self.device)

        self.src_lang = "ory"   # Odia
        self.tgt_lang = "eng"   # English

    def to_english_from_text(self, odia_text: str) -> str:
        inputs = self.processor(
            text=odia_text,
            src_lang=self.src_lang,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                tgt_lang=self.tgt_lang,
                max_new_tokens=256
            )

        # ✅ remove batch dimension safely
        token_ids = generated_tokens[0].cpu().tolist()

        translation = self.processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True
        )

        return translation.strip()
