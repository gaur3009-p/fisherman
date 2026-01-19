import torch
from transformers import AutoProcessor, SeamlessM4TModel


class TranslationService:
    """
    Odia → English using Meta SeamlessM4T-v2 (STRONGER than NLLB).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = "facebook/seamless-m4t-v2-large"

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = SeamlessM4TModel.from_pretrained(
            self.model_name
        ).to(self.device)

        # Language codes
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
    
        # ---- FINAL SAFE DECODING (CRITICAL FIX) ----
        tokens = generated_tokens
    
        # Tensor → list
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
    
        # Flatten nested lists
        while isinstance(tokens, list) and len(tokens) == 1:
            tokens = tokens[0]
    
        # Absolute safety check
        if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
            raise RuntimeError(f"Invalid token format from model: {tokens}")
    
        translation = self.processor.tokenizer.decode(
            tokens,
            skip_special_tokens=True
        )
    
        return translation.strip()
