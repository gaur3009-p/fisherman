import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


class TranslationService:
    """
    Odia → English translation using IndicTrans2 + IndicProcessor
    (Official AI4Bharat inference method)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Public, supported model
        self.model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # IMPORTANT: do NOT force flash_attention unless available
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        # Indic processor (critical)
        self.processor = IndicProcessor(inference=True)

        # Language codes
        self.src_lang = "ory_Orya"   # Odia
        self.tgt_lang = "eng_Latn"   # English

    def to_english_from_text(self, odia_text: str) -> str:
        """
        Translate Odia text → English using official IndicTrans2 flow
        """

        # 1. Preprocess (VERY IMPORTANT)
        batch = self.processor.preprocess_batch(
            [odia_text],
            src_lang=self.src_lang,
            tgt_lang=self.tgt_lang
        )

        # 2. Tokenize
        inputs = self.tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True
        ).to(self.device)

        # 3. Generate
        with torch.no_grad():
            generated_tokens = self.model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,
                num_beams=5,
                num_return_sequences=1
            )

        # 4. Decode
        decoded = self.tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # 5. Postprocess (entity + script fix)
        translations = self.processor.postprocess_batch(
            decoded,
            lang=self.tgt_lang
        )

        return translations[0].strip()
