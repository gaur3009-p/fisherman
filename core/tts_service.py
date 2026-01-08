import os
import uuid
from gtts import gTTS
from config.settings import Settings


class TTSService:
    """
    Text-to-Speech service using gTTS.
    - No system dependencies
    - Works in cloud / Colab / Docker
    - Generates .mp3 output
    """

    def __init__(self):
        # Ensure TTS directory exists
        os.makedirs(Settings.TTS_DIR, exist_ok=True)

    def generate(self, text: str) -> str:
        """
        Convert text to speech and save as MP3.

        Args:
            text (str): Text to convert to speech

        Returns:
            str: Path to generated MP3 file
        """
        file_id = str(uuid.uuid4())
        output_path = os.path.join(Settings.TTS_DIR, f"{file_id}.mp3")

        # Generate speech
        tts = gTTS(
            text=text,
            lang="en",
            slow=False
        )
        tts.save(output_path)

        return output_path
