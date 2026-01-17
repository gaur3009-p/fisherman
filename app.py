import gradio as gr
import uuid
from datetime import datetime

# Core services
from core.audio_service import AudioService
from core.asr_service import ASRService
from core.translation_service import TranslationService
from core.rag_engine import RAGEngine
from core.reasoning_engine import ReasoningEngine
from core.tts_service import TTSService

# Data layer
from data_layer.vector_store import VectorStore
from data_layer.dataset_manager import DatasetManager


# ---------------------------
# INITIALIZE ENTERPRISE SERVICES
# ---------------------------

audio_service = AudioService()
asr_service = ASRService()
translation_service = TranslationService()

vector_store = VectorStore()
rag_engine = RAGEngine(vector_store)

reasoning_engine = ReasoningEngine()
tts_service = TTSService()

dataset_manager = DatasetManager()


# ---------------------------
# MAIN PIPELINE (END-TO-END)
# ---------------------------

def process_fisherman_voice(audio):
    """
    Full enterprise pipeline:
    Voice → STT → Translation → RAG → Reasoning → Dataset → TTS
    """

    if audio is None:
        return (
            "No audio provided",
            "",
            "",
            "",
            "",
            "",
            None
        )

    # 1. Save audio
    sample_rate, audio_np = audio
    record_id = str(uuid.uuid4())
    _, audio_path = audio_service.save(audio_np, sample_rate)

    native_text, language = asr_service.transcribe(audio_path)

    # SAFETY: if transcription is empty or junk
    if len(native_text) < 5:
        native_text = "Unable to clearly transcribe speech."
    
    english_text = translation_service.to_english(audio_path)
    
    # SAFETY: avoid hallucinated translations
    if len(english_text) < 5:
        english_text = "Translation unclear. Needs human review."

    # 4. Retrieve RAG context (past NGO cases)
    rag_context = rag_engine.retrieve(english_text)

    # 5. Reasoning (enterprise intelligence)
    issue_title, sentiment, urgency, ngo_action = reasoning_engine.analyze(
        english_text
    )

    # 6. Clean NGO-ready summary
    clean_summary = (
        f"The fisherman reports the following issue: {english_text}. "
        f"This case is categorized under '{issue_title}'. "
        f"The emotional condition of the fisherman appears {sentiment.lower()}."
    )

    # 7. Add to vector store (institutional memory)
    vector_store.add(english_text)

    # 8. Persist EVERYTHING using DatasetManager (single source of truth)
    dataset_manager.save_record({
        "record_id": record_id,
        "audio_path": audio_path,
        "native_text": native_text,
        "english_text": english_text,
        "issue_title": issue_title,
        "clean_summary": clean_summary,
        "sentiment": sentiment,
        "urgency": urgency,
        "ngo_action": ngo_action,
        "rag_context": rag_context,
        "timestamp": datetime.utcnow()
    })

    # 9. Generate TTS for field data collector
    tts_audio_path = tts_service.generate(
        f"Issue identified: {issue_title}. "
        f"Fisherman emotional state: {sentiment}. "
        f"Urgency level: {urgency}. "
        f"Suggested NGO action: {ngo_action}"
    )

    # 10. Return outputs to UI
    return (
        native_text,
        english_text,
        issue_title,
        sentiment,
        urgency,
        ngo_action,
        tts_audio_path
    )


# ---------------------------
# GRADIO ENTERPRISE UI
# ---------------------------

with gr.Blocks(title="NGO Enterprise Voice Intelligence Platform") as app:

    gr.Markdown(
        """
        ## 🌊 NGO Enterprise Voice Intelligence System  
        **Collect → Understand → Act on Fishermen Difficulties**

        This system converts spoken problems into structured,
        actionable intelligence for NGOs and policy teams.
        """
    )

    with gr.Row():
        audio_input = gr.Audio(
            type="numpy",
            label="🎙️ Fisherman Voice Input"
        )

    with gr.Row():
        native_output = gr.Textbox(
            label="📝 Native Language Transcription",
            lines=3
        )

    with gr.Row():
        english_output = gr.Textbox(
            label="🌍 English Translation",
            lines=3
        )

    with gr.Row():
        issue_title_output = gr.Textbox(
            label="🏷️ Issue Title"
        )
        sentiment_output = gr.Textbox(
            label="😊 Emotional State"
        )
        urgency_output = gr.Textbox(
            label="🚨 Urgency Level"
        )

    with gr.Row():
        ngo_action_output = gr.Textbox(
            label="🤝 Suggested NGO Action",
            lines=3
        )

    with gr.Row():
        tts_output = gr.Audio(
            label="🔊 Audio Summary for Field Worker"
        )

    submit_btn = gr.Button("Analyze & Save Case", variant="primary")

    submit_btn.click(
        fn=process_fisherman_voice,
        inputs=[audio_input],
        outputs=[
            native_output,
            english_output,
            issue_title_output,
            sentiment_output,
            urgency_output,
            ngo_action_output,
            tts_output
        ]
    )


# ---------------------------
# LAUNCH APP
# ---------------------------

app.launch(share=True)
