class PromptEngine:
    def build(self, english_text, context):
        return f"""
You are an NGO intelligence analyst.

Original statement:
"{english_text}"

Related historical cases:
{context}

TASK:
1. Rewrite clearly in professional NGO language
2. Assign ONE best-fit title
3. Detect emotional state (Happy / Neutral / Distressed)
4. Assess urgency (Low / Medium / High)
5. Suggest concrete NGO help steps

Respond STRICTLY as JSON:
{{
  "title": "",
  "summary": "",
  "sentiment": "",
  "urgency": "",
  "ngo_action": ""
}}
"""
