class ReasoningEngine:
    def analyze(self, english_text):

        if "unclear" in english_text.lower():
            return (
                "Needs Human Review",
                "Neutral",
                "Low",
                "Assign a field worker to re-record or manually interpret."
            )

        text = english_text.lower()

        if any(w in text for w in ["fish", "fishing", "livelihood", "work"]):
            return (
                "Fishing Livelihood",
                "Neutral",
                "Medium",
                "Assess employment conditions and provide livelihood support."
            )

        return (
            "Livelihood Issue",
            "Neutral",
            "Medium",
            "Conduct field assessment and provide local support."
        )
