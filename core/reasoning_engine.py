class ReasoningEngine:
    def analyze(self, english_text):
        text = english_text.lower()

        if any(w in text for w in ["storm", "rain", "weather", "wind"]):
            return "Weather Risk", "Distressed", "High", "Provide weather alerts, compensation, and safety guidance."

        if any(w in text for w in ["loan", "debt", "money"]):
            return "Financial Stress", "Distressed", "High", "Connect to debt relief and financial aid programs."

        if any(w in text for w in ["happy", "good", "fine"]):
            return "Stable Livelihood", "Happy", "Low", "Continue monitoring and support schemes."

        return "Livelihood Issue", "Neutral", "Medium", "Conduct field assessment and provide local support."
