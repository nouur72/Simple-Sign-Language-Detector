class TextToSpeech:
    def __init__(self):
        self.engine = None
        self.last_spoken = ""
        self._initialize()

    def _initialize(self):
        try:
            import pyttsx3
        except ImportError:
            self.engine = None
            return

        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", 160)
            self.engine.setProperty("volume", 0.9)
        except Exception:
            self.engine = None

    def speak(self, text):
        if not text:
            return

        normalized_text = str(text).strip()
        if not normalized_text or normalized_text == self.last_spoken:
            return

        if self.engine is None:
            print(f"[TTS] {normalized_text}")
            self.last_spoken = normalized_text
            return

        try:
            self.engine.say(normalized_text)
            self.engine.runAndWait()
        except Exception as exc:
            print(f"TTS error: {exc}")

        self.last_spoken = normalized_text
