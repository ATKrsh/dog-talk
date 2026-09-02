"""
Dog Talk - LLM Interpreter
Uses local Ollama LLM for natural language interpretation of dog behavior.
Falls back gracefully if LLM is unavailable.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMInterpreter:
    """
    Generates natural language interpretations of dog behavior
    using a local LLM via Ollama.
    """

    SYSTEM_PROMPT = """You are an expert canine behaviorist and ethologist. You analyze dog body language and vocalizations to interpret what a dog is feeling and communicating.

Your responses should be:
- Warm, friendly, and engaging (you're talking to pet owners)
- Scientifically accurate based on ethology research
- Concise (2-3 sentences max)
- Written in present tense as if observing the dog right now
- Include specific behavioral observations
- Add safety warnings when the dog shows signs of stress/aggression

Do NOT use bullet points, headers, or formatting. Write natural, flowing sentences."""

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "llama3.2-vision", timeout: float = 10.0):
        self.host = host
        self.model = model
        self.timeout = timeout
        self.available = False
        self._check_availability()

    def _check_availability(self):
        """Check if Ollama is running and model is available."""
        try:
            import ollama
            client = ollama.Client(host=self.host)
            models = client.list()
            model_names = [m.get("name", m.get("model", "")) for m in models.get("models", [])]
            
            if any(self.model in name for name in model_names):
                self.available = True
                logger.info(f"Ollama available with model: {self.model}")
            else:
                # Try a smaller model
                for fallback in ["llama3.2", "llama3.1", "gemma2", "phi3", "mistral"]:
                    if any(fallback in name for name in model_names):
                        self.model = next(name for name in model_names if fallback in name)
                        self.available = True
                        logger.info(f"Using fallback model: {self.model}")
                        break
                
                if not self.available:
                    logger.warning(f"No suitable LLM found. Available: {model_names}")
                    
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self.available = False

    def interpret(self, body_language: dict, vocalization: dict,
                  knowledge_interpretation: str = "",
                  image_base64: str = None) -> str:
        """
        Generate natural language interpretation of dog behavior.
        
        Args:
            body_language: Dict with tail, ears, posture info
            vocalization: Dict with sound type and meaning
            knowledge_interpretation: Pre-generated interpretation from knowledge base
            image_base64: Optional base64 JPEG for vision model analysis
            
        Returns:
            Natural language interpretation string
        """
        if not self.available:
            return knowledge_interpretation or self._generate_fallback(body_language, vocalization)

        try:
            import ollama

            # Build prompt
            prompt = self._build_prompt(body_language, vocalization, knowledge_interpretation)
            
            client = ollama.Client(host=self.host)
            
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            # Add image if available and model supports vision
            if image_base64 and "vision" in self.model.lower():
                import base64
                messages[-1]["images"] = [image_base64]

            response = client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0.7, "num_predict": 150}
            )

            result = response.get("message", {}).get("content", "")
            if result.strip():
                return result.strip()
            
            return knowledge_interpretation or self._generate_fallback(body_language, vocalization)

        except Exception as e:
            logger.warning(f"LLM interpretation failed: {e}")
            return knowledge_interpretation or self._generate_fallback(body_language, vocalization)

    def _build_prompt(self, body_language: dict, vocalization: dict,
                      knowledge_interpretation: str) -> str:
        """Build a structured prompt for the LLM."""
        parts = ["Analyze this dog's current behavior:\n"]
        
        # Body language
        if body_language:
            parts.append("BODY LANGUAGE:")
            if "tail" in body_language:
                t = body_language["tail"]
                parts.append(f"- Tail: {t.get('position', 'unknown')} position, {t.get('movement', 'unknown')} movement")
            if "ears" in body_language:
                e = body_language["ears"]
                parts.append(f"- Ears: {e.get('position', 'unknown')}")
            if "posture" in body_language:
                p = body_language["posture"]
                parts.append(f"- Posture: {p.get('stance', 'unknown')}")
            if "hackles" in body_language:
                h = body_language["hackles"]
                parts.append(f"- Hackles: {'raised' if h.get('raised') else 'flat'}")

        # Vocalization
        if vocalization:
            parts.append(f"\nVOCALIZATION: {vocalization.get('type', 'none')}")
            if vocalization.get('meaning'):
                parts.append(f"Typical meaning: {vocalization['meaning']}")

        # Knowledge base interpretation
        if knowledge_interpretation:
            parts.append(f"\nKNOWLEDGE BASE SUGGESTS: {knowledge_interpretation}")

        parts.append("\nProvide a warm, natural interpretation of what this dog is feeling and communicating. Include what they might do next.")

        return "\n".join(parts)

    def _generate_fallback(self, body_language: dict, vocalization: dict) -> str:
        """Generate a basic interpretation without LLM."""
        parts = []
        
        # Interpret body language
        if body_language:
            tail = body_language.get("tail", {})
            if tail.get("meaning"):
                parts.append(tail["meaning"])
            
            posture = body_language.get("posture", {})
            if posture.get("meaning"):
                parts.append(posture["meaning"])

        # Interpret vocalization
        if vocalization and vocalization.get("meaning"):
            parts.append(vocalization["meaning"])

        if not parts:
            return "Observing this dog's behavior... The signals are mixed or unclear. Continue watching for more definitive body language."

        return " ".join(parts)
