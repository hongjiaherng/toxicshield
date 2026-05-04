import os, logging
from typing import List, Optional
from logging.handlers import RotatingFileHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# load environmental variables
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "log", "decision_analyst.log")

# initialize the logger
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[RotatingFileHandler(LOG_PATH, maxBytes=5_000_000)])
logger = logging.getLogger(__name__)



# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDER_CONFIG = {
    "glm": "ZHIPU",
    "gemini": "GEMINI",
    "gpt": "OPENAI",
    "claude": "ANTHROPIC",
}


def get_provider_credentials(model_name: str) -> tuple[str, str]:
    """Resolve API key and base URL based on model name prefix."""
    for prefix, env_prefix in PROVIDER_CONFIG.items():
        if model_name.lower().startswith(prefix):
            api_key = os.getenv(f"{env_prefix}_API_KEY")
            api_base = os.getenv(f"{env_prefix}_API_BASE")
            if not api_key or not api_base:
                raise ValueError(f"Missing {env_prefix}_API_KEY or {env_prefix}_API_BASE")
            return api_key, api_base
    raise ValueError(f"Unknown model provider for: {model_name}")


def create_chat_llm(model_name: str, temperature: float = 0.0) -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for the given model."""
    api_key, api_base = get_provider_credentials(model_name)
    return ChatOpenAI(
        model=model_name,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=temperature,
    )


# =============================================================================
# Data Models
# =============================================================================

class ExplanationResult(BaseModel):
    """Structured output for classification explanation."""
    input_text: str = Field(description="Original input text")
    predicted_label: str = Field(description="Local model's classification")
    justification: str = Field(description="Explanation (≤30 words)")


# =============================================================================
# Classification Explainer
# =============================================================================

EXPLANATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
     Explain why the text was classified with this label by identifying specific linguistic patterns or features. 
     Use the provided examples to inform your reasoning, but present the insights as your own direct analysis. 
     Do not mention that you are using references or examples.
     Max 30 words. 
     
     Similar toxic examples for reference:
    {similar_examples}"""),
    ("user", """Text: {input_text}
Label: {predicted_label}

Why was this classified as "{predicted_label}"? (≤30 words):"""),
])


class ClassificationExplainer:
    def __init__(self, model_name: str = "glm-5", temperature: float = 0.3):
        self.llm = create_chat_llm(model_name, temperature)
        self._chain = EXPLANATION_PROMPT | self.llm | StrOutputParser()

    def explain(
        self,
        input_text: str,
        predicted_label: str,
        similar_examples: Optional[List[dict]] = None,
    ) -> ExplanationResult:
        # Format similar examples for the prompt
        examples_str = ""
        if similar_examples:
            for i, ex in enumerate(similar_examples, 1):
                examples_str += f"{i}. [{ex.get('category', 'Unknown')}] {ex.get('text', '')}\n"

        justification = self._chain.invoke({
            "input_text": input_text,
            "predicted_label": predicted_label,
            "similar_examples": examples_str or "No similar examples provided.",
        })
        return ExplanationResult(
            input_text=input_text,
            predicted_label=predicted_label,
            justification=justification.strip(),
        )
