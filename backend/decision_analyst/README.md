# decision_analyst

LLM-powered toxicity classification explainer. Given a text and its predicted label, it generates a short justification for *why* the text was classified that way, using similar examples from the vector DB as context.

## Quick Start

```bash
# Run from backend/ root — imports resolve relative to this directory
cd backend
uv run python
```

```python
from decision_analyst.decision_analyst import ClassificationExplainer
from vectordb.vectordb import VectorDB

db = VectorDB()
db.setup()

explainer = ClassificationExplainer(model_name="glm-5")

similar_examples = db.hybrid_search("You're such an idiot", k=3)
result = explainer.explain(
    input_text="You're such an idiot",
    predicted_label="TOXIC",
    similar_examples=similar_examples,
)

print(result.justification)  # e.g. "The text features direct personal insults..."
```

## API

### `ClassificationExplainer(model_name, temperature)`

| Param          | Default         | Description                                        |
|----------------|-----------------|----------------------------------------------------|
| `model_name`   | `"gpt-4o-mini"` | LLM to use. Prefix (case-insensitive) determines provider (see below) |
| `temperature`  | `0.3`           | LLM sampling temperature                           |

**Supported model providers** — matched by case-insensitive prefix of `model_name`:

| Prefix    | Env vars required                        | Example `model_name` |
|-----------|------------------------------------------|----------------------|
| `glm`     | `ZHIPU_API_KEY`, `ZHIPU_API_BASE`        | `glm-5`              |
| `gemini`  | `GEMINI_API_KEY`, `GEMINI_API_BASE`      | `gemini-2.0-flash`   |
| `gpt`     | `OPENAI_API_KEY`, `OPENAI_API_BASE`      | `gpt-4o-mini`        |
| `claude`  | `ANTHROPIC_API_KEY`, `ANTHROPIC_API_BASE`| `claude-sonnet-4-6`  |

Env vars are loaded from `backend/.env` via `dotenv` (follows standard `.env` lookup — places a `.env` at `backend/` root).

### `explainer.explain(input_text, predicted_label, similar_examples) → ExplanationResult`

| Param              | Type             | Description                                        |
|--------------------|------------------|----------------------------------------------------|
| `input_text`       | `str`            | The text to explain                                |
| `predicted_label`  | `str`            | `"TOXIC"` or `"NON_TOXIC"`                         |
| `similar_examples` | `list[dict]`     | Output from `db.hybrid_search()`. Each dict needs `category` and `text` keys. Pass `None` to skip context (LLM explains without examples). |

Returns an `ExplanationResult` (Pydantic model):

```python
result.input_text        # str — original input text (echoed back)
result.predicted_label   # str — the label you passed in (echoed back)
result.justification     # str — LLM-generated explanation (~30 words, soft limit from prompt)
```

## How It Works

1. `similar_examples` from `VectorDB.hybrid_search()` are formatted as numbered references
2. A LangChain chain (`ChatPromptTemplate → ChatOpenAI → StrOutputParser`) generates the explanation
3. The system prompt instructs the LLM to analyze linguistic patterns in ~30 words, using the examples as context but without disclosing them

## Running Tests

```bash
cd backend
uv run python decision_analyst/test_decision_analyst.py
```

Requires API keys and Qdrant connection. Output goes to `backend/log/test_decision_analyst.log`.

## Adding a New Provider

1. Add the env vars to `backend/.env`: `<PREFIX>_API_KEY` and `<PREFIX>_API_BASE`
2. Add an entry to `PROVIDER_CONFIG` in `decision_analyst.py`:
   ```python
   PROVIDER_CONFIG = {
       # ...existing entries
       "deepseek": "DEEPSEEK",
   }
   ```
3. Pass a `model_name` starting with that prefix (e.g. `"deepseek-chat"`)

The provider just needs to expose an OpenAI-compatible API at the configured base URL.
