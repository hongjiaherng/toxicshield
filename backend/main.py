import contextlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import pipeline
import demoji
import asyncio
import random

# Use the lightweight toxicity classifier
MODEL_NAME = "martin-ha/toxic-comment-model"
classifier = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    # Depending on demoji version, download_codes() is either deprecated or automatic.
    # Yielding immediately prevents startup crashes.
    yield


app = FastAPI(lifespan=lifespan)

# Allow Chrome extension to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(request: PredictRequest):
    # Preprocessing
    # Replace emojis with text descriptions, separated by spaces
    processed_text = demoji.replace_with_desc(request.text, " ")
    processed_text = processed_text.lower()

    # If the text is empty after preprocessing, return safe directly
    if not processed_text.strip():
        return {"label": "non-toxic", "confidence": 1.0}

    # Inference using torch.no_grad() is handled automatically by the pipeline
    result = classifier(processed_text)[0]

    # Return label ("toxic" or "non-toxic") and score
    return {"label": result["label"], "confidence": result["score"]}


class ExplainRequest(BaseModel):
    text: str


@app.post("/explain")
async def explain(request: ExplainRequest):
    # Mocking Vector DB lookup and LLM inference
    await asyncio.sleep(1.5)

    # We can run the text through our small classifier again just to know whether to mock a "safe" or "toxic" explanation
    processed_text = demoji.replace_with_desc(request.text, " ").lower()

    if not processed_text.strip():
        return {
            "explanation": "The provided text is empty, so it does not contain any toxic elements."
        }

    result = classifier(processed_text)[0]
    is_toxic = result["label"] == "toxic"

    if is_toxic:
        explanations = [
            "Based on historical database matches, this text contains aggressive language and hostility typically associated with harassment. The phrasing suggests an intent to belittle or insult.",
            "This text triggers multiple toxicity flags. It closely matches patterns in our Vector DB involving explicit disrespect or hate speech toward individuals or groups.",
            "The phrasing used here is frequently flagged for toxicity due to its abrasive nature. It utilizes inflammatory words that violate standard community guidelines.",
        ]
    else:
        explanations = [
            "This text appears safe. It does not contain prominent markers of hate speech, profanity, or harassment according to historical conversational patterns.",
            "Our analysis indicates this is benign discussion. It lacks the aggressive syntax or derogatory terminology typically found in toxic content.",
            "The text is generally neutral or positive, functioning within the bounds of standard polite communication without any flagged inflammatory elements.",
        ]

    return {"explanation": random.choice(explanations)}
