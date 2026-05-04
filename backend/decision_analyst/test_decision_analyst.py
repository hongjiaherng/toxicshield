"""Test cases for ClassificationExplainer - Toxicity classification."""

import sys, os, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decision_analyst.decision_analyst import ClassificationExplainer
from vectordb.vectordb import VectorDB
from vectordb.config import COLLECTION_NAME

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "log", "test_decision_analyst.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH, mode="w")
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)

db = VectorDB()
db.setup()


def _log(msg):
    logger.info(msg)


def test_toxic_explicit_insult():
    _log("[TEST] Toxic - Explicit Insult")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "You're such an idiot, nobody cares about your stupid opinion. Delete your account."
    predicted_label = "TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_toxic_threat():
    _log("\n[TEST] Toxic - Veiled Threat")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "Keep talking and see what happens to you. Watch your back."
    predicted_label = "TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_toxic_harassment():
    _log("\n[TEST] Toxic - Harassment")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "@user you're worthless and everyone hates you. Go away. Nobody wants you here."
    predicted_label = "TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_toxic_hate_slur():
    _log("\n[TEST] Toxic - Hate Speech")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "People like you don't deserve to exist. You're filth."
    predicted_label = "TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_non_toxic_disagreement():
    _log("\n[TEST] Non-Toxic - Respectful Disagreement")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "I strongly disagree with your point. Here's why I think you're wrong about this policy."
    predicted_label = "NON_TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_non_toxic_criticism():
    _log("\n[TEST] Non-Toxic - Constructive Criticism")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "This take is pretty bad tbh. The data doesn't support your conclusion at all."
    predicted_label = "NON_TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_non_toxic_casual():
    _log("\n[TEST] Non-Toxic - Casual Opinion")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "hot take: pineapple on pizza is actually fine? idk why everyone hates on it lol"
    predicted_label = "NON_TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


def test_non_toxic_news():
    _log("\n[TEST] Non-Toxic - News")

    explainer = ClassificationExplainer(model_name="glm-5")
    input_text = "Breaking: The bill passed with a 52-48 vote after 12 hours of debate."
    predicted_label = "NON_TOXIC"

    similar_examples = db.hybrid_search(input_text, k=3)
    result = explainer.explain(
        input_text=input_text,
        predicted_label=predicted_label,
        similar_examples=similar_examples,
    )
    _log(f"  {'Input':<35} {result.input_text}")
    _log(f"  {'Label':<35} {result.predicted_label}")
    _log(f"  {'Similar Examples':<35}")
    for ex in similar_examples:
        _log(f"    - [{ex['category']}] {ex['text'][:60]}...")
    _log(f"  {'Explanation':<35} {result.justification}")


if __name__ == "__main__":
    _log("=" * 60)
    _log("CLASSIFICATION EXPLAINER - TOXICITY TEST SUITE")
    _log("=" * 60)

    tests = [
        test_toxic_explicit_insult,
        test_toxic_threat,
        test_toxic_harassment,
        test_toxic_hate_slur,
        test_non_toxic_disagreement,
        test_non_toxic_criticism,
        test_non_toxic_casual,
        test_non_toxic_news,
    ]

    passed = 0
    failed = 0
    rows = []

    for test in tests:
        try:
            test()
            passed += 1
            rows.append((test.__name__, "PASS", ""))
        except Exception as e:
            failed += 1
            rows.append((test.__name__, "FAIL", str(e)))

    _log(f"\n{'─' * 60}")
    _log(f"{'Test':<35} {'Status':^8} {'Details'}")
    _log(f"{'─' * 60}")
    for name, status, detail in rows:
        mark = "✓" if status == "PASS" else "✗"
        _log(f"  {mark} {name:<33} {status:^8} {detail}")
    _log(f"{'─' * 60}")
    _log(f"  {passed} passed, {failed} failed")
    _log("=" * 60)

    sys.exit(0 if failed == 0 else 1)
