"""
Test suite for Hybrid Toxicity Vector Database.
Validates embedding creation, insertion, and retrieval.
"""

import sys, os, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectordb import VectorDB
from config import COLLECTION_NAME

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "log", "test_vectordb.log")

os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH, mode="w")
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)


def _log(msg):
    logger.info(msg)


# 50 samples: 25 toxic, 25 non-toxic
SAMPLES = [
    # --- TOXIC SAMPLES (25) ---
    # Insults and Flaming
    {"text": "You are an absolute failure at everything you do.", "is_toxic": True, "category": "Insults and Flaming"},
    {"text": "Nobody cares about your stupid opinion, loser.", "is_toxic": True, "category": "Insults and Flaming"},
    {"text": "Keep crying, you pathetic waste of space.", "is_toxic": True, "category": "Insults and Flaming"},
    {"text": "Imagine being this dumb and still posting online lol", "is_toxic": True, "category": "Insults and Flaming"},
    {"text": "You're genuinely the worst person I've ever interacted with.", "is_toxic": True, "category": "Insults and Flaming"},

    # Hate and Harassment
    {"text": "People like you shouldn't be allowed to have an opinion.", "is_toxic": True, "category": "Hate and Harassment"},
    {"text": "Go back to where you came from, we don't want you here.", "is_toxic": True, "category": "Hate and Harassment"},
    {"text": "You're disgusting and everyone thinks so.", "is_toxic": True, "category": "Hate and Harassment"},
    {"text": "Stop harassing people with your presence.", "is_toxic": True, "category": "Hate and Harassment"},
    {"text": "Your entire existence is a mistake.", "is_toxic": True, "category": "Hate and Harassment"},

    # Threats
    {"text": "I'm going to find you and make you regret this.", "is_toxic": True, "category": "Threats"},
    {"text": "You better watch your back from now on.", "is_toxic": True, "category": "Threats"},
    {"text": "Something bad is coming your way very soon.", "is_toxic": True, "category": "Threats"},
    {"text": "I know where you live, keep that in mind.", "is_toxic": True, "category": "Threats"},
    {"text": "You're going to pay for what you said.", "is_toxic": True, "category": "Threats"},

    # Extremism
    {"text": "The only solution is to destroy the entire system by any means.", "is_toxic": True, "category": "Extremism"},
    {"text": "Violence is the only language these people understand.", "is_toxic": True, "category": "Extremism"},
    {"text": "We need to take extreme measures to cleanse society.", "is_toxic": True, "category": "Extremism"},
    {"text": "There's no peaceful way forward anymore, burn it all.", "is_toxic": True, "category": "Extremism"},
    {"text": "Radical action is the only path to true change.", "is_toxic": True, "category": "Extremism"},

    # Other Offensive Texts
    {"text": "That's so r*tarded, I can't even.", "is_toxic": True, "category": "Other Offensive Texts"},
    {"text": "You look like a complete idiot in that photo.", "is_toxic": True, "category": "Other Offensive Texts"},
    {"text": "Kill yourself, no one would miss you.", "is_toxic": True, "category": "Other Offensive Texts"},
    {"text": "This is the dumbest thing I've read all day, goodbye.", "is_toxic": True, "category": "Other Offensive Texts"},
    {"text": "I hope something terrible happens to you soon.", "is_toxic": True, "category": "Other Offensive Texts"},

    # --- NON-TOXIC SAMPLES (25) ---
    {"text": "I respectfully disagree with your point about the policy.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Great work on this project! Very well organized.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Could you clarify what you mean by that statement?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I had a different experience, but I appreciate your perspective.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "This tutorial was really helpful, thank you for sharing.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I think there might be a misunderstanding here.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Interesting take! I hadn't considered that angle before.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Can someone explain how this feature works?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I'm looking forward to the next update.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Thanks for the feedback, I'll look into it.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "That's a valid concern that deserves attention.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I learned something new today from this discussion.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Would you mind sharing your sources for this?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I see where you're coming from, even if I disagree.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "This is exactly what I was looking for, thanks!", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Has anyone else encountered this issue before?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I appreciate you taking the time to explain.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Neutral question: what are the alternatives here?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Just wanted to say this community is great.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I'm curious about the reasoning behind this decision.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "That's a creative solution to the problem.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Thanks for your patience while I figure this out.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Good point, I'll keep that in mind going forward.", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "Is there documentation available for this?", "is_toxic": False, "category": "Non-Toxic"},
    {"text": "I'm happy to help if anyone needs assistance.", "is_toxic": False, "category": "Non-Toxic"},
]


def test_embedding_models():
    """Test that FastEmbed models load and create embeddings."""
    _log("[TEST] Embedding Models Initialization")

    vdb = _check_vectordb()

    test_texts = ["Hello world", "This is a test"]
    dense_embeddings = list(vdb.dense_model.embed(test_texts))

    assert len(dense_embeddings) == 2, "Should generate 2 dense embeddings"
    assert len(dense_embeddings[0]) == vdb.dense_size, "Dense dimension mismatch"

    sparse_embeddings = list(vdb.sparse_model.embed(test_texts))
    assert len(sparse_embeddings) == 2, "Should generate 2 sparse embeddings"

    _log(f"  {'Check':<35} {'Result'}")
    _log(f"  {'─' * 55}")
    _log(f"  {'Dense vector size':<35} {vdb.dense_size}")
    _log(f"  {'Sparse embeddings generated':<35} OK")


def test_collection_setup():
    """Test collection exists with hybrid config."""
    _log("\n[TEST] Collection Setup (Hybrid)")

    vdb = _check_vectordb()

    assert vdb.client.collection_exists(COLLECTION_NAME), "Collection should exist"

    collection_info = vdb.client.get_collection(COLLECTION_NAME)
    assert "dense" in collection_info.config.params.vectors, "Dense vector config missing"
    assert "sparse" in collection_info.config.params.sparse_vectors, "Sparse vector config missing"

    _log(f"  {'Check':<35} {'Result'}")
    _log(f"  {'─' * 55}")
    _log(f"  {'Collection exists':<35} {COLLECTION_NAME}")
    _log(f"  {'Dense vectors':<35} COSINE")
    _log(f"  {'Sparse vectors':<35} BM25")


def test_sample_insertion():
    """Test insertion of SAMPLES and verify data exists in the collection."""
    _log("\n[TEST] Sample Insertion Check")

    vdb = _check_vectordb()

    vdb.insert(SAMPLES)

    collection_info = vdb.client.get_collection(COLLECTION_NAME)
    count = collection_info.points_count

    assert count == len(SAMPLES), f"Expected {len(SAMPLES)} points, got {count}"

    results = vdb.client.scroll(collection_name=COLLECTION_NAME, limit=5)[0]
    for point in results:
        assert "text" in point.payload, "Payload missing 'text'"
        assert "is_toxic" in point.payload, "Payload missing 'is_toxic'"
        assert "category" in point.payload, "Payload missing 'category'"

    _log(f"  {'Check':<35} {'Result'}")
    _log(f"  {'─' * 55}")
    _log(f"  {'Inserted samples':<35} {len(SAMPLES)} (25 toxic + 25 non-toxic)")
    _log(f"  {'Collection point count':<35} {count}")
    _log(f"  {'Payload fields':<35} text, is_toxic, category")


def _check_vectordb():
    """Create a VectorDB instance and ensure collection exists."""
    vdb = VectorDB(cloud_inference=False)
    vdb.setup()
    return vdb


def test_hybrid_retrieval():
    """Test hybrid retrieval with RRF fusion."""
    _log("\n[TEST] Hybrid Context Retrieval")

    vdb = _check_vectordb()

    query = "You are a complete idiot and nobody likes you"
    results = vdb.hybrid_search(query, k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    for result in results:
        assert "text" in result, "Result missing 'text'"
        assert "category" in result, "Result missing 'category'"
        assert "score" in result, "Result missing 'score'"
        assert result["category"] != "Non-Toxic", "Should not return non-toxic results"

    _log(f"  {'Query':<35} \"{query}\"")
    _log(f"  ✓ Retrieved {len(results)} toxic results (hybrid RRF)")
    _log(f"  {'─' * 55}")
    _log(f"  {'#':<3} {'Score':<12} {'Category':<22} {'Text'}")
    _log(f"  {'─' * 55}")
    for i, r in enumerate(results[:3], 1):
        _log(f"  {i:<3} {r['score']:<12.4f} {r['category']:<22} {r['text']}")


def test_dense_only_retrieval():
    """Test dense-only retrieval."""
    _log("\n[TEST] Dense-Only Retrieval")

    vdb = _check_vectordb()

    query = "You're disgusting and terrible"
    results = vdb.dense_search(query, k=3)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    _log(f"  {'Query':<35} \"{query}\"")
    _log(f"  ✓ Dense-only retrieval works")
    _log(f"  {'─' * 55}")
    _log(f"  {'#':<3} {'Score':<12} {'Category':<22} {'Text'}")
    _log(f"  {'─' * 55}")
    for i, r in enumerate(results, 1):
        _log(f"  {i:<3} {r['score']:<12.4f} {r['category']:<22} {r['text']}")


def test_sparse_only_retrieval():
    """Test sparse-only retrieval."""
    _log("\n[TEST] Sparse-Only Retrieval")

    vdb = _check_vectordb()

    query = "stupid idiot loser"
    results = vdb.sparse_search(query, k=3)

    assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"

    _log(f"  {'Query':<35} \"{query}\"")
    _log(f"  ✓ Sparse-only retrieval works ({len(results)} results)")
    _log(f"  {'─' * 55}")
    _log(f"  {'#':<3} {'Score':<12} {'Category':<22} {'Text'}")
    _log(f"  {'─' * 55}")
    for i, r in enumerate(results, 1):
        _log(f"  {i:<3} {r['score']:<12.4f} {r['category']:<22} {r['text']}")


def test_exact_match():
    """Test that exact toxic queries return similar content."""
    _log("\n[TEST] Exact Match Retrieval")

    vdb = _check_vectordb()

    query = "You are an absolute failure at everything you do."
    results = vdb.hybrid_search(query, k=1)

    assert len(results) == 1, "Should return 1 result"
    assert "failure" in results[0]["text"].lower(), "Should return the matching toxic text"

    _log(f"  {'Query':<35} \"{query}\"")
    _log(f"  {'─' * 55}")
    _log(f"  {'Match found':<35} {results[0]['text']}")
    _log(f"  {'Category':<35} {results[0]['category']}")
    _log(f"  {'Score':<35} {results[0]['score']:.4f}")


def test_filter_enforcement():
    """Test that filter only returns toxic results."""
    _log("\n[TEST] Filter Enforcement")

    vdb = _check_vectordb()

    query = "I really appreciate your help, thank you so much!"
    results = vdb.hybrid_search(query, k=3)

    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    for r in results:
        assert r["category"] != "Non-Toxic", "Filter failed - returned non-toxic"

    _log(f"  {'Query':<35} \"{query}\"")
    _log(f"  ✓ Filter enforced: non-toxic query still returns only toxic results")


def run_all_tests():
    """Run all tests."""
    _log("=" * 60)
    _log("HYBRID TOXICITY VDB TEST SUITE")
    _log("=" * 60)

    tests = [
        test_embedding_models,
        test_collection_setup,
        test_sample_insertion,
        test_hybrid_retrieval,
        test_dense_only_retrieval,
        test_sparse_only_retrieval,
        test_exact_match,
        test_filter_enforcement,
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

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
