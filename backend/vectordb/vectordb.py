from qdrant_client import QdrantClient, models
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue,
    SparseVectorParams, SparseVector, SparseIndexParams
)
from fastembed import TextEmbedding, SparseTextEmbedding
from logging.handlers import RotatingFileHandler
from typing import List, Dict
from config import COLLECTION_NAME, DENSE_MODEL, SPARSE_MODEL, DENSE_EMBEDDING_SIZE
import os, logging

# load environmental variables
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "log", "vectordb.log")

# initialize the logger
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        RotatingFileHandler(LOG_PATH, maxBytes=5_000_000),
        logging.StreamHandler()
    ])
logger = logging.getLogger(__name__)


class VectorDB:
    """
    Hybrid vector database using dense + sparse embeddings.
    """

    def __init__(self, cloud_inference=False):
        self.cloud_inference = cloud_inference
        self.dense_size = DENSE_EMBEDDING_SIZE

        # Initialize models locally ONLY if NOT using cloud inference
        if not cloud_inference: 
            self.dense_model = TextEmbedding(model_name=DENSE_MODEL)
            self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

        # Initialize Qdrant client with cloud cluster
        try:
            self.client = QdrantClient(
                api_key=os.getenv("QDRANT_API_KEY"),
                url=os.getenv("QDRANT_ENDPOINT"),
                cloud_inference=cloud_inference
            )
            logger.info("Qdrant client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            raise


    # ---------------------------
    # Setup
    # ---------------------------

    def setup(self):
        """
        Create collection with hybrid config if not found.
        """

        try:
            if self.client.collection_exists(COLLECTION_NAME):
                logger.info(f"Connected to existing collection: '{COLLECTION_NAME}'")
            else:
                self.client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config={
                        "dense": VectorParams(
                            size=DENSE_EMBEDDING_SIZE,
                            distance=Distance.COSINE,
                        )
                    },
                    sparse_vectors_config={
                        "sparse": SparseVectorParams(
                            index=SparseIndexParams(on_disk=False),
                        )
                    }
                )

                # Index for the toxicity flag (Boolean)
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="is_toxic",
                    field_schema=models.PayloadSchemaType.BOOL,
                )
                
                # Index for the category (Keyword)
                self.client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name="category",
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )

                logger.info(f"Created new collection: '{COLLECTION_NAME}'")
        except Exception as e:
            logger.error(f"Failed to set up collection '{COLLECTION_NAME}': {e}")
            raise


    # ---------------------------
    # Insert
    # ---------------------------

    def insert(self, samples: list[Dict]):
        """
        Insert samples using either local embeddings or cloud inference.
        """
        try:
            points = self._build_points(samples)

            kwargs = {
                "collection_name": COLLECTION_NAME,
                "points": points,
            }

            self.client.upsert(**kwargs)
            logger.info(f"Inserted {len(samples)} samples.")

        except Exception as e:
            logger.error(f"Insert failed: {e}")
            raise

    def _build_points(self, samples: list[dict]) -> list[PointStruct]:
        """
        Unified builder that delegates vector generation.
        """
        if self.cloud_inference:
            return self._build_points_cloud(samples)
        return self._build_points_local(samples)

    def _build_points_local(self, samples: list[dict]):
        """
        Build points with locally generated dense + sparse embeddings.
        """

        texts = [s["text"] for s in samples]

        dense_embeddings = list(self.dense_model.embed(texts))
        sparse_embeddings = list(self.sparse_model.embed(texts))

        points = []
        for idx, (sample, dense_vec, sparse_vec) in enumerate(
            zip(samples, dense_embeddings, sparse_embeddings)
        ):
            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "dense": dense_vec.tolist(),
                        "sparse": SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload=self._build_payload(sample)
                )
            )
        return points

    def _build_points_cloud(self, samples: list[dict]):
        """
        Build points without vectors (Qdrant will infer embeddings).
        """

        points = []
        for idx, sample in enumerate(samples):
            points.append(
                PointStruct(
                    id=idx,
                    vector={
                        "dense": models.Document(
                            text=sample["text"],
                            model=DENSE_MODEL
                        ),
                        "sparse": models.Document(
                            text=sample["text"],
                            model=SPARSE_MODEL
                        )
                    },
                    payload=self._build_payload(sample)
                )
            )
        return points


    # ---------------------------
    # Search APIs
    # ---------------------------

    def dense_search(self, query_text:str, k:int = 3) -> List[Dict[str, str]]:
        query = self._get_query_vector(query_text, mode="dense")

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query,
            using="dense",
            query_filter=Filter(
                must=[FieldCondition(key="is_toxic", match=MatchValue(value=True))]
            ),
            limit=k
        )

        return self._format_response(response)

    def sparse_search(self, query_text:str, k:int = 3) -> List[Dict[str, str]]:
        query = self._get_query_vector(query_text, mode="sparse")

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query,
            using="sparse",
            query_filter=Filter(
                must=[FieldCondition(key="is_toxic", match=MatchValue(value=True))]
            ),
            limit=k
        )

        return self._format_response(response)

    def hybrid_search(self, query_text:str, k:int = 3) -> List[Dict[str, str]]:
        dense_query = self._get_query_vector(query_text, mode="dense")
        sparse_query = self._get_query_vector(query_text, mode="sparse")

        response = self.client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                models.Prefetch(
                    query=sparse_query,
                    using="sparse",
                    filter=Filter(
                        must=[FieldCondition(key="is_toxic", match=MatchValue(value=True))]
                    ),
                    limit=20,
                ),
                models.Prefetch(
                    query=dense_query,
                    using="dense",
                    filter=Filter(
                        must=[FieldCondition(key="is_toxic", match=MatchValue(value=True))]
                    ),
                    limit=20,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=k
        )

        return self._format_response(response)
    

    # ---------------------------
    # Query Vector Strategy
    # ---------------------------

    def _get_query_vector(self, query_text: str, mode: str):
        """
        Unified query vector generator.
        """
        if self.cloud_inference:
            return query_text

        if mode == "dense":
            return list(self.dense_model.embed([query_text]))[0].tolist()

        if mode == "sparse":
            emb = list(self.sparse_model.embed([query_text]))[0]
            return SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )

        raise ValueError(f"Unsupported mode: {mode}")


    # ---------------------------
    # Formatter
    # ---------------------------

    def _build_payload(self, sample):
        return {
            "text": sample["text"],
            "is_toxic": sample["is_toxic"],
            "category": sample["category"],
        }

    def _format_response(self, response):
        return [
            {
                "text": hit.payload["text"],
                "category": hit.payload["category"],
                "score": hit.score,
            }
            for hit in response.points
        ]


if __name__ == "__main__":
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

    vectordb = VectorDB(cloud_inference=False)
    vectordb.setup()
    # vectordb.insert(SAMPLES)

    query = "You are an absolute failure at everything you do."
    print(f"\n{'='*60}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*60}")

    # Hybrid results
    print("\n--- HYBRID SEARCH ---")
    results = vectordb.hybrid_search(query, k=3)
    for i, r in enumerate(results, 1):
        print(f"[{i}] (score: {r['score']}) [{r['category']}]")
        print(f"    {r['text']}")

    # Dense only
    print("\n--- DENSE ONLY ---")
    dense_results = vectordb.dense_search(query, k=3)
    for i, r in enumerate(dense_results, 1):
        print(f"[{i}] [{r['category']}] {r['text'][:50]}...")

    # Sparse only
    print("\n--- SPARSE ONLY ---")
    sparse_results = vectordb.sparse_search(query, k=3)
    for i, r in enumerate(sparse_results, 1):
        print(f"[{i}] [{r['category']}] {r['text'][:50]}...")