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
        # logging.StreamHandler()
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
        try:
            logger.info(f"Dense search | query='{query_text}' k={k} cloud_inference={self.cloud_inference}")
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

            results = self._format_response(response)
            logger.info(f"Dense search returned {len(results)} results.")
            return results
        
        except Exception as e:
            logger.error(f"Dense search failed | query='{query_text}': {e}", exc_info=True)
            raise

    def sparse_search(self, query_text:str, k:int = 3) -> List[Dict[str, str]]:
        try:
            logger.info(f"Dense search | query='{query_text}' k={k} cloud_inference={self.cloud_inference}")
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

            results = self._format_response(response)
            logger.info(f"Sparse search returned {len(results)} results.")
            return results
        
        except Exception as e:
            logger.error(f"Sparse search failed | query='{query_text}': {e}", exc_info=True)
            raise

    def hybrid_search(self, query_text:str, k:int = 3) -> List[Dict[str, str]]:
        try:
            logger.info(f"Dense search | query='{query_text}' k={k} cloud_inference={self.cloud_inference}")
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

            results = self._format_response(response)
            logger.info(f"Hybrid search returned {len(results)} results.")
            return results
        
        except Exception as e:
            logger.error(f"Hybrid search failed | query='{query_text}': {e}", exc_info=True)
            raise
    

    # ---------------------------
    # Query Vector Strategy
    # ---------------------------

    def _get_query_vector(self, query_text: str, mode: str):
        """
        Unified query vector generator.
        """

        if mode == "dense":
            if self.cloud_inference:
                return models.Document(text=query_text, model=DENSE_MODEL)
            
            vector = list(self.dense_model.embed([query_text]))[0]
            return [float(x) for x in vector]

        if mode == "sparse":
            emb = list(self.sparse_model.embed([query_text]))[0]
            return SparseVector(
                indices=emb.indices.tolist(),
                values=emb.values.tolist(),
            )


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