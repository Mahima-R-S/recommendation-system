import os
from dataclasses import dataclass

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
TMDB_MOVIE_BASE = "https://www.themoviedb.org/movie"
TMDB_TV_BASE = "https://www.themoviedb.org/tv"


@dataclass
class Recommendation:
    title: str
    media_type: str
    overview: str
    genres: str
    release_year: str
    vote_average: float
    tmdb_url: str
    similarity_score: float
    explanation: str


class Recommender:
    def __init__(self, index_dir: str = "data/faiss_index"):
        self.index_dir = index_dir
        self.model = None
        self.index = None
        self.metadata = None
        self._loaded = False

    def load(self):
        """Load model and index into memory. Called once on first use."""
        if self._loaded:
            return

        print("Loading sentence transformer model...")
        self.model = SentenceTransformer(MODEL_NAME)

        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "metadata.pkl")

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Run: python -m src.embeddings.build first."
            )

        print("Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        self.metadata = pd.read_pickle(meta_path)
        self._loaded = True
        print(f"Ready. {self.index.ntotal} items indexed.")

    def _embed_query(self, query: str) -> np.ndarray:
        """Convert user's mood text into a vector."""
        vec = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vec.astype(np.float32)

    def _explain(self, query: str, row: pd.Series, score: float) -> str:


    # Split query into meaningful phrases
    # We split on "but", "and", "like", "," — common connectors in mood queries
        import re
        raw_phrases = re.split(r'\bbut\b|\band\b|\blike\b|,', query, flags=re.IGNORECASE)
        phrases = [p.strip() for p in raw_phrases if len(p.strip()) > 3]

        if not phrases:
            return f"semantic similarity ({score:.2f})"

    # Embed all phrases at once (batched, fast)
        phrase_vecs = self.model.encode(
            phrases,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    # Get this movie's vector from the FAISS index
    # reconstruct() fetches the stored vector by index position
        movie_idx = self.metadata.index[self.metadata["title"] == row["title"]].tolist()
        if not movie_idx:
            return f"semantic similarity ({score:.2f})"

        movie_vec = self.index.reconstruct(int(movie_idx[0]))  # shape: (384,)

    # Cosine similarity between each phrase vector and the movie vector
    # Both are normalized so dot product = cosine similarity
        phrase_scores = phrase_vecs @ movie_vec  # shape: (num_phrases,)

    # Rank phrases by how much they contributed
        ranked = sorted(zip(phrases, phrase_scores), key=lambda x: x[1], reverse=True)

    # Take top 2 contributing phrases
        top_phrases = [p for p, s in ranked[:2] if s > 0.1]

        if not top_phrases:
            return f"semantic similarity ({score:.2f})"

    # Format nicely
        contributions = " · ".join([
            f'"{p}" (match: {s:.2f})'
            for p, s in ranked[:2]
            if s > 0.1
        ])

        return f"driven by: {contributions}"

    def recommend(
        self,
        query: str,
        top_k: int = 10,
        filter_type: str = "all",   # "all", "movie", "tv"
        randomness: float = 0.0,    # 0 = pure ranking, >0 adds surprise
    ) -> list:
        self.load()

        query_vec = self._embed_query(query)

        # Fetch more than top_k to allow for filtering
        fetch_k = min(top_k * 5, self.index.ntotal)
        scores, indices = self.index.search(query_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue

            row = self.metadata.iloc[idx]

            # Apply type filter
            if filter_type != "all" and row.get("type") != filter_type:
                continue

            year = str(row.get("release_date", ""))[:4]
            media = row.get("type", "movie")
            tmdb_id = row.get("id", "")
            base_url = TMDB_TV_BASE if media == "tv" else TMDB_MOVIE_BASE

            results.append(
                Recommendation(
                    title=row.get("title", "Unknown"),
                    media_type="Series" if media == "tv" else "Movie",
                    overview=str(row.get("overview", "")),
                    genres=str(row.get("genres", "")),
                    release_year=year if year.isdigit() else "N/A",
                    vote_average=float(row.get("vote_average", 0)),
                    tmdb_url=f"{base_url}/{tmdb_id}",
                    similarity_score=float(score),
                    explanation=self._explain(query, row, float(score)),
                )
            )

            if len(results) >= top_k * 2:
                break

        # Add randomness for "surprise me" feature
        if randomness > 0 and results:
            noise = np.random.uniform(0, randomness, len(results))
            order = np.argsort(
                [-r.similarity_score + n for r, n in zip(results, noise)]
            )
            results = [results[i] for i in order]

        return results[:top_k]


# Singleton — Streamlit reruns the script on every interaction,
# this ensures the model loads only once instead of every time
_instance = None


def get_recommender(index_dir: str = "data/faiss_index") -> Recommender:
    global _instance
    if _instance is None:
        _instance = Recommender(index_dir)
        _instance.load()
    return _instance
