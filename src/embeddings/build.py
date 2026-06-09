import argparse
import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


def build_text_description(row: pd.Series) -> str:
    
    parts = []

    title = row.get("title", "")
    media_type = "series" if row.get("type") == "tv" else "film"
    parts.append(f"{title} is a {media_type}.")

    genres = row.get("genres", "")
    if genres:
        parts.append(f"It belongs to the {genres} genre.")

    overview = row.get("overview", "")
    if overview:
        parts.append(overview)

    rating = row.get("vote_average", 0)
    if rating >= 8.0:
        parts.append("It is critically acclaimed and highly rated.")
    elif rating >= 7.0:
        parts.append("It is well-regarded by audiences.")
    elif rating < 5.0:
        parts.append("It has mixed reviews.")

    year = str(row.get("release_date", ""))[:4]
    if year.isdigit():
        yr = int(year)
        if yr < 1990:
            parts.append("It is a classic older production.")
        elif yr < 2000:
            parts.append("It was made in the 1990s.")
        elif yr < 2010:
            parts.append("It was made in the 2000s.")
        elif yr >= 2020:
            parts.append("It is a recent production.")

    return " ".join(parts)


def build_index(input_csv: str, output_dir: str, batch_size: int = 64):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    df = df.dropna(subset=["overview"]).reset_index(drop=True)
    print(f"  {len(df)} items loaded.")

    print("Building text descriptions...")
    df["text_description"] = df.apply(build_text_description, axis=1)

    # Print a sample so you can see what the model actually reads
    print("\nSample description:")
    print(df["text_description"].iloc[0])
    print()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print("Encoding descriptions...")
    descriptions = df["text_description"].tolist()
    embeddings = model.encode(
        descriptions,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # normalizing means cosine similarity = dot product
    )

    print(f"Embedding shape: {embeddings.shape}")
    # Should print (4597, 384) — 4597 movies, 384 dimensions each

    # FAISS IndexFlatIP = exact search using inner product
    # on normalized vectors this equals cosine similarity
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"FAISS index built with {index.ntotal} vectors.")

    os.makedirs(output_dir, exist_ok=True)

    faiss.write_index(index, os.path.join(output_dir, "index.faiss"))

    meta_cols = ["id", "type", "title", "overview", "genres",
                 "release_date", "vote_average", "vote_count", "popularity"]
    df[meta_cols].to_pickle(os.path.join(output_dir, "metadata.pkl"))

    print(f"\n✅ Index saved to {output_dir}/")
    print(f"   index.faiss  — {index.ntotal} vectors, dim={dim}")
    print(f"   metadata.pkl — {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/movies.csv")
    parser.add_argument("--output", type=str, default="data/faiss_index")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    build_index(args.input, args.output, args.batch_size)