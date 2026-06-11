import argparse
import os
import time

import pandas as pd
import requests
import urllib3
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

TMDB_BASE = "https://api.themoviedb.org/3"


def make_session() -> requests.Session:
    """Create a session with automatic retries on connection failures."""
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = make_session()


def get_genres(api_key: str) -> dict:
    """Fetching genre id → name mapping from TMDB."""
    url = f"{TMDB_BASE}/genre/movie/list"
    r = SESSION.get(url, params={"api_key": api_key, "language": "en-US"}, timeout=10, verify=False)
    r.raise_for_status()
    genres = {g["id"]: g["name"] for g in r.json()["genres"]}

    url_tv = f"{TMDB_BASE}/genre/tv/list"
    r2 = SESSION.get(url_tv, params={"api_key": api_key, "language": "en-US"}, timeout=10, verify=False)
    r2.raise_for_status()
    genres.update({g["id"]: g["name"] for g in r2.json()["genres"]})
    return genres


def fetch_movies_page(api_key: str, page: int) -> list:
    """Fetching one page of popular movies (20 items)."""
    url = f"{TMDB_BASE}/discover/movie"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "include_adult": False,
        "page": page,
        "vote_count.gte": 50,
    }
    r = SESSION.get(url, params=params, timeout=10, verify=False)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_tv_page(api_key: str, page: int) -> list:
    """Fetching one page of popular TV series (20 items)."""
    url = f"{TMDB_BASE}/discover/tv"
    params = {
        "api_key": api_key,
        "language": "en-US",
        "sort_by": "popularity.desc",
        "page": page,
        "vote_count.gte": 50,
    }
    r = SESSION.get(url, params=params, timeout=10, verify=False)
    r.raise_for_status()
    return r.json().get("results", [])


def parse_movie(item: dict, genres: dict) -> dict:
    genre_names = [genres.get(gid, "") for gid in item.get("genre_ids", [])]
    return {
        "id": item["id"],
        "type": "movie",
        "title": item.get("title", ""),
        "overview": item.get("overview", ""),
        "genres": ", ".join(filter(None, genre_names)),
        "release_date": item.get("release_date", ""),
        "vote_average": item.get("vote_average", 0),
        "vote_count": item.get("vote_count", 0),
        "popularity": item.get("popularity", 0),
        "original_language": item.get("original_language", ""),
    }


def parse_tv(item: dict, genres: dict) -> dict:
    genre_names = [genres.get(gid, "") for gid in item.get("genre_ids", [])]
    return {
        "id": item["id"],
        "type": "tv",
        "title": item.get("name", ""),
        "overview": item.get("overview", ""),
        "genres": ", ".join(filter(None, genre_names)),
        "release_date": item.get("first_air_date", ""),
        "vote_average": item.get("vote_average", 0),
        "vote_count": item.get("vote_count", 0),
        "popularity": item.get("popularity", 0),
        "original_language": item.get("original_language", ""),
    }


def collect(api_key: str, pages: int = 250, output: str = "data/movies.csv"):
    print("Fetching genre map...")
    genres = get_genres(api_key)

    records = []
    half = pages // 2  # split pages between movies and TV

    print(f"Collecting {half} pages of movies (~{half * 20} items)...")
    for page in tqdm(range(1, half + 1)):
        try:
            items = fetch_movies_page(api_key, page)
            records.extend([parse_movie(i, genres) for i in items if i.get("overview")])
            time.sleep(0.1)  # respect rate limits
        except requests.HTTPError as e:
            print(f"  Skipping page {page}: {e}")

    print(f"Collecting {half} pages of TV series (~{half * 20} items)...")
    for page in tqdm(range(1, half + 1)):
        try:
            items = fetch_tv_page(api_key, page)
            records.extend([parse_tv(i, genres) for i in items if i.get("overview")])
            time.sleep(0.1)
        except requests.HTTPError as e:
            print(f"  Skipping page {page}: {e}")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["id", "type"])
    df = df[df["overview"].str.len() > 30]
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\n✅ Saved {len(df)} items to {output}")
    print(df[["type", "title", "genres", "vote_average"]].head(10).to_string())
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int, default=250)
    parser.add_argument("--output", type=str, default="data/movies.csv")
    args = parser.parse_args()

    api_key = os.getenv("TMDB_API_KEY")
    if not api_key:
        raise ValueError("TMDB_API_KEY not set. Add it to your .env file.")

    collect(api_key, pages=args.pages, output=args.output)
