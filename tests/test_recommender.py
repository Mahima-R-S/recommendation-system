import pandas as pd
import pytest

from src.recommender.engine import Recommendation, Recommender


# ── Sample data fixture ───────────────────────────────────────────────────────
# A fixture is reusable test data. pytest injects it automatically
# into any test function that has a matching parameter name.

@pytest.fixture
def sample_row():
    return pd.Series({
        "id": 1,
        "type": "movie",
        "title": "The Dark Knight",
        "overview": "A dark and gritty superhero film about Batman facing the Joker.",
        "genres": "Action, Crime, Drama",
        "release_date": "2008-07-18",
        "vote_average": 9.0,
        "vote_count": 30000,
        "popularity": 100.0,
    })


@pytest.fixture
def sample_metadata():
    return pd.DataFrame([
        {
            "id": 1,
            "type": "movie",
            "title": "The Dark Knight",
            "overview": "A dark and gritty superhero film about Batman facing the Joker.",
            "genres": "Action, Crime, Drama",
            "release_date": "2008-07-18",
            "vote_average": 9.0,
            "vote_count": 30000,
            "popularity": 100.0,
        },
        {
            "id": 2,
            "type": "tv",
            "title": "Breaking Bad",
            "overview": "A chemistry teacher turns to drug manufacturing after a cancer diagnosis.",
            "genres": "Crime, Drama, Thriller",
            "release_date": "2008-01-20",
            "vote_average": 9.5,
            "vote_count": 12000,
            "popularity": 90.0,
        },
    ])


# ── Recommendation dataclass tests ────────────────────────────────────────────

def test_recommendation_fields():
    """Recommendation dataclass stores all fields correctly."""
    r = Recommendation(
        title="Inception",
        media_type="Movie",
        overview="A thief enters dreams.",
        genres="Sci-Fi, Thriller",
        release_year="2010",
        vote_average=8.8,
        tmdb_url="https://www.themoviedb.org/movie/27205",
        similarity_score=0.85,
        explanation='driven by: "sci-fi" (match: 0.72)',
    )
    assert r.title == "Inception"
    assert r.similarity_score == 0.85
    assert "driven by" in r.explanation


def test_recommendation_score_range():
    """Similarity score should be between 0 and 1 for normalized vectors."""
    r = Recommendation(
        title="Test", media_type="Movie", overview="x",
        genres="Drama", release_year="2020",
        vote_average=7.0, tmdb_url="http://x.com",
        similarity_score=0.73, explanation="test",
    )
    assert 0.0 <= r.similarity_score <= 1.0


# ── Recommender error handling ────────────────────────────────────────────────

def test_recommender_missing_index(monkeypatch):
    """Should raise FileNotFoundError if FAISS index doesn't exist."""
    monkeypatch.setattr(
        "src.recommender.engine.SentenceTransformer",
        lambda *a, **kw: None,
    )
    rec = Recommender(index_dir="/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        rec.load()


# ── Text description builder tests ───────────────────────────────────────────

def test_description_contains_title():
    from src.embeddings.build import build_text_description
    row = pd.Series({
        "title": "Inception", "type": "movie",
        "genres": "Sci-Fi, Thriller",
        "overview": "A thief who steals corporate secrets through dreams.",
        "vote_average": 8.8, "release_date": "2010-07-16",
    })
    desc = build_text_description(row)
    assert "Inception" in desc


def test_description_identifies_tv():
    from src.embeddings.build import build_text_description
    row = pd.Series({
        "title": "The Bear", "type": "tv",
        "genres": "Drama, Comedy",
        "overview": "A chef returns home to run his family restaurant.",
        "vote_average": 8.5, "release_date": "2022-06-23",
    })
    desc = build_text_description(row)
    assert "series" in desc.lower()


def test_description_marks_classic():
    from src.embeddings.build import build_text_description
    row = pd.Series({
        "title": "Casablanca", "type": "movie",
        "genres": "Drama, Romance",
        "overview": "A cynical American runs a nightclub in Casablanca during WWII.",
        "vote_average": 8.5, "release_date": "1942-11-26",
    })
    desc = build_text_description(row)
    assert "classic" in desc.lower()


def test_description_marks_acclaimed():
    from src.embeddings.build import build_text_description
    row = pd.Series({
        "title": "Perfect Film", "type": "movie",
        "genres": "Drama",
        "overview": "An outstanding cinematic achievement.",
        "vote_average": 8.5, "release_date": "2020-01-01",
    })
    desc = build_text_description(row)
    assert "acclaimed" in desc.lower() or "well-regarded" in desc.lower()