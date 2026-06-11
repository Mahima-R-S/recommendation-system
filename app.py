import os

import streamlit as st

from src.recommender.engine import get_recommender

st.set_page_config(page_title="MoodMatch", page_icon="🎬", layout="centered")

st.title("🎬 MoodMatch")
st.markdown("Describe what you want to watch in plain English.")

query = st.text_input("Your mood", placeholder="e.g. something light and funny like The Office")

filter_type = st.radio(
    "Show me",
    ["Movies & Series", "Movies only", "Series only"],
    horizontal=True,
)

type_map = {"Movies & Series": "all", "Movies only": "movie", "Series only": "tv"}

if st.button("Find matches", type="primary") and query.strip():
    with st.spinner("Searching..."):
        rec = get_recommender(os.getenv("FAISS_INDEX_DIR", "data/faiss_index"))
        results = rec.recommend(
            query=query.strip(),
            top_k=10,
            filter_type=type_map[filter_type],
        )

    if not results:
        st.warning("No results found. Try a different query.")
    else:
        for i, r in enumerate(results):
            st.markdown(f"**{i+1}. {r.title}** ({r.release_year}) — {r.media_type}")
            st.markdown(f"⭐ {r.vote_average:.1f} · {r.genres}")
            st.markdown(f"{r.overview[:250]}…")
            st.markdown(f"🔍 *{r.explanation}*")
            st.markdown(f"[View on TMDB ↗]({r.tmdb_url})")
            st.divider()
