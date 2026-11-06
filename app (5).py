"""Streamlit demo: predict the audience rating category of a Netflix title.

Run locally:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import config as C
from src.inference import available_options, load_bundle, make_title_row, predict_rows
from src.plot_style import apply_theme

st.set_page_config(page_title="Netflix Audience Rating Predictor", page_icon="🎬", layout="wide")

CATEGORY_EMOJI = {"Kids": "🟢", "Family": "🟡", "Teens": "🔵", "Adults": "🔴"}
CATEGORY_HELP = {
    "Kids": "TV-Y, TV-Y7, TV-G, G",
    "Family": "TV-PG, PG",
    "Teens": "TV-14, PG-13",
    "Adults": "TV-MA, R, NC-17",
}


@st.cache_resource
def get_bundle():
    return load_bundle()


@st.cache_data(show_spinner=False)
def run_prediction(args: tuple) -> pd.DataFrame:
    content_type, title, duration, countries, genres, rel_year, added, has_dir = args
    row = make_title_row(content_type, title, duration, list(countries), list(genres), rel_year, added, has_dir)
    return predict_rows(row, get_bundle())


bundle = get_bundle()
options = available_options(bundle)

st.markdown("# 🎬 Netflix Audience Rating Predictor")
st.caption("A tuned Random Forest predicts whether a title is aimed at **Kids, Family, Teens or Adults** "
           "from its content attributes. Change the inputs on the left and press **Predict**.")

with st.sidebar:
    st.header("Describe the title")
    title = st.text_input("Title (optional)", "My New Netflix Title")
    content_type = st.radio("Type", ["Movie", "TV Show"], horizontal=True)
    if content_type == "Movie":
        duration_value = st.slider("Runtime (minutes)", 40, 240, 100)
    else:
        duration_value = st.slider("Number of seasons", 1, 15, 2)
    genres = st.multiselect("Genres", options["genres"], default=["Dramas"] if "Dramas" in options["genres"] else None)
    countries = st.multiselect("Countries", options["countries"], default=["United States"])
    release_year = st.slider("Release year", 1990, 2021, 2019)
    lo = max(release_year, 2008)
    year_added = st.number_input("Year added to Netflix", min_value=lo, max_value=2021, value=min(max(2020, lo), 2021), step=1)
    has_director = st.checkbox("Director is listed", value=True)
    go = st.button("Predict", type="primary", use_container_width=True)

if go or "last" not in st.session_state:
    st.session_state["last"] = run_prediction(
        (content_type, title, duration_value, tuple(countries), tuple(genres),
         release_year, int(year_added), has_director)
    )

result = st.session_state["last"].iloc[0]
pred = result["predicted_category"]
probs = pd.Series({c: result[f"p_{c}"] for c in C.CATEGORY_ORDER})

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Prediction")
    st.header(f"{CATEGORY_EMOJI[pred]} {pred}")
    st.write(f"Typical ratings: **{CATEGORY_HELP[pred]}**")
    st.metric("Model confidence", f"{probs[pred]:.0%}")
    if probs.sort_values(ascending=False).iloc[0] < 0.5:
        st.info("The model is not very sure about this one. Neighbouring categories are close, "
                "which is also where most of its mistakes happen on the test set.")

with right:
    st.subheader("Probability for each category")
    apply_theme()
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.barh(probs.index[::-1], probs.values[::-1], color=[C.CATEGORY_COLORS[c] for c in probs.index[::-1]])
    ax.bar_label(bars, labels=[f"{v:.0%}" for v in probs.values[::-1]], padding=4, color=C.LIGHT)
    ax.set_xlim(0, 1); ax.grid(axis="y", visible=False); ax.set_xlabel("Probability")
    fig.tight_layout()
    st.pyplot(fig)

st.divider()
with st.expander("About this model"):
    st.markdown(
        "- **Data:** 8,708 rated Netflix titles, 53 content features (no rating information used)\n"
        "- **Model:** Random Forest tuned with randomized search, scored on macro-F1\n"
        "- **Test accuracy:** about 63%, macro-F1 about 0.61 on 1,742 unseen titles\n"
        "- **Limit:** real ratings depend on violence and language, which are not in the data, "
        "so treat this as an estimate, not an official rating."
    )
