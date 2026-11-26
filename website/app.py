from __future__ import annotations

import hashlib
import json
import logging
import math
import base64
from html import escape
from pathlib import Path

import streamlit as st

from spoiler_shield.catalog import CatalogError, Movie, load_catalog
from spoiler_shield.model_adapter import Prediction
from spoiler_shield.distilbert_adapter import DistilbertSpoilerPredictor, MAX_REVIEW_SENTENCES
from spoiler_shield.rendering import REVIEW_STYLES, render_annotated_review, render_protected_review
from spoiler_shield.reviews import Review, ReviewDataError, load_reviews
from spoiler_shield.text import split_preserving_formatting


ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
MOVIES_PER_PAGE = 12
REVIEWS_PER_PAGE = 5
MAX_REVIEW_CHARACTERS = 12_000
MAX_SESSION_REVIEWS = 100
LOGGER = logging.getLogger("spoiler_shield.analysis")
HOME_PAGE: st.Page
MOVIE_PAGES: dict[str, st.Page]


st.set_page_config(
    page_title="Spoiler Shield movie database",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner=False)
def get_movies() -> list[Movie]:
    return load_catalog(DATA_ROOT / "plot_synopses.json", ROOT / "data" / "movie_titles.json")


@st.cache_data(show_spinner=False)
def get_reviews() -> dict[str, tuple[Review, ...]]:
    return load_reviews(DATA_ROOT / "spoiler_detection_dataset_1.3.1.json")


@st.cache_resource(show_spinner=False, max_entries=1)
def get_predictor() -> DistilbertSpoilerPredictor:
    return DistilbertSpoilerPredictor()


@st.cache_data(show_spinner=False)
def hero_background() -> str:
    image = base64.b64encode((ROOT / "assets" / "cinema-hero.png").read_bytes()).decode("ascii")
    return (
        '<style>.st-key-library_canvas::before { background-image: '
        'linear-gradient(to bottom, rgba(255,255,255,0) 68%, #ffffff 96%), '
        'linear-gradient(rgba(0,0,0,.23), rgba(0,0,0,.23)), '
        f'url("data:image/png;base64,{image}"); }}</style>'
    )


def open_movie(movie_id: str, view: str | None = None) -> None:
    st.session_state.selected_movie_id = movie_id
    if view is None:
        st.session_state.setdefault(f"detail_view_{movie_id}", "Reviews")
    else:
        st.session_state[f"detail_view_{movie_id}"] = view
    st.switch_page(MOVIE_PAGES[movie_id], query_params={})


def close_movie() -> None:
    st.switch_page(HOME_PAGE, query_params={})


def rating_label(value: int | None) -> str:
    return f"{value}/10" if value is not None else "Not rated"


def render_masthead(in_movie: bool) -> None:
    with st.container(horizontal=True, horizontal_alignment="distribute",
                      vertical_alignment="center", key="masthead"):
        with st.container(key="home_logo", width="content"):
            st.page_link(HOME_PAGE, label="Spoiler:primary[Shield]", help="Return to the movie library")
        if in_movie:
            st.button("Movie library", key="back_to_library", on_click=close_movie, type="tertiary")
        else:
            st.html('<span class="current-page" aria-current="page">Movie library</span>', width="content")


def hidden_label(count: int) -> str:
    if count == 0:
        return "No sentences hidden"
    return f"{count} sentence{'s' if count != 1 else ''} hidden"


def render_review_header(headline: str, byline: str, rating: int | None, count: int) -> None:
    status_class = "shield-status flagged" if count else "shield-status"
    st.html(
        '<header class="review-heading"><div class="review-heading-row">'
        f'<h3>{escape(headline)}</h3><span class="review-rating">{rating_label(rating)}</span></div>'
        f'<div class="review-meta"><span>{escape(byline)}</span>'
        f'<span class="{status_class}">{hidden_label(count)}</span></div></header>'
    )


def render_dataset_review(review: Review) -> None:
    with st.container(key=f"review_card_{review.review_id}", gap="small"):
        render_review_header(review.summary, f"IMDb user {review.user_id}", review.rating, review.spoiler_count)
        st.html(render_annotated_review(review.sentences))


def added_spoiler_count(review: dict[str, object]) -> int:
    predictions = review["predictions"]
    assert isinstance(predictions, list)
    return sum(item.is_spoiler for item in predictions if isinstance(item, Prediction))


def render_added_review(review: dict[str, object]) -> None:
    predictions = review["predictions"]
    assert isinstance(predictions, list)
    spoiler_count = added_spoiler_count(review)
    with st.container(key=f"review_card_added_{review['id']}", gap="small"):
        model_name = str(review.get("model_name", "earlier session analysis"))
        render_review_header(str(review["summary"]), f"Your review ({model_name})", int(str(review["rating"])), spoiler_count)
        st.html(render_protected_review(predictions))


@st.dialog("Add a Review")
def choose_review_movie(movies: list[Movie]) -> None:
    st.caption("Pick the movie you want to write about.")
    movie = st.selectbox(
        "Choose a movie", movies, index=None,
        format_func=lambda item: f"{item.title} ({item.year})" if item.year else item.title,
        placeholder="Search for a movie", key="review_movie_choice",
    )
    if st.button("Write review", key="picker_write_review", type="primary", disabled=movie is None, width="stretch") and movie is not None:
        open_movie(movie.movie_id, "Add a review")


def render_library(movies: list[Movie], reviews_by_movie: dict[str, tuple[Review, ...]]) -> None:
    added = st.session_state.added_reviews
    review_total = sum(map(len, reviews_by_movie.values())) + sum(map(len, added.values()))
    st.html(hero_background())
    with st.container(key="library_intro", gap="small"):
        st.html(
            '<div class="library-intro"><div><h1>Movie library</h1>'
            '<p>Explore movies and read reviews with spoiler sentences hidden.</p></div>'
            '<dl class="library-totals">'
            f'<div><dt>Movies</dt><dd>{len(movies):,}</dd></div>'
            f'<div><dt>Reviews</dt><dd>{review_total:,}</dd></div></dl></div>'
        )
        with st.container(horizontal=True, vertical_alignment="center", wrap=False, key="search_actions"):
            query = st.text_input(
                "Search movies", live="300ms", label_visibility="collapsed",
                placeholder="Search by movie title, year or genre", max_chars=100, key="movie_search",
                persist_state="session",
            )
            if st.button("Add a Review", key="hero_add_review", type="primary"):
                choose_review_movie(movies)
    normalized_query = query.strip().casefold()
    # Persist one current-query page value, not a growing key for every search.
    if st.session_state.get("library_page_query") != normalized_query:
        st.session_state.pop("library_page", None)
        st.session_state.library_page_query = normalized_query
    filtered = [
        movie for movie in movies
        if not normalized_query
        or normalized_query in movie.title.casefold()
        or normalized_query in str(movie.year or "")
        or any(normalized_query in genre.casefold() for genre in movie.genres)
    ]
    if normalized_query:
        st.html(f'<p class="search-feedback">{len(filtered)} matching movie{"s" if len(filtered) != 1 else ""}</p>')
    if not filtered:
        st.html('<p class="search-feedback">Try a different title, year or genre.</p>')
        return

    total_pages = math.ceil(len(filtered) / MOVIES_PER_PAGE)
    listing = st.container(key="listing")
    page = 1
    if total_pages > 1:
        with st.container(key="library_pagination", width="content"):
            page = st.pagination(total_pages, key="library_page", width="content",
                                 persist_state="session")

    start = (page - 1) * MOVIES_PER_PAGE
    with listing:
        grid = st.grid(3, min_column_width=260, gap="medium")
        for movie in filtered[start:start + MOVIES_PER_PAGE]:
            metadata = ", ".join(part for part in [movie.duration, *movie.genres] if part)
            review_count = len(reviews_by_movie.get(movie.movie_id, ())) + len(added.get(movie.movie_id, ()))
            with grid.cell(), st.container(gap=None, height="stretch", key=f"filmrow_{movie.movie_id}"):
                st.html(
                    '<div class="film-line">'
                    f'<span class="film-year">{movie.year or ""}</span>'
                    f'<div class="film-copy"><h2>{escape(movie.title)}</h2>'
                    f'<span>{escape(metadata)}</span></div>'
                    f'<span class="film-rating"><span class="mobile-label">IMDb </span>'
                    f'{escape(movie.rating) if movie.rating else "Unrated"}</span>'
                    f'<span class="film-review-count">{review_count} reviews</span></div>'
                )
                st.button(
                    f"Read reviews of {movie.title}", key=f"open_{movie.movie_id}", width="stretch",
                    on_click=open_movie, args=(movie.movie_id,),
                )


def render_reviews(
    movie_id: str,
    dataset_reviews: tuple[Review, ...],
    added_reviews: list[dict[str, object]],
) -> None:
    all_count = len(added_reviews) + len(dataset_reviews)
    total_pages = max(1, math.ceil(all_count / REVIEWS_PER_PAGE))
    reviews_slot = st.container()
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", key="review_pagination"):
        st.caption(f"{all_count} reviews. Spoiler sentences are hidden until you reveal them.")
        page = st.pagination(total_pages, key=f"review_page_{movie_id}", width="content",
                             persist_state="session")

    start = (page - 1) * REVIEWS_PER_PAGE
    end = start + REVIEWS_PER_PAGE
    combined: list[tuple[str, object]] = [
        *(("added", review) for review in added_reviews),
        *(("dataset", review) for review in dataset_reviews),
    ]
    with reviews_slot:
        for source, review in combined[start:end]:
            if source == "added":
                assert isinstance(review, dict)
                render_added_review(review)
            else:
                assert isinstance(review, Review)
                render_dataset_review(review)


def show_movie_reviews(movie_id: str) -> None:
    st.session_state[f"detail_view_{movie_id}"] = "Reviews"
    st.session_state.pop(f"review_page_{movie_id}", None)


def render_submission_result(movie_id: str) -> None:
    result = st.session_state.pop("review_submission_result", None)
    if not result or result["movie_id"] != movie_id:
        return
    review = next(
        (item for item in st.session_state.added_reviews.get(movie_id, [])
         if item["id"] == result["review_id"]),
        None,
    )
    if review is None:
        return
    if result["duplicate"]:
        st.info("This review is already saved. No extra copy was added.", title="Review already added")
    else:
        st.success("Your review is ready. You can reveal any hidden sentences below.", title="Review added")
    render_added_review(review)
    st.button("Read all reviews", key="read_after_adding", on_click=show_movie_reviews, args=(movie_id,))


def submission_fingerprint(movie_id: str, summary: str, rating: int, text: str) -> str:
    return hashlib.sha256(json.dumps(
        [movie_id, summary.strip(), rating, text], ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def render_add_review(movie: Movie) -> None:
    with st.container(key="review_composer", gap="medium"):
        st.html(
            '<div class="composer-heading"><h2>Write your review</h2>'
            f'<p>Share your thoughts on {escape(movie.title)}.</p></div>'
        )
        # Commit draft edits on blur and retain them when this view is unmounted.
        # Only the submit button runs analysis; a form would buffer these edits.
        with st.container(key="add_review_fields"):
            headline_column, rating_column = st.columns([4, 1], gap="medium")
            with headline_column:
                summary = st.text_input(
                    "Review headline",
                    placeholder="Your take on the movie",
                    max_chars=160,
                    required=True,
                    key=f"review_headline_{movie.movie_id}",
                    persist_state="session",
                )
            with rating_column:
                rating = st.number_input(
                    "Rating out of 10", min_value=1, max_value=10, value=8, step=1,
                    key=f"review_rating_{movie.movie_id}",
                    persist_state="session",
                )
            review_text = st.text_area(
                "Your review",
                height=260,
                placeholder="What worked for you? What stayed with you?",
                max_chars=MAX_REVIEW_CHARACTERS,
                required=True,
                key=f"review_text_{movie.movie_id}",
                persist_state="session",
                help=f"Spacing and line breaks are preserved. Up to 12,000 characters and {MAX_REVIEW_SENTENCES} sentences.",
            )
            st.html(
                '<div class="analysis-note"><strong>Trained spoiler detection</strong>'
                '<span>DistilBERT analyses English sentences with relevant movie synopsis context. '
                'It can miss spoilers or hide safe sentences.</span></div>'
            )
            submitted = st.button(
                "Add and analyse review", key="submit_review", type="primary", width="stretch",
            )
        st.caption(f"Your review is saved for this browser session. Up to {MAX_SESSION_REVIEWS} reviews per session.")

    if not submitted:
        render_submission_result(movie.movie_id)
        return

    clean_summary = summary.strip()
    clean_review = review_text.strip()
    if not clean_summary or not clean_review:
        st.error("Enter both a headline and review text.", title="Review not added")
        return
    if len(review_text) > MAX_REVIEW_CHARACTERS:
        st.error(f"Reviews must be {MAX_REVIEW_CHARACTERS:,} characters or fewer.", title="Review too long")
        return

    # Bound the fingerprint history together with the stored reviews. Repeating
    # a populated form, even after another submission, must not add another copy.
    fingerprint = submission_fingerprint(movie.movie_id, clean_summary, int(rating), review_text)
    existing = next(
        (item for item in st.session_state.added_reviews.get(movie.movie_id, [])
         if (item.get("fingerprint") or submission_fingerprint(
             movie.movie_id, str(item["summary"]), int(item["rating"]),
             "".join(prediction.text for prediction in item["predictions"]),
         )) == fingerprint),
        None,
    )
    if existing is not None:
        st.session_state.review_submission_result = {
            "movie_id": movie.movie_id, "review_id": existing["id"], "duplicate": True,
        }
        render_submission_result(movie.movie_id)
        return
    if sum(map(len, st.session_state.added_reviews.values())) >= MAX_SESSION_REVIEWS:
        st.error(
            f"This session has reached its limit of {MAX_SESSION_REVIEWS} added reviews. "
            "Your existing reviews and the text in this form are still available.",
            title="Session review limit reached",
        )
        return

    try:
        chunks = split_preserving_formatting(review_text)
        if sum(bool(text.strip()) for text in chunks) > MAX_REVIEW_SENTENCES:
            st.error(f"Reviews must contain {MAX_REVIEW_SENTENCES} sentences or fewer. Your draft is still in the form.",
                     title="Too many sentences")
            return
        with st.spinner("Analysing your review. The first analysis may take a few minutes while the model loads…"):
            predictor = get_predictor()
            model_name, model_id = predictor.name, predictor.model_id
            if not isinstance(model_name, str) or not model_name or not isinstance(model_id, str) or not model_id:
                raise RuntimeError("The predictor must provide its model name and export ID")
            predictions = predictor.predict(movie, chunks)
            if len(predictions) != len(chunks) or any(item.text != text for item, text in zip(predictions, chunks)):
                raise RuntimeError("Prediction results did not preserve the review text")
    except Exception:
        LOGGER.exception("Review analysis failed for movie %s", movie.movie_id)
        st.error("Your review could not be analysed. Your text is still in the form so you can try again.",
                 title="Analysis unavailable")
        return

    st.session_state.added_review_counter += 1
    added_review: dict[str, object] = {
        "id": st.session_state.added_review_counter,
        "summary": clean_summary,
        "rating": int(rating),
        "predictions": predictions,
        "fingerprint": fingerprint,
        "model_name": model_name,
        "model_id": model_id,
    }
    st.session_state.added_reviews.setdefault(movie.movie_id, []).insert(0, added_review)
    st.session_state.pop(f"review_page_{movie.movie_id}", None)
    st.session_state.review_submission_result = {
        "movie_id": movie.movie_id, "review_id": added_review["id"], "duplicate": False,
    }
    # Rebuild the header totals before displaying the successful result.
    st.rerun()


def render_movie_detail(
    movie: Movie,
    dataset_reviews: tuple[Review, ...],
    added_reviews: list[dict[str, object]],
) -> None:
    review_count = len(dataset_reviews) + len(added_reviews)
    hidden_count = sum(review.spoiler_count for review in dataset_reviews)
    hidden_count += sum(added_spoiler_count(review) for review in added_reviews)
    with st.container(key="film_header", gap="medium"):
        st.title(movie.title, anchor=False)
        st.html(
            '<dl class="movie-metadata">'
            f'<div><dt>Release year</dt><dd>{movie.year or "Not available"}</dd></div>'
            f'<div><dt>Runtime</dt><dd>{escape(movie.duration or "Not available")}</dd></div>'
            f'<div><dt>Genres</dt><dd>{escape(", ".join(movie.genres) or "Not available")}</dd></div>'
            '</dl>'
        )
        st.html(
            '<dl class="movie-facts">'
            f'<div><dt>IMDb rating</dt><dd>{escape(movie.rating) + "/10" if movie.rating else "Unrated"}</dd></div>'
            f'<div><dt>Reviews</dt><dd>{review_count}</dd></div>'
            f'<div><dt>Hidden sentences</dt><dd>{hidden_count}</dd></div>'
            '</dl>'
        )
        if movie.plot_summary:
            with st.expander("Movie overview"):
                st.write(movie.plot_summary)

    view_key = f"detail_view_{movie.movie_id}"
    st.session_state.setdefault(view_key, "Reviews")
    view = st.segmented_control(
        "Movie section",
        ["Reviews", "Add a review"],
        key=view_key,
        label_visibility="collapsed",
        persist_state="session",
    )
    if view == "Add a review":
        render_add_review(movie)
    else:
        render_reviews(movie.movie_id, dataset_reviews, added_reviews)


st.session_state.setdefault("selected_movie_id", None)
st.session_state.setdefault("added_reviews", {})
st.session_state.setdefault("added_review_counter", 0)

st.html(ROOT / "spoiler_shield" / "site.css")
st.html(REVIEW_STYLES)

try:
    movies = get_movies()
    reviews_by_movie = get_reviews()
except (CatalogError, ReviewDataError) as exc:
    st.error(str(exc), title="Movie database could not be loaded")
    st.info("Confirm that the repository's data folder is present, then restart the app.")
    st.stop()

def render_movie_page(movie: Movie) -> None:
    st.session_state.selected_movie_id = movie.movie_id
    with st.container(key="movie_page"):
        render_masthead(True)
        render_movie_detail(
            movie,
            reviews_by_movie.get(movie.movie_id, ()),
            st.session_state.added_reviews.get(movie.movie_id, []),
        )


def render_library_page() -> None:
    st.session_state.selected_movie_id = None
    with st.container(key="library_canvas"):
        render_masthead(False)
        render_library(movies, reviews_by_movie)


HOME_PAGE = st.Page(render_library_page, title="Movie library", default=True,
                    help="Browse and search the movie library.")
MOVIE_PAGES = {
    movie.movie_id: st.Page(
        lambda movie=movie: render_movie_page(movie), title=movie.title,
        url_path=f"movie-{movie.movie_id}", help=f"Read or add reviews of {movie.title}.",
    )
    for movie in movies
}
current_page = st.navigation([HOME_PAGE, *MOVIE_PAGES.values()], position="hidden")
current_page.run()
