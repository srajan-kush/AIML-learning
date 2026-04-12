import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_mistralai import ChatMistralAI

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineExtract",
    page_icon="🎬",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece2;
}

.main { background-color: #0d0d0d; }
.block-container { padding-top: 2.5rem; max-width: 780px; }

h1.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 900;
    line-height: 1.1;
    letter-spacing: -1px;
    color: #f5c842;
    margin-bottom: 0.2rem;
}

.hero-sub {
    font-size: 0.95rem;
    color: #888;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

.divider {
    width: 60px;
    height: 3px;
    background: #f5c842;
    margin: 0.6rem 0 1.8rem 0;
    border-radius: 2px;
}

/* Text area */
.stTextArea textarea {
    background: #1a1a1a !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 8px !important;
    color: #f0ece2 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    resize: vertical;
}
.stTextArea textarea:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.15) !important;
}

/* Button */
.stButton > button {
    background: #f5c842 !important;
    color: #0d0d0d !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 2rem !important;
    font-size: 0.9rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Result card */
.movie-card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-top: 1.6rem;
}

.movie-title {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5c842;
    margin-bottom: 0.2rem;
}

.movie-year {
    color: #666;
    font-size: 0.85rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.tag-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1.2rem; }

.tag {
    background: #222;
    border: 1px solid #333;
    border-radius: 20px;
    padding: 0.2rem 0.75rem;
    font-size: 0.78rem;
    color: #bbb;
    letter-spacing: 0.04em;
}
.tag.genre { border-color: #f5c842; color: #f5c842; }

.meta-row {
    display: flex;
    gap: 2.5rem;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
}

.meta-item label {
    display: block;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.25rem;
}
.meta-item span {
    font-size: 1rem;
    font-weight: 500;
    color: #f0ece2;
}

.rating-badge {
    display: inline-block;
    background: #f5c842;
    color: #0d0d0d;
    font-weight: 700;
    font-size: 1rem;
    border-radius: 6px;
    padding: 0.1rem 0.6rem;
}

.summary-text {
    font-size: 0.92rem;
    color: #aaa;
    line-height: 1.7;
    border-top: 1px solid #222;
    padding-top: 1rem;
    margin-top: 0.5rem;
}

/* Error / info */
.stAlert { border-radius: 8px !important; }

/* Spinner */
.stSpinner { color: #f5c842 !important; }

/* Hide default Streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Models ────────────────────────────────────────────────────────────────────
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str


@st.cache_resource
def get_chain():
    model = ChatMistralAI(model="mistral-small-2506")
    parser = PydanticOutputParser(pydantic_object=Movie)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract movie information from the paragraph\n{format_instructions}"),
        ("human", "{paragraph}"),
    ])
    return model, parser, prompt


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">CineExtract</h1>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Paste any movie paragraph — get structured data instantly</p>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
paragraph = st.text_area(
    label="Movie paragraph",
    placeholder='e.g. "Inception (2010), directed by Christopher Nolan, stars Leonardo DiCaprio…"',
    height=150,
    label_visibility="collapsed",
)

extract_btn = st.button("⚡ Extract Movie Info")

# ── Extraction ────────────────────────────────────────────────────────────────
if extract_btn:
    if not paragraph.strip():
        st.warning("Please enter a paragraph first.")
    else:
        with st.spinner("Extracting…"):
            try:
                llm, parser, prompt = get_chain()
                final_prompt = prompt.invoke({
                    "paragraph": paragraph,
                    "format_instructions": parser.get_format_instructions(),
                })
                response = llm.invoke(final_prompt)
                movie: Movie = parser.parse(response.content)

                # ── Render card ───────────────────────────────────────────────
                genres_html = "".join(f'<span class="tag genre">{g}</span>' for g in movie.genre)
                cast_html = "".join(f'<span class="tag">{c}</span>' for c in movie.cast)
                rating_html = (
                    f'<span class="rating-badge">★ {movie.rating}</span>'
                    if movie.rating else '<span style="color:#555">N/A</span>'
                )
                director_val = movie.director or "—"
                year_val = str(movie.release_year) if movie.release_year else "—"

                st.markdown(f"""
<div class="movie-card">
    <div class="movie-title">{movie.title}</div>
    <div class="movie-year">{year_val}</div>
    <div class="tag-row">{genres_html}</div>
    <div class="meta-row">
        <div class="meta-item">
            <label>Director</label>
            <span>{director_val}</span>
        </div>
        <div class="meta-item">
            <label>Rating</label>
            <span>{rating_html}</span>
        </div>
    </div>
    <div class="tag-row">{cast_html}</div>
    <div class="summary-text">{movie.summary}</div>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Extraction failed: {e}")