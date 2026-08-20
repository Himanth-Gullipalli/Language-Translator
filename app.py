"""
app.py
──────
Streamlit web application for Neural Machine Translation & Sentiment Analysis.

Run with:
    streamlit run app.py

UI Flow:
  1. User enters English text
  2. User selects target language
  3. Click "Analyse"
  4. Sentiment analysis result shown (label + emotion + confidence gauges)
  5. If NEGATIVE → ask "Would you like to see the translation?"
     If POSITIVE/NEUTRAL → show translation automatically
  6. Translation displayed with BLEU / ROUGE / Token-Accuracy metrics
"""

import os
import sys
import time

import streamlit as st
import plotly.graph_objects as go

# ── Project root so src.* imports work ────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── Page configuration (must be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="NMT & Sentiment Analysis",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — dark glassmorphism theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  /* ── Global ─────────────────────────────────────────── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 70%, #0f3460 100%);
    min-height: 100vh;
  }

  /* ── Hero header ──────────────────────────────────────── */
  .hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
  }
  .hero-sub {
    text-align: center;
    color: #94a3b8;
    font-size: 1.05rem;
    margin-bottom: 2rem;
    font-weight: 300;
  }

  /* ── Glass cards ──────────────────────────────────────── */
  .glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 1.5rem 1.75rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.25rem;
    transition: border-color 0.3s ease;
  }
  .glass-card:hover {
    border-color: rgba(167,139,250,0.35);
  }
  .card-title {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #a78bfa;
    margin-bottom: 0.6rem;
  }
  .card-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.1;
  }
  .card-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.25rem;
  }

  /* ── Sentiment badges ─────────────────────────────────── */
  .badge-positive {
    display: inline-block;
    background: linear-gradient(135deg, #059669, #34d399);
    color: white; border-radius: 50px;
    padding: 0.35rem 1.1rem;
    font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 15px rgba(52,211,153,0.35);
  }
  .badge-negative {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #f87171);
    color: white; border-radius: 50px;
    padding: 0.35rem 1.1rem;
    font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 15px rgba(248,113,113,0.35);
  }
  .badge-neutral {
    display: inline-block;
    background: linear-gradient(135deg, #d97706, #fbbf24);
    color: white; border-radius: 50px;
    padding: 0.35rem 1.1rem;
    font-weight: 600; font-size: 1rem;
    box-shadow: 0 4px 15px rgba(251,191,36,0.35);
  }

  /* ── Metric cards row ────────────────────────────────── */
  .metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
  }
  .metric-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 0.75rem 1.1rem;
    flex: 1;
    min-width: 110px;
    text-align: center;
  }
  .metric-chip-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #94a3b8;
    margin-bottom: 0.3rem;
  }
  .metric-chip-value {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  /* ── Translation box ─────────────────────────────────── */
  .translation-box {
    background: rgba(96,165,250,0.07);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    color: #e2e8f0;
    line-height: 1.6;
    word-break: break-word;
  }

  /* ── Emotion pill ────────────────────────────────────── */
  .emotion-pill {
    display: inline-block;
    background: rgba(167,139,250,0.15);
    border: 1px solid rgba(167,139,250,0.4);
    border-radius: 50px;
    padding: 0.3rem 0.9rem;
    font-size: 0.88rem;
    color: #c4b5fd;
    margin-top: 0.5rem;
  }

  /* ── Warning box ─────────────────────────────────────── */
  .warning-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    color: #fca5a5;
    font-size: 0.92rem;
    line-height: 1.5;
  }

  /* ── Section divider ─────────────────────────────────── */
  .section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.07);
    margin: 1.5rem 0;
  }

  /* ── Streamlit component overrides ───────────────────── */
  .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
  }
  .stTextArea textarea:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.12) !important;
  }
  .stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    color: #f1f5f9 !important;
  }
  .stButton > button {
    background: linear-gradient(135deg, #6d28d9, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(109,40,217,0.45) !important;
  }
  label, .stSelectbox label, .stTextArea label {
    color: #94a3b8 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
  }
  .stSpinner > div {
    border-top-color: #a78bfa !important;
  }
  footer { display: none !important; }
  #MainMenu { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Import project modules (lazy-loaded in functions to avoid startup errors)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_sentiment():
    """Load sentiment model once and cache it."""
    from src.sentiment.predict import load_sentiment_model
    return load_sentiment_model()


@st.cache_resource(show_spinner=False)
def _load_translation(language: str):
    """Load translation model for a given language once and cache it."""
    from src.translation.predict import load_model_for_language
    return load_model_for_language(language)


def run_sentiment(text: str) -> dict:
    from src.sentiment.predict import full_analysis
    return full_analysis(text)


def run_translation(text: str, language: str) -> dict:
    from src.translation.predict import translate
    return translate(text, language)


def run_metrics(reference: str, hypothesis: str) -> dict:
    """
    Compute metrics. Falls back gracefully if no reference is available.
    When running live (no reference), we compute reference-free proxies.
    """
    from src.translation.metrics import compute_all_metrics
    if reference and reference.strip():
        return compute_all_metrics(reference, hypothesis)
    # No reference → return placeholder
    return {
        "bleu":           "N/A",
        "rouge1":         "N/A",
        "rouge2":         "N/A",
        "rougeL":         "N/A",
        "token_accuracy": "N/A",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotly Gauge Helper
# ─────────────────────────────────────────────────────────────────────────────

def confidence_gauge(value: float, label: str, color: str) -> go.Figure:
    """Create a Plotly gauge chart for model confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value * 100, 1),
        number={"suffix": "%", "font": {"size": 26, "color": "#f1f5f9"}},
        gauge={
            "axis":      {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
            "bar":       {"color": color, "thickness": 0.25},
            "bgcolor":   "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0,  40],  "color": "rgba(239,68,68,0.1)"},
                {"range": [40, 70],  "color": "rgba(251,191,36,0.1)"},
                {"range": [70, 100], "color": "rgba(52,211,153,0.1)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": round(value * 100, 1),
            },
        },
        title={"text": label, "font": {"size": 13, "color": "#94a3b8"}},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f1f5f9"},
    )
    return fig


def probability_bar_chart(probabilities: dict) -> go.Figure:
    """Horizontal bar chart showing all class probabilities."""
    labels = list(probabilities.keys())
    values = list(probabilities.values())
    colors_map = {
        "Negative": "#f87171",
        "Neutral":  "#fbbf24",
        "Positive": "#34d399",
    }
    bar_colors = [colors_map.get(l, "#a78bfa") for l in labels]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color="rgba(255,255,255,0.1)", width=1),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#f1f5f9", size=12),
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, max(values) * 1.25],
            showgrid=False, zeroline=False,
            tickfont=dict(color="#64748b"),
        ),
        yaxis=dict(
            tickfont=dict(color="#94a3b8", size=13),
            showgrid=False,
        ),
        showlegend=False,
    )
    return fig


def metrics_radar_chart(metrics: dict) -> go.Figure:
    """Radar chart for translation metrics (only when values are numeric)."""
    if "N/A" in metrics.values():
        return None

    categories = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Token Acc"]
    values     = [
        metrics.get("bleu",           0),
        metrics.get("rouge1",         0),
        metrics.get("rouge2",         0),
        metrics.get("rougeL",         0),
        metrics.get("token_accuracy", 0),
    ]
    values_closed = values + [values[0]]
    cats_closed   = categories + [categories[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=cats_closed,
        fill="toself",
        fillcolor="rgba(96,165,250,0.15)",
        line=dict(color="#60a5fa", width=2),
        marker=dict(color="#60a5fa", size=7),
        name="Scores",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="#64748b", size=9),
                gridcolor="rgba(255,255,255,0.08)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#94a3b8", size=11),
                gridcolor="rgba(255,255,255,0.08)",
            ),
        ),
        showlegend=False,
        height=280,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Result Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_sentiment_result(result: dict):
    """Render sentiment analysis results inside glass cards."""
    label      = result["label"]
    confidence = result["confidence"]
    probs      = result["probabilities"]
    emotion    = result["dominant_emotion"]
    emotion_desc = result["description"]

    badge_class = {
        "Positive": "badge-positive",
        "Negative": "badge-negative",
        "Neutral":  "badge-neutral",
    }.get(label, "badge-neutral")

    sentiment_color = {
        "Positive": "#34d399",
        "Negative": "#f87171",
        "Neutral":  "#fbbf24",
    }.get(label, "#a78bfa")

    # ── Header row: badge + emotion pill ──────────────────────────────────
    st.markdown(f"""
    <div class="glass-card">
      <div class="card-title">🎯 Sentiment Classification</div>
      <span class="{badge_class}">{label}</span>
      <div class="emotion-pill">{emotion_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two columns: gauge + bar chart ────────────────────────────────────
    col_g, col_b = st.columns([1, 1.4])

    with col_g:
        st.markdown('<div class="card-title">📊 Confidence</div>', unsafe_allow_html=True)
        st.plotly_chart(
            confidence_gauge(confidence, label, sentiment_color),
            use_container_width=True, config={"displayModeBar": False}
        )

    with col_b:
        st.markdown('<div class="card-title">📈 Class Probabilities</div>', unsafe_allow_html=True)
        st.plotly_chart(
            probability_bar_chart(probs),
            use_container_width=True, config={"displayModeBar": False}
        )


# ─────────────────────────────────────────────────────────────────────────────
# Translation Result Renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_translation_result(
    translation: str,
    language: str,
    metrics: dict,
):
    """Render the translated text and metric score cards."""
    lang_flags = {
        "french":  "🇫🇷",
        "spanish": "🇪🇸",
        "german":  "🇩🇪",
        "hindi":   "🇮🇳",
    }
    flag = lang_flags.get(language, "🌐")

    st.markdown(f"""
    <div class="glass-card">
      <div class="card-title">{flag} Translation → {language.capitalize()}</div>
      <div class="translation-box">{translation if translation else "<em style='color:#64748b'>No translation available — the input may be too short or unrecognised.</em>"}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric chips ─────────────────────────────────────────────────────
    metric_labels = {
        "bleu":           ("BLEU", "%"),
        "rouge1":         ("ROUGE-1", "%"),
        "rouge2":         ("ROUGE-2", "%"),
        "rougeL":         ("ROUGE-L", "%"),
        "token_accuracy": ("Token Acc", "%"),
    }

    chips_html = '<div class="metric-row">'
    for key, (name, unit) in metric_labels.items():
        val = metrics.get(key, "N/A")
        display = f"{val}{unit}" if val != "N/A" else "N/A"
        chips_html += f"""
        <div class="metric-chip">
          <div class="metric-chip-label">{name}</div>
          <div class="metric-chip-value">{display}</div>
        </div>"""
    chips_html += "</div>"

    st.markdown(f"""
    <div class="glass-card">
      <div class="card-title">📐 Translation Metrics</div>
      {chips_html}
      <div class="card-sub" style="margin-top:0.75rem;">
        * Metrics are computed against a reference translation when available.
          In live mode (no reference), scores show N/A.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Radar chart ────────────────────────────────────────────────────────
    radar = metrics_radar_chart(metrics)
    if radar:
        with st.expander("📡 Metrics Radar Chart", expanded=False):
            st.plotly_chart(radar, use_container_width=True,
                            config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Hero ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding: 2rem 0 1rem 0;">
      <div class="hero-title">🌐 Neural Machine Translation</div>
      <div class="hero-title" style="font-size:1.8rem;">& Sentiment Analysis</div>
      <div class="hero-sub">
        Deep learning–powered multi-task NLP · Seq2Seq with Bahdanau Attention · BiLSTM Sentiment Classifier
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input section ────────────────────────────────────────────────────────
    col_left, col_right = st.columns([1.6, 1], gap="large")

    with col_left:
        st.markdown('<div class="card-title">✍️ Input Text (English)</div>',
                    unsafe_allow_html=True)
        user_text = st.text_area(
            label="Input text",
            label_visibility="collapsed",
            placeholder=(
                "Type or paste your English text here…\n\n"
                "Examples:\n"
                "  • I absolutely love this product!\n"
                "  • The service was really disappointing.\n"
                "  • The meeting is scheduled for Monday at 10 AM."
            ),
            height=180,
            key="input_text",
        )

    with col_right:
        st.markdown('<div class="card-title">🗺️ Target Language</div>',
                    unsafe_allow_html=True)
        language_options = {
            "🇫🇷  French":  "french",
            "🇪🇸  Spanish": "spanish",
            "🇩🇪  German":  "german",
            "🇮🇳  Hindi":   "hindi",
        }
        selected_display = st.selectbox(
            label="Target language",
            label_visibility="collapsed",
            options=list(language_options.keys()),
            key="language_select",
        )
        language = language_options[selected_display]

        st.markdown('<div style="height: 0.75rem;"></div>', unsafe_allow_html=True)

        # Optional reference translation for metrics
        with st.expander("📎 Provide Reference Translation (optional)", expanded=False):
            reference_text = st.text_area(
                "Reference translation",
                placeholder="Paste a known correct translation to compute BLEU / ROUGE scores…",
                height=80,
                key="reference_text",
            )
        reference_text = st.session_state.get("reference_text", "")

    # ── Analyse button ───────────────────────────────────────────────────────
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        analyse_clicked = st.button(
            "🔍  Analyse & Translate",
            key="analyse_btn",
            use_container_width=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Session state defaults ────────────────────────────────────────────────
    if "sentiment_result" not in st.session_state:
        st.session_state.sentiment_result   = None
    if "show_translation" not in st.session_state:
        st.session_state.show_translation   = False
    if "translation_result" not in st.session_state:
        st.session_state.translation_result = None
    if "last_language" not in st.session_state:
        st.session_state.last_language      = language
    if "last_text" not in st.session_state:
        st.session_state.last_text          = ""

    # ── On "Analyse" click ───────────────────────────────────────────────────
    if analyse_clicked:
        if not user_text.strip():
            st.warning("⚠️  Please enter some text before clicking Analyse.")
            return

        # Reset translation state on new input
        st.session_state.show_translation   = False
        st.session_state.translation_result = None
        st.session_state.last_text          = user_text
        st.session_state.last_language      = language

        # ── Sentiment analysis ─────────────────────────────────────────────
        with st.spinner("🧠  Analysing sentiment…"):
            try:
                _load_sentiment()   # pre-warm cache
                sentiment = run_sentiment(user_text)
                st.session_state.sentiment_result = sentiment
            except FileNotFoundError as exc:
                st.error(
                    f"**Sentiment model not found.**\n\n"
                    f"{exc}\n\n"
                    "Train the model using `notebooks/sentiment/2_sentiment_training.ipynb`."
                )
                return

        # ── Auto-translate if Positive / Neutral ───────────────────────────
        label = sentiment["label"]
        if label in ("Positive", "Neutral"):
            st.session_state.show_translation = True
            with st.spinner(f"🌐  Translating to {language.capitalize()}…"):
                try:
                    _load_translation(language)
                    trans_result = run_translation(user_text, language)
                    metrics = run_metrics(
                        reference_text,
                        trans_result["translation"],
                    )
                    st.session_state.translation_result = {
                        "output":   trans_result["translation"],
                        "metrics":  metrics,
                        "language": language,
                    }
                except FileNotFoundError as exc:
                    st.session_state.translation_result = {"error": str(exc)}

    # ── Render Sentiment Results ──────────────────────────────────────────────
    if st.session_state.sentiment_result:
        result = st.session_state.sentiment_result
        render_sentiment_result(result)

        # ── Negative flow: ask user ────────────────────────────────────────
        if result["label"] == "Negative" and not st.session_state.show_translation:
            st.markdown("""
            <div class="warning-box">
              ⚠️ <strong>Negative Sentiment Detected.</strong><br>
              The input text appears to express negative sentiment.
              Would you still like to see the translation?
            </div>
            """, unsafe_allow_html=True)

            col_yes, col_no, _ = st.columns([1, 1, 2])
            with col_yes:
                if st.button("✅  Yes, translate it", key="yes_translate"):
                    st.session_state.show_translation = True
                    lang = st.session_state.last_language
                    txt  = st.session_state.last_text
                    with st.spinner(f"🌐  Translating to {lang.capitalize()}…"):
                        try:
                            _load_translation(lang)
                            trans_result = run_translation(txt, lang)
                            metrics = run_metrics(
                                reference_text,
                                trans_result["translation"],
                            )
                            st.session_state.translation_result = {
                                "output":   trans_result["translation"],
                                "metrics":  metrics,
                                "language": lang,
                            }
                        except FileNotFoundError as exc:
                            st.session_state.translation_result = {"error": str(exc)}
                    st.rerun()

            with col_no:
                if st.button("❌  No, skip translation", key="no_translate"):
                    st.session_state.show_translation   = False
                    st.session_state.translation_result = None
                    st.info("Translation skipped.")

    # ── Render Translation Results ────────────────────────────────────────────
    if st.session_state.show_translation and st.session_state.translation_result:
        trans = st.session_state.translation_result

        if "error" in trans:
            st.error(
                f"**Translation model not found.**\n\n{trans['error']}\n\n"
                f"Train the model using `notebooks/translation/2_translation_training.ipynb`."
            )
        else:
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            render_translation_result(
                translation=trans["output"],
                language=trans["language"],
                metrics=trans["metrics"],
            )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; color: #334155; font-size: 0.78rem; margin-top: 3rem; padding-bottom: 1rem;">
      Neural Machine Translation & Sentiment Analysis &nbsp;·&nbsp;
      Seq2Seq + Bahdanau Attention &nbsp;·&nbsp; Bidirectional LSTM
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
