"""
Mental Health Sentiment Detector — BERT-only version.
Uses the local fine-tuned BERT model exclusively (no company LLM calls).
"""

import io
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime
import mental_health_classifier as bert_clf

# ── Constants ─────────────────────────────────────────────────────────────────

# 5 user-facing categories (bipolar→depression, personality_disorder→anxiety internally)
CATEGORIES = [
    "anxiety",
    "depression",
    "normal",
    "stress",
    "suicidal",
]

CATEGORY_META = {
    "anxiety": {
        "color": "#F59E0B",
        "icon": "😰",
        "label": "Anxiety",
        "description": "Signs of worry, nervousness, or unease about an uncertain outcome.",
        "bg_color": "rgba(245,158,11,0.08)",
        "suggested_action": "Try box breathing: inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 5 times.",
        "resources": [
            ("🌬️ Box breathing", "Inhale 4s → hold 4s → exhale 4s → hold 4s. Repeat 5 times."),
            ("🌐 ADAA", "Visit adaa.org — Anxiety & Depression Association of America"),
            ("🩺 Professional help", "Consider speaking with a licensed therapist or counsellor"),
        ],
    },
    "depression": {
        "color": "#6366F1",
        "icon": "😔",
        "label": "Depression",
        "description": "Persistent feelings of sadness, hopelessness, or loss of interest.",
        "bg_color": "rgba(99,102,241,0.08)",
        "suggested_action": "Reach out to one trusted person today and establish a small daily routine.",
        "resources": [
            ("📞 NAMI Helpline", "1-800-950-NAMI (6264) — National Alliance on Mental Illness"),
            ("🤝 Reach out", "Talk to a trusted friend, family member, or counsellor today"),
            ("🚶 Gentle routine", "Establish a daily routine with light physical activity"),
        ],
    },
    "normal": {
        "color": "#10B981",
        "icon": "😊",
        "label": "Normal",
        "description": "No significant indicators of mental health distress detected.",
        "bg_color": "rgba(16,185,129,0.08)",
        "suggested_action": "Keep up your healthy habits — sleep, exercise, and social connection.",
        "resources": [
            ("💪 Healthy habits", "Maintain sleep, exercise, and social connection routines"),
            ("🧘 Mindfulness", "Continue practising mindfulness and self-care daily"),
        ],
    },
    "stress": {
        "color": "#EF4444",
        "icon": "😤",
        "label": "Stress",
        "description": "Feelings of emotional or physical tension from demanding situations.",
        "bg_color": "rgba(239,68,68,0.08)",
        "suggested_action": "Take a 5-minute break now, step outside if possible, and prioritise your top 3 tasks.",
        "resources": [
            ("⏸️ Take breaks", "Short 5-min breaks every hour reduce cortisol significantly"),
            ("🌐 AIS", "Visit stress.org — American Institute of Stress"),
            ("😴 Sleep hygiene", "Prioritise 7-9h sleep and limit caffeine after noon"),
        ],
    },
    "suicidal": {
        "color": "#FF2D55",
        "icon": "🆘",
        "label": "Suicidal Ideation",
        "description": "Expressions of thoughts about ending one's life — requires immediate attention.",
        "bg_color": "rgba(255,45,85,0.12)",
        "suggested_action": "Please call or text 988 (US Suicide & Crisis Lifeline) right now.",
        "resources": [
            ("☎️ 988 Lifeline", "Call or text 988 — National Suicide Prevention Lifeline, 24/7"),
            ("💬 Crisis Text Line", "Text HOME to 741741 — Free, confidential support"),
            ("🌍 IASP", "iasp.info/resources/Crisis_Centres — International crisis centres"),
        ],
    },
}

# Simple keyword sets used to highlight likely indicator phrases
KEYWORDS = {
    "anxiety":              ["anxious", "worried", "worry", "nervou", "panic", "fear", "heart races",
                             "racing heart", "can't sleep", "overthink", "what if", "uneasy", "dread"],
    "depression":           ["depress", "hopeless", "empty", "no joy", "sad", "worthless", "numb",
                             "exhausted", "can't get up", "staying in bed", "nothing matters"],
    "normal":               [],
    "stress":               ["stress", "overwhelm", "deadline", "pressure", "too much", "burned out",
                             "can't cope", "exhausted", "overload"],
    "suicidal":             ["end my life", "kill myself", "don't want to live", "better off without me",
                             "no reason to go on", "suicide", "die", "end it all", "give up"],
}


def _extract_indicators(text: str, category: str) -> list[str]:
    """Return keyword phrases found in the text for the detected category."""
    text_lower = text.lower()
    found = []
    for kw in KEYWORDS.get(category, []):
        if kw in text_lower:
            # Find the surrounding snippet (up to 6 words) in original text
            idx = text_lower.find(kw)
            snippet = text[max(0, idx - 10): idx + len(kw) + 10].strip()
            found.append(snippet)
        if len(found) >= 3:
            break
    return found


def classify_text(text: str) -> dict:
    """Run BERT inference and return a full result dict."""
    result = bert_clf.classify(text)
    category = result["category"]
    meta = CATEGORY_META[category]

    return {
        "category": category,
        "confidence": result["confidence"],
        "all_scores": result["all_scores"],
        "key_indicators": _extract_indicators(text, category),
        "brief_explanation": meta["description"],
        "suggested_action": meta["suggested_action"],
        "model_used": "fine-tuned BERT",
    }


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    defaults = {"history": [], "last_result": None, "last_text": ""}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def add_history(text: str, result: dict):
    st.session_state["history"].append({
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_preview":      text[:120] + ("…" if len(text) > 120 else ""),
        "full_text":         text,
        "category":          result["category"],
        "confidence":        result["confidence"],
        "all_scores":        result.get("all_scores", {}),
        "key_indicators":    result.get("key_indicators", []),
        "brief_explanation": result.get("brief_explanation", ""),
        "suggested_action":  result.get("suggested_action", ""),
    })
    st.session_state["last_result"] = result
    st.session_state["last_text"] = text


# ── Charts ────────────────────────────────────────────────────────────────────

def _hex_rgba(hex_color: str, alpha: float = 0.25) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def gauge_chart(confidence: int, category: str) -> go.Figure:
    color = CATEGORY_META[category]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        delta={"reference": 50, "valueformat": ".0f", "suffix": "%"},
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 12}, "tickcolor": "#64748b"},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": "#f1f5f9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "#e2e8f0"},
                {"range": [33, 66], "color": "#cbd5e1"},
                {"range": [66, 100], "color": "#bfdbfe"},
            ],
            "threshold": {"line": {"color": color, "width": 5}, "thickness": 0.8, "value": confidence},
        },
        title={"text": "Model Confidence", "font": {"size": 16, "color": "#475569"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b")
    return fig


def radar_chart(all_scores: dict, category: str) -> go.Figure:
    color = CATEGORY_META[category]["color"]
    labels = [CATEGORY_META[c]["label"] for c in CATEGORIES]
    values = [all_scores.get(c, 0) for c in CATEGORIES]
    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]
    fill_color = _hex_rgba(color, 0.25)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_c, theta=labels_c, fill="toself",
        fillcolor=fill_color, line=dict(color=color, width=2),
        name="Scores", hovertemplate="%{theta}: %{r}%<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="#f8fafc",
            radialaxis=dict(visible=True, range=[0, 100],
                            tickfont=dict(size=9, color="#94a3b8"),
                            gridcolor="#e2e8f0", linecolor="#e2e8f0"),
            angularaxis=dict(tickfont=dict(size=12, color="#475569"),
                             gridcolor="#e2e8f0", linecolor="#e2e8f0"),
        ),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
        showlegend=False, margin=dict(l=40, r=40, t=50, b=30), height=280,
        title=dict(text="Score Breakdown", font=dict(size=15, color="#475569")),
    )
    return fig


def history_bar_chart(history: list) -> go.Figure:
    counts = {c: sum(1 for h in history if h["category"] == c) for c in CATEGORIES}
    labels = [f"{CATEGORY_META[c]['icon']} {CATEGORY_META[c]['label']}" for c in CATEGORIES]
    colors = [CATEGORY_META[c]["color"] for c in CATEGORIES]
    fig = go.Figure(go.Bar(
        x=labels, y=list(counts.values()), marker_color=colors,
        text=list(counts.values()), textposition="auto",
        hovertemplate="%{x}: %{y} analyses<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Sentiment Distribution", font=dict(size=15, color="#475569")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b", margin=dict(l=10, r=10, t=50, b=10),
        height=300, yaxis=dict(gridcolor="#e2e8f0", zeroline=False),
        xaxis=dict(gridcolor="#e2e8f0"), bargap=0.3,
    )
    return fig


def history_pie_chart(history: list) -> go.Figure:
    counts = {c: sum(1 for h in history if h["category"] == c) for c in CATEGORIES}
    active = [c for c in CATEGORIES if counts[c] > 0]
    labels = [f"{CATEGORY_META[c]['icon']} {CATEGORY_META[c]['label']}" for c in active]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=[counts[c] for c in active],
        marker_colors=[CATEGORY_META[c]["color"] for c in active],
        hole=0.5, textinfo="label+percent",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        pull=[0.05 if c == max(active, key=lambda x: counts[x]) else 0 for c in active],
    ))
    fig.update_layout(
        title=dict(text="Category Breakdown", font=dict(size=15, color="#475569")),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
        margin=dict(l=10, r=10, t=50, b=10), height=300, showlegend=False,
    )
    return fig


def timeline_chart(history: list) -> go.Figure:
    if not history:
        return go.Figure()
    fig = go.Figure()
    for cat in CATEGORIES:
        subset = [h for h in history if h["category"] == cat]
        if not subset:
            continue
        meta = CATEGORY_META[cat]
        fig.add_trace(go.Scatter(
            x=[h["timestamp"] for h in subset],
            y=[h["confidence"] for h in subset],
            mode="markers+lines",
            marker=dict(color=meta["color"], size=10, symbol="circle",
                        line=dict(color="#ffffff", width=1.5)),
            line=dict(color=meta["color"], width=1, dash="dot"),
            name=f"{meta['icon']} {meta['label']}",
            hovertemplate=f"<b>{meta['label']}</b><br>Time: %{{x}}<br>Confidence: %{{y}}%<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Analysis Timeline", font=dict(size=15, color="#475569")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b",
        yaxis=dict(title="Confidence %", range=[0, 105], gridcolor="#e2e8f0", zeroline=False),
        xaxis=dict(gridcolor="#e2e8f0"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(size=11)),
        margin=dict(l=10, r=10, t=50, b=10), height=320,
    )
    return fig


def export_csv(records: list) -> bytes:
    import io
    buf = io.StringIO()
    pd.DataFrame(records).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MH Detector — BERT",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: #f8fafc; min-height: 100vh; }

    .hero-banner {
        background: linear-gradient(135deg, #e0f2fe 0%, #ede9fe 100%);
        border-radius: 20px; padding: 32px 36px; margin-bottom: 20px;
        border: 1px solid rgba(99,102,241,0.2);
        box-shadow: 0 4px 20px rgba(99,102,241,0.1);
    }
    .hero-title {
        font-size: 2.2rem; font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #db2777);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;
    }
    .hero-subtitle { color: #475569; font-size: 1rem; margin-top: 8px; }

    .model-badge {
        display: inline-block;
        background: linear-gradient(90deg,#2563eb22,#7c3aed22);
        border: 1px solid #7c3aed55;
        border-radius: 20px; padding: 4px 14px;
        font-size: 0.78rem; font-weight: 600; color: #5b21b6; margin-top: 8px;
    }

    .crisis-alert {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #FF2D55; border-radius: 16px;
        padding: 20px 24px; margin-bottom: 16px;
        animation: pulse-border 1.5s infinite;
        box-shadow: 0 0 30px rgba(255,45,85,0.4);
    }
    @keyframes pulse-border {
        0%, 100% { box-shadow: 0 0 20px rgba(255,45,85,0.4); }
        50%       { box-shadow: 0 0 40px rgba(255,45,85,0.8); }
    }

    .result-card {
        border-radius: 16px; padding: 22px 26px; margin-top: 12px;
        border-left: 5px solid; transition: all 0.3s ease;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); background: #ffffff;
    }
    .result-card:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(0,0,0,0.12); }

    .indicator-chip {
        display: inline-block; background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.1); border-radius: 20px;
        padding: 5px 14px; margin: 3px 4px 3px 0;
        font-size: 0.82rem; color: #1e293b;
    }

    .resource-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    }
    .resource-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.08); }
    .resource-title { font-weight: 600; font-size: 0.9rem; }
    .resource-desc  { color: #64748b; font-size: 0.82rem; margin-top: 2px; }

    .history-item {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 10px 12px;
        margin-bottom: 7px; font-size: 0.78rem; color: #1e293b;
    }

    .pinned-card {
        background: #f1f5f9; border: 1px solid #cbd5e1;
        border-radius: 12px; padding: 14px 16px; margin-bottom: 12px;
    }

    .char-counter {
        text-align: right; color: #94a3b8;
        font-size: 0.78rem; margin-top: -8px; margin-bottom: 4px;
    }

    .action-badge {
        display: inline-block; background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25); border-radius: 8px;
        padding: 10px 16px; font-size: 0.9rem; margin-top: 8px;
        width: 100%; color: #3730a3;
    }

    .empty-state {
        height: 360px; display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 12px; opacity: 0.5;
    }
    .empty-state-icon { font-size: 5rem; animation: float 3s ease-in-out infinite; }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%       { transform: translateY(-12px); }
    }
    .empty-state-text { font-size: 1rem; color: #475569; }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 20px; font-weight: 500; }

    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Init ──────────────────────────────────────────────────────────────────────

init_state()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        "<div style='font-size:1.4rem;font-weight:700;color:#2563eb;margin-bottom:2px'>🧠 MH Detector</div>"
        "<div class='model-badge'>🤖 Fine-tuned BERT</div>"
        "<div style='font-size:0.75rem;color:#64748b;margin-top:6px;margin-bottom:16px'>"
        "100% local inference — no external API calls</div>",
        unsafe_allow_html=True,
    )

    if st.session_state["last_result"]:
        lr   = st.session_state["last_result"]
        lm   = CATEGORY_META[lr["category"]]
        st.markdown(
            f"""<div class="pinned-card">
                <div style="font-size:0.7rem;color:#64748b;margin-bottom:6px">LATEST RESULT</div>
                <div style="font-size:1.1rem;font-weight:700;color:{lm['color']}">
                    {lm['icon']} {lm['label']}
                </div>
                <div style="font-size:0.82rem;color:#475569;margin-top:4px">
                    Confidence: <b style="color:#1e293b">{lr['confidence']}%</b>
                </div>
                <div style="font-size:0.78rem;color:#64748b;margin-top:6px;font-style:italic">
                    {st.session_state['last_text'][:90]}{'…' if len(st.session_state['last_text']) > 90 else ''}
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("📖 Category Legend", expanded=False):
        for cat, meta in CATEGORY_META.items():
            st.markdown(
                f"<span style='color:{meta['color']}'>{meta['icon']} **{meta['label']}**</span> — "
                f"<span style='color:#94a3b8;font-size:0.8rem'>{meta['description']}</span>",
                unsafe_allow_html=True,
            )

    st.divider()

    h_count = len(st.session_state["history"])
    st.markdown(
        f"<div style='font-weight:600;margin-bottom:8px;color:#1e293b'>Analysis History "
        f"<span style='background:#e0f2fe;border-radius:10px;padding:2px 8px;font-size:0.75rem;"
        f"color:#0369a1'>{h_count}</span></div>",
        unsafe_allow_html=True,
    )

    if not st.session_state["history"]:
        st.caption("No analyses yet.")
    else:
        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.update({"history": [], "last_result": None, "last_text": ""})
                st.rerun()
        with col_export:
            csv_data = export_csv([
                {"timestamp": h["timestamp"], "category": h["category"],
                 "confidence": h["confidence"], "preview": h["text_preview"],
                 "explanation": h["brief_explanation"]}
                for h in st.session_state["history"]
            ])
            st.download_button(
                "📥 Export", data=csv_data,
                file_name=f"mh_bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", use_container_width=True,
            )

        for h in reversed(st.session_state["history"]):
            meta = CATEGORY_META[h["category"]]
            st.markdown(
                f"""<div class="history-item">
                    <span style="color:{meta['color']}">{meta['icon']} <b>{meta['label']}</b></span>
                    &nbsp;·&nbsp;<span style="color:#475569">{h['confidence']}%</span><br/>
                    <span style="color:#94a3b8;font-size:0.7rem">{h['timestamp']}</span><br/>
                    <span style="color:#475569">{h['text_preview']}</span>
                </div>""",
                unsafe_allow_html=True,
            )

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    """<div class="hero-banner">
        <div class="hero-title">🧠 Mental Health Sentiment Detector</div>
        <div class="hero-subtitle">
            Detects <b>anxiety</b>, <b>depression</b>, <b>stress</b>,
            <b>suicidal ideation</b> or <b>normal</b> state using a
            <b>locally running fine-tuned BERT model</b> —
            no internet or API call required.
        </div>
    </div>""",
    unsafe_allow_html=True,
)

st.warning(
    "**Disclaimer:** This tool is for educational and research purposes only. "
    "It is **not** a clinical diagnostic tool. "
    "If you or someone you know is in crisis, call **988** (US) or your local emergency number immediately.",
    icon="⚠️",
)

# ── Tabs ──────────────────────────────────────────────────────────────────────

h = st.session_state["history"]
tab_single, tab_batch, tab_stats = st.tabs([
    "📝 Single Analysis",
    "📋 Batch Analysis",
    f"📊 Statistics ({len(h)})" if h else "📊 Statistics",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Single Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_single:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### ✍️ Input Text")

        sample_texts = {
            "— Select a sample —": "",
            "😰 Anxious post":  "I can't stop worrying about everything. My heart races constantly and I can't sleep. What if something terrible happens?",
            "😔 Depressed post": "Nothing brings me joy anymore. I've been staying in bed all day. I feel completely empty and hopeless about the future.",
            "😤 Stressed post":  "I have three deadlines tomorrow and my boss keeps piling more work on me. I'm overwhelmed and can't handle this anymore.",
            "🆘 Crisis signal":  "I've been thinking that everyone would be better off without me. I don't see a reason to keep going.",
            "😊 Normal post":    "Had a great walk in the park today! The weather was perfect and I finally finished that book I've been reading. Feeling refreshed.",
        }

        if "_text_buf" not in st.session_state:
            st.session_state["_text_buf"] = ""

        def _on_sample_change():
            chosen = st.session_state["_sample_sel"]
            if chosen != "— Select a sample —":
                st.session_state["_text_buf"] = sample_texts[chosen]

        st.selectbox(
            "Load a sample text:",
            list(sample_texts.keys()),
            key="_sample_sel",
            on_change=_on_sample_change,
        )

        # 1. Wrap the input and button in a form to batch keystrokes
        with st.form(key="single_analysis_form", clear_on_submit=False):
            user_text = st.text_area(
                "Paste or type social media text here:",
                value=st.session_state["_text_buf"],
                height=200,
                placeholder="e.g. 'I feel so overwhelmed lately, can't sleep and just don't care about anything anymore...'",
            )
            
            # The form submit button intercepts the Enter key properly
            analyze_btn = st.form_submit_button(
                "🔍 Analyse Sentiment ✨", 
                type="primary", 
                use_container_width=True
            )
            
        # 2. Place the clear button outside the form so it works independently
        if st.button("🗑️ Clear Text", use_container_width=True):
            st.session_state["_text_buf"] = ""
            st.rerun()
            
        # 3. Sync the text buffer ONLY when they click analyze
        if analyze_btn:
            st.session_state["_text_buf"] = user_text
            
    with col_result:
        st.markdown("#### 📊 Detection Result")

        if analyze_btn and user_text.strip():
            with st.spinner("Running BERT inference…"):
                result = classify_text(user_text)

            add_history(user_text, result)
            category   = result["category"]
            meta       = CATEGORY_META[category]
            confidence = result["confidence"]
            all_scores = result["all_scores"]

            # ── Alerts ───────────────────────────────────────────────────────
            if category == "suicidal":
                st.markdown(
                    """<div class="crisis-alert">
                        <div style="font-size:1.3rem;font-weight:700;color:#ff6b8a">
                            🆘 CRISIS SIGNAL DETECTED
                        </div>
                        <div style="color:#fca5a5;margin-top:6px">
                            This text may indicate a mental health crisis.
                            Please reach out immediately:
                        </div>
                        <div style="margin-top:10px;font-size:1rem;font-weight:700;color:white">
                            ☎️ Call or text <span style="color:#ff6b8a">988</span>
                            &nbsp;·&nbsp; Text <span style="color:#ff6b8a">HOME</span> to 741741
                        </div>
                    </div>""",
                    unsafe_allow_html=True,
                )
            elif category == "normal":
                st.balloons()

            # ── Model badge ───────────────────────────────────────────────────
            st.markdown(
                '<div style="font-size:0.75rem;color:#94a3b8;margin-bottom:4px">'
                '🤖 Classified by: <b>fine-tuned BERT</b> (local inference)</div>',
                unsafe_allow_html=True,
            )

            # ── Result card ───────────────────────────────────────────────────
            st.markdown(
                f"""<div class="result-card" style="border-color:{meta['color']}">
                    <div style="font-size:1.6rem;font-weight:700;color:{meta['color']}">
                        {meta['icon']} {meta['label']}
                    </div>
                    <div style="color:#475569;margin-top:5px;font-size:0.9rem">
                        {meta['description']}
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )

            # ── Gauge + Radar ─────────────────────────────────────────────────
            gcol, rcol = st.columns(2)
            with gcol:
                st.plotly_chart(gauge_chart(confidence, category), use_container_width=True)
            with rcol:
                st.plotly_chart(radar_chart(all_scores, category), use_container_width=True)

            # ── Explanation ───────────────────────────────────────────────────
            st.info(f"💡 {result['brief_explanation']}")

            # ── Suggested action ──────────────────────────────────────────────
            st.markdown(
                f'<div class="action-badge">🎯 <b>Suggested next step:</b> {result["suggested_action"]}</div>',
                unsafe_allow_html=True,
            )

            # ── Key indicators ────────────────────────────────────────────────
            if result["key_indicators"]:
                st.markdown("**🔍 Key Indicators Detected:**")
                chips = "".join(
                    f'<span class="indicator-chip">🔹 {ind}</span>'
                    for ind in result["key_indicators"]
                )
                st.markdown(chips, unsafe_allow_html=True)

            # ── Resources ─────────────────────────────────────────────────────
            st.markdown("**📚 Resources & Recommended Actions:**")
            for title, desc in meta["resources"]:
                st.markdown(
                    f'<div class="resource-card">'
                    f'<div class="resource-title" style="color:{meta["color"]}">{title}</div>'
                    f'<div class="resource-desc">{desc}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.toast(f"{meta['icon']} {meta['label']} — {confidence}% confidence", icon="✅")

        else:
            st.markdown(
                """<div class="empty-state">
                    <div class="empty-state-icon">🧠</div>
                    <div class="empty-state-text">Type or paste text on the left, then click <b>Analyse Sentiment</b></div>
                </div>""",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Batch Analysis
# ══════════════════════════════════════════════════════════════════════════════

with tab_batch:
    st.markdown("#### 📋 Batch Analysis")
    st.caption("Enter one social media post per line. All posts are classified by the BERT model.")

    bcol1, bcol2 = st.columns([3, 1])
    with bcol1:
        batch_input = st.text_area(
            "Posts (one per line):", height=180,
            placeholder=(
                "I can't sleep and feel so anxious all the time...\n"
                "Had a lovely day with friends at the park!\n"
                "Everything feels pointless lately, nothing matters..."
            ),
        )
    with bcol2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if batch_input.strip():
            line_count = len([l for l in batch_input.strip().splitlines() if l.strip()])
            st.metric("Posts detected", line_count)

    batch_btn = st.button("🔍 Analyse All Posts", type="primary", disabled=not batch_input.strip())

    if batch_btn and batch_input.strip():
        lines = [l.strip() for l in batch_input.strip().splitlines() if l.strip()]
        prog = st.progress(0, text=f"Analysing post 1 of {len(lines)}…")
        batch_results = []

        for i, line in enumerate(lines):
            prog.progress(i / len(lines), text=f"Analysing post {i + 1} of {len(lines)}…")
            res = classify_text(line)
            add_history(line, res)
            batch_results.append({
                "#": i + 1,
                "Post Preview":     line[:80] + ("…" if len(line) > 80 else ""),
                "Category":         f"{CATEGORY_META[res['category']]['icon']} {CATEGORY_META[res['category']]['label']}",
                "Confidence (%)":   res["confidence"],
                "Key Indicators":   ", ".join(res.get("key_indicators", [])),
                "Suggested Action": res.get("suggested_action", ""),
            })

        prog.progress(1.0, text="Done!")

        df = pd.DataFrame(batch_results)

        exp_col, _ = st.columns([1, 3])
        with exp_col:
            st.download_button(
                "📥 Export Results (CSV)",
                data=export_csv(batch_results),
                file_name=f"batch_bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        st.dataframe(
            df, use_container_width=True, hide_index=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", min_value=0, max_value=100
                ),
            },
        )

        # Crisis / high-risk callout
        flagged = [r for r in batch_results if "Suicidal" in r["Category"]]
        if flagged:
            st.error(
                f"⚠️ **{len(flagged)} post(s) flagged as high-risk** "
                "(Suicidal / Bipolar / Personality Disorder). Please review immediately.",
                icon="🆘",
            )

        # Summary charts
        st.markdown("#### 📊 Batch Summary")
        summary = df["Category"].value_counts().reset_index()
        summary.columns = ["Category", "Count"]
        lbl_color = {f"{m['icon']} {m['label']}": m["color"] for m in CATEGORY_META.values()}

        sc1, sc2 = st.columns(2)
        with sc1:
            fig_b = px.bar(
                summary, x="Category", y="Count",
                color="Category",
                color_discrete_sequence=[lbl_color.get(c, "#888") for c in summary["Category"]],
                title="Distribution",
            )
            fig_b.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#1e293b", showlegend=False, height=300,
                yaxis=dict(gridcolor="#e2e8f0"), margin=dict(t=40),
            )
            st.plotly_chart(fig_b, use_container_width=True)
        with sc2:
            fig_p = go.Figure(go.Pie(
                labels=summary["Category"], values=summary["Count"],
                marker_colors=[lbl_color.get(c, "#888") for c in summary["Category"]],
                hole=0.45, textinfo="label+percent",
            ))
            fig_p.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", font_color="#1e293b",
                showlegend=False, height=300, margin=dict(t=40),
                title=dict(text="Proportion", font=dict(size=14)),
            )
            st.plotly_chart(fig_p, use_container_width=True)

        st.toast(f"✅ Batch complete — {len(lines)} posts analysed.", icon="📋")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Statistics
# ══════════════════════════════════════════════════════════════════════════════

with tab_stats:
    st.markdown("#### 📊 Session Statistics")

    if not st.session_state["history"]:
        st.markdown(
            """<div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-text">Run some analyses first to see statistics here</div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        history    = st.session_state["history"]
        total      = len(history)
        avg_conf   = round(sum(h["confidence"] for h in history) / total, 1)
        most_common = max(CATEGORIES, key=lambda c: sum(1 for h in history if h["category"] == c))
        high_risk  = sum(1 for h in history if h["category"] in ("suicidal", "depression"))

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Analyses", total)
        k2.metric("Avg Confidence", f"{avg_conf}%")
        k3.metric("Most Detected",
                  f"{CATEGORY_META[most_common]['icon']} {CATEGORY_META[most_common]['label']}")
        k4.metric("High-Risk Flags", high_risk)

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(history_bar_chart(history), use_container_width=True)
        with c2:
            st.plotly_chart(history_pie_chart(history), use_container_width=True)

        st.plotly_chart(timeline_chart(history), use_container_width=True)

        st.markdown("#### 📄 Analysis Log")
        log_records = [
            {
                "Timestamp":      h["timestamp"],
                "Category":       f"{CATEGORY_META[h['category']]['icon']} {CATEGORY_META[h['category']]['label']}",
                "Confidence (%)": h["confidence"],
                "Key Indicators": ", ".join(h.get("key_indicators", [])),
                "Explanation":    h["brief_explanation"],
                "Suggested Action": h.get("suggested_action", ""),
                "Preview":        h["text_preview"],
            }
            for h in reversed(history)
        ]
        df_log = pd.DataFrame(log_records)

        st.download_button(
            "📥 Export Full Log (CSV)",
            data=export_csv(log_records),
            file_name=f"mh_bert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

        st.dataframe(
            df_log, use_container_width=True, hide_index=True,
            column_config={
                "Confidence (%)": st.column_config.ProgressColumn(
                    "Confidence (%)", min_value=0, max_value=100
                ),
            },
        )
