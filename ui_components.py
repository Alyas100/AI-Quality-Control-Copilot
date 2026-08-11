"""
UI Components
================================================================================
Custom HTML/CSS components used to give the Streamlit app a polished,
non-default look: metric cards, risk banners, ripeness badges, a lightweight
chat interface for the Copilot, and Plotly figures for the FFA gauge and
feature-importance chart.

Implementation note on "Tailwind CSS utility classes": Streamlit's
`st.markdown(html, unsafe_allow_html=True)` inserts HTML via innerHTML, and
browsers do not execute <script> tags inserted that way -- so the Tailwind
CDN's JIT compiler script never actually runs if dropped in via st.markdown.
Rather than ship a visual that silently fails to style itself, this module
defines a small hand-written utility stylesheet using Tailwind's own
color/spacing/radius scale (same design system, same look) and injects it
once as a single <style> block, which browsers DO apply regardless of how
it was inserted. Plain HTML + these classes gives the intended Tailwind-style
result reliably.
"""

import html
import re

import plotly.graph_objects as go

from ml_engine import FEATURE_LABELS

# ------------------------------------------------------------------ styles

GLOBAL_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
  --slate-50:#f8fafc; --slate-100:#f1f5f9; --slate-200:#e2e8f0; --slate-400:#94a3b8;
  --slate-600:#475569; --slate-800:#1e293b; --slate-900:#0f172a;
  --amber-500:#f59e0b; --amber-50:#fffbeb;
  --emerald-500:#10b981; --emerald-50:#ecfdf5;
  --red-500:#ef4444; --red-50:#fef2f2;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.qc-heading { font-family: 'Sora', sans-serif; }

.qc-hero {
  background: linear-gradient(120deg, #0f172a 0%, #1e293b 55%, #7c2d12 130%);
  border-radius: 20px; padding: 2.1rem 2.4rem; margin-bottom: 1.4rem;
  box-shadow: 0 10px 30px -12px rgba(15,23,42,0.45);
}
.qc-hero h1 { font-family:'Sora',sans-serif; color: white; font-size: 1.9rem; font-weight: 800; margin: 0 0 .4rem 0; }
.qc-hero p { color: #cbd5e1; font-size: .95rem; margin: 0; max-width: 640px; }
.qc-badge {
  display:inline-block; background:rgba(245,158,11,0.15); color:#fbbf24;
  border:1px solid rgba(245,158,11,0.4); padding:.28rem .75rem; border-radius:999px;
  font-size:.7rem; font-weight:700; letter-spacing:.06em; margin-bottom:1rem;
}

.qc-card {
  background:white; border:1px solid var(--slate-200); border-radius:16px;
  padding:1.1rem 1.25rem; box-shadow:0 1px 3px rgba(15,23,42,0.06); height:100%;
}
.qc-card .qc-icon { font-size:1.25rem; }
.qc-card .qc-label { font-size:.74rem; font-weight:600; color:var(--slate-400); text-transform:uppercase; letter-spacing:.05em; margin-top:.3rem; }
.qc-card .qc-value { font-family:'Sora',sans-serif; font-size:1.65rem; font-weight:700; color:var(--slate-900); margin-top:.1rem; line-height:1.2; }
.qc-card .qc-sub { font-size:.78rem; color:var(--slate-600); margin-top:.3rem; }

.qc-banner { border-radius:14px; padding:1rem 1.3rem; display:flex; gap:.8rem; align-items:flex-start; border-left:5px solid; }
.qc-banner.green { background:var(--emerald-50); border-color:var(--emerald-500); }
.qc-banner.amber { background:var(--amber-50); border-color:var(--amber-500); }
.qc-banner.red   { background:var(--red-50);   border-color:var(--red-500); }
.qc-banner .qc-banner-icon { font-size:1.4rem; line-height:1.4rem; }
.qc-banner .qc-banner-title { font-family:'Sora',sans-serif; font-weight:700; font-size:.95rem; }
.qc-banner.green .qc-banner-title { color:#065f46; }
.qc-banner.amber .qc-banner-title { color:#92400e; }
.qc-banner.red   .qc-banner-title { color:#991b1b; }
.qc-banner .qc-banner-body { font-size:.86rem; color:var(--slate-600); margin-top:.2rem; line-height:1.45; }

.qc-pill { display:inline-flex; align-items:center; gap:.45rem; padding:.45rem .9rem; border-radius:999px; font-weight:600; font-size:.85rem; color:white; }

.qc-chat-msg { border-radius:14px; padding:.85rem 1.05rem; margin-bottom:.7rem; font-size:.9rem; line-height:1.55; }
.qc-chat-user { background:var(--slate-800); color:white; margin-left:2.2rem; }
.qc-chat-ai   { background:white; border:1px solid var(--slate-200); margin-right:1.2rem; }
.qc-chat-role { font-size:.7rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; opacity:.65; margin-bottom:.35rem; }

[data-testid="stSidebar"] { background:var(--slate-50); }
div.block-container { padding-top:1.6rem; }
</style>
"""


# ------------------------------------------------------------------ HTML snippet builders

def metric_card(icon: str, label: str, value, sub: str = "", accent: str = None) -> str:
    style = f"border-top:3px solid {accent};" if accent else ""
    return f"""<div class="qc-card" style="{style}">
      <div class="qc-icon">{icon}</div>
      <div class="qc-label">{html.escape(str(label))}</div>
      <div class="qc-value">{html.escape(str(value))}</div>
      <div class="qc-sub">{html.escape(str(sub))}</div>
    </div>"""


def risk_banner(risk: dict, extra: str = "") -> str:
    return f"""<div class="qc-banner {risk['color']}">
      <div class="qc-banner-icon">{risk['icon']}</div>
      <div>
        <div class="qc-banner-title">{risk['level'].upper()} RISK</div>
        <div class="qc-banner-body">{html.escape(risk['message'])} {html.escape(extra)}</div>
      </div>
    </div>"""


def ripeness_badge(category: str, meta: dict, confidence: float) -> str:
    return (
        f'<span class="qc-pill" style="background:{meta["color"]};">'
        f'{meta["emoji"]} {html.escape(category)} &middot; {confidence * 100:.0f}% confidence</span>'
    )


def format_llm_text(text: str) -> str:
    """Escape LLM output for safe HTML embedding, then re-enable the light
    **bold** markdown we asked the model to use."""
    escaped = html.escape(text)
    formatted = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    formatted = formatted.replace("\n", "<br>")
    return formatted


def chat_bubble(role: str, content: str, already_formatted: bool = False) -> str:
    cls = "qc-chat-user" if role == "user" else "qc-chat-ai"
    role_label = "You &rarr; Copilot" if role == "user" else "🤖 AI Copilot"
    body = content if already_formatted else html.escape(content)
    return f"""<div class="qc-chat-msg {cls}">
      <div class="qc-chat-role">{role_label}</div>
      <div>{body}</div>
    </div>"""


# ------------------------------------------------------------------ Plotly figures

def build_ffa_gauge(predicted_ffa: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_ffa,
        number={"suffix": "%", "font": {"size": 34, "family": "Sora, sans-serif", "color": "#0f172a"}},
        gauge={
            "axis": {"range": [0, 6], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": "#0f172a", "thickness": 0.28},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 2.5], "color": "#a7f3d0"},
                {"range": [2.5, 3.5], "color": "#fde68a"},
                {"range": [3.5, 6], "color": "#fecaca"},
            ],
            "threshold": {"line": {"color": "#dc2626", "width": 3}, "thickness": 0.85, "value": 3.5},
        },
    ))
    fig.update_layout(
        height=220,               # Give the gauge explicit vertical breathing room
        margin=dict(l=30, r=30, t=50, b=10),  # Add padding so the arch doesn't clip
        
        # This controls where the big text sits relative to the arch
        annotations=[dict(
            text=f"{predicted_ffa:.2f}%",
            x=0.5, y=0.15,        # Lower the text position down away from the arc line
            font=dict(size=28, color="#0F172A", weight="bold"),
            showarrow=False
        )]
    )
    return fig


def build_importance_chart(importances: dict) -> go.Figure:
    labels = [FEATURE_LABELS[k] for k in importances]
    values = [round(v * 100, 1) for v in importances.values()]
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#d97706",
        text=[f"{v}%" for v in values], textposition="outside",
    ))
    fig.update_layout(
        height=210, margin=dict(l=10, r=35, t=10, b=25),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, visible=False, range=[0, max(values) * 1.25]),
        font={"family": "IBM Plex Sans, sans-serif", "color": "#1e293b", "size": 12},
    )
    return fig
