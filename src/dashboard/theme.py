"""
Shared design tokens for the Dash and Streamlit dashboards — one visual
identity, two frameworks. Change a color here, both apps update.

Design direction: an industrial supply-chain control tower. Dark graphite
(not pure black), two accents drawn from the material world of the
domain (construction safety-amber, structural steel-blue), and a 4-color
semantic risk system used consistently everywhere — the "signal" system
that is this design's signature element.
"""

COLORS = {
    "bg": "#14161A",
    "surface": "#1B1E24",
    "surface_alt": "#22262E",
    "border": "#2A2F38",
    "text": "#EDEFF2",
    "text_muted": "#8B93A1",
    "accent_amber": "#F2A93B",   # brand / primary
    "accent_blue": "#4FA8E0",    # data / forecast lines
    "danger": "#E5484D",         # stockout
    "warning": "#F2A93B",        # low stock
    "success": "#34D399",        # healthy / normal
    "overcapacity": "#A78BFA",   # near/at capacity
}

RISK_COLORS = {
    "Stockout": COLORS["danger"],
    "Low Stock": COLORS["warning"],
    "Overcapacity": COLORS["overcapacity"],
    "Normal": COLORS["success"],
}

FONTS = {
    "display": "'Space Grotesk', sans-serif",
    "body": "'Inter', sans-serif",
    "mono": "'IBM Plex Mono', monospace",
}

GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Space+Grotesk:wght@500;600;700&"
    "family=Inter:wght@400;500;600&"
    "family=IBM+Plex+Mono:wght@400;500;600&display=swap"
)


def plotly_layout_defaults(title: str = "") -> dict:
    """Common Plotly layout kwargs matching the dark control-tower theme."""
    return dict(
        title=dict(text=title, font=dict(family=FONTS["display"], size=15, color=COLORS["text"])),
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font=dict(family=FONTS["body"], color=COLORS["text_muted"], size=12),
        xaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], color=COLORS["text_muted"]),
        yaxis=dict(gridcolor=COLORS["border"], zerolinecolor=COLORS["border"], color=COLORS["text_muted"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text_muted"])),
        margin=dict(l=50, r=30, t=50, b=40),
    )
