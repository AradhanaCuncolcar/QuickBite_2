# app.py — QuickBite Diner Intelligence (v3)
# ------------------------------------------------------------
# Run locally:   streamlit run app.py
# Needs (same folder): food_models.joblib, diners.csv, orders.csv
# ------------------------------------------------------------
import itertools
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import streamlit as st

st.set_page_config(page_title="QuickBite Diner Intelligence", page_icon="🍔", layout="wide")
HERE = Path(__file__).parent

# ---------- design tokens ----------
INK, MUTED, PAPER, LINE = "#18212F", "#5B6576", "#F4F6F9", "#DDE2EA"
RISK, WATCH, SAFE = "#D7263D", "#E08A00", "#0F9D7A"
SEG_META = {
    "Families":        dict(color="#E39B0B", emoji="👨‍👩‍👧", blurb="Big, infrequent family orders. Highest basket value, lowest loyalty."),
    "Budget Students": dict(color="#3159D6", emoji="🎒", blurb="Young, frequent, small orders. Discount-driven and very loyal."),
    "Premium Foodies": dict(color="#6A8F3A", emoji="🍣", blurb="Quality over price. Rarely use discounts, highest lifetime spend."),
}
FALLBACK = ["#8E44AD", "#16A085", "#C0392B", "#2C3E50"]
ITEM_EMOJI = {"pizza": "🍕", "burger": "🍔", "fries": "🍟", "cola": "🥤", "wrap": "🌯", "salad": "🥗",
              "sushi": "🍣", "coffee": "☕", "dessert": "🍰", "pasta": "🍝", "wine": "🍷", "biryani": "🍛",
              "raita": "🥣", "steak": "🥩", "green_tea": "🍵", "miso_soup": "🍜", "garlic_bread": "🥖"}
LABELS = {"orders_per_month": "Orders per month", "avg_order_value": "Avg order value",
          "avg_items_per_order": "Items per order", "days_since_last_order": "Days since last order",
          "discount_rate": "Discount usage", "age": "Age", "total_spend": "Total spend"}


def item_name(i):
    return f"{ITEM_EMOJI.get(i, '🍽️')} {i.replace('_', ' ').capitalize()}"


def seg_color(name, names):
    return SEG_META.get(name, {}).get("color") or FALLBACK[list(names).index(name) % len(FALLBACK)]


def prob_color(p, thr):
    return SAFE if p >= max(thr, 0.7) else (WATCH if p >= thr else RISK)


# ---------- styles ----------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Figtree:wght@400;500;600;700&display=swap');
.stApp {{ background:{PAPER}; }}
.block-container {{ padding-top:3.6rem; max-width:1250px; }}
h1, h2, h3, h4 {{ font-family:'Bricolage Grotesque', 'Figtree', system-ui, sans-serif !important; color:{INK}; letter-spacing:-0.01em; }}
.qb-body, .qb-body * {{ font-family:'Figtree', system-ui, sans-serif; }}
[data-testid="stSidebar"] {{ background:#FFFFFF; border-right:1px solid {LINE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap:4px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; padding:10px 16px; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid {LINE}; border-radius:10px; padding:14px 16px; }}
[data-testid="stMetricValue"] {{ font-family:'Bricolage Grotesque', sans-serif; font-weight:700; }}

.qb-head h1 {{ font-size:2.4rem; font-weight:800; margin:0; line-height:1.05; padding:0; }}
.qb-head p {{ color:{MUTED}; margin:6px 0 4px; max-width:62ch; font-size:1.02rem; }}

/* the order ticket: the one bold element on the page.
   Two halves with masked corner cut-outs, so whatever is behind (including the animated background) shows through. */
.qb-ticket {{ filter:drop-shadow(0 12px 18px rgba(24,33,47,.16)) drop-shadow(0 1px 0 {LINE}); }}
.qb-ticket .upper, .qb-ticket .lower {{ background:#fff; position:relative; }}
.qb-ticket .upper {{ border-radius:14px 14px 0 0; overflow:hidden;
  -webkit-mask:radial-gradient(circle 11px at 0 100%, #0000 98%, #000) left / 51% 100% no-repeat,
               radial-gradient(circle 11px at 100% 100%, #0000 98%, #000) right / 51% 100% no-repeat;
          mask:radial-gradient(circle 11px at 0 100%, #0000 98%, #000) left / 51% 100% no-repeat,
               radial-gradient(circle 11px at 100% 100%, #0000 98%, #000) right / 51% 100% no-repeat; }}
.qb-ticket .lower {{ border-radius:0 0 14px 14px;
  -webkit-mask:radial-gradient(circle 11px at 0 0, #0000 98%, #000) left / 51% 100% no-repeat,
               radial-gradient(circle 11px at 100% 0, #0000 98%, #000) right / 51% 100% no-repeat;
          mask:radial-gradient(circle 11px at 0 0, #0000 98%, #000) left / 51% 100% no-repeat,
               radial-gradient(circle 11px at 100% 0, #0000 98%, #000) right / 51% 100% no-repeat; }}
.qb-ticket .lower:before {{ content:''; position:absolute; top:0; left:18px; right:18px; border-top:2px dashed {LINE}; }}
.qb-ticket .strip {{ height:8px; }}
.qb-ticket .top {{ padding:18px 22px 18px; display:grid; grid-template-columns:1fr auto; gap:18px; align-items:start; }}
.qb-ticket .seg {{ font-family:'Bricolage Grotesque',sans-serif; font-size:1.8rem; font-weight:800; color:{INK}; line-height:1.1; }}
.qb-ticket .blurb {{ color:{MUTED}; font-size:.95rem; margin-top:6px; max-width:40ch; }}
.qb-ticket .prob {{ text-align:right; }}
.qb-ticket .prob .num {{ font-family:'Bricolage Grotesque',sans-serif; font-size:3.6rem; font-weight:800; line-height:.95; }}
.qb-ticket .prob .lab {{ color:{MUTED}; font-size:.9rem; margin-top:4px; }}
.qb-ticket .bottom {{ padding:16px 22px 18px; display:grid; grid-template-columns:1fr 1fr; gap:14px 18px; }}

/* order replay strip */
.qb-replay {{ display:flex; align-items:center; gap:12px; background:rgba(255,255,255,.78); border:1px solid {LINE};
             border-radius:999px; padding:6px 6px 6px 14px; margin:12px 0 6px; overflow:hidden;
             -webkit-backdrop-filter:blur(8px); backdrop-filter:blur(8px); }}
.qb-replay .tag {{ display:flex; align-items:center; gap:7px; font-weight:700; font-size:.85rem; color:{INK}; white-space:nowrap; }}
.qb-replay .dot {{ width:8px; height:8px; border-radius:50%; background:{RISK}; animation:qb-pulse 1.6s ease-out infinite; }}
.qb-replay .window {{ overflow:hidden; flex:1; -webkit-mask:linear-gradient(90deg,#0000,#000 4%,#000 96%,#0000);
                     mask:linear-gradient(90deg,#0000,#000 4%,#000 96%,#0000); }}
.qb-replay .track {{ display:flex; gap:10px; width:max-content; animation:qb-scroll 90s linear infinite; }}
.qb-replay:hover .track {{ animation-play-state:paused; }}
.qb-replay .ord {{ white-space:nowrap; font-size:.88rem; color:{INK}; background:#fff; border:1px solid {LINE};
                  border-left:4px solid; border-radius:8px; padding:3px 10px; }}
.qb-replay .ord span {{ color:{MUTED}; margin-left:6px; }}
@keyframes qb-scroll {{ from {{ transform:translateX(0); }} to {{ transform:translateX(-50%); }} }}
@keyframes qb-pulse {{ 0% {{ box-shadow:0 0 0 0 rgba(215,38,61,.55); }} 100% {{ box-shadow:0 0 0 9px rgba(215,38,61,0); }} }}
@media (prefers-reduced-motion: reduce) {{ .qb-replay .track, .qb-replay .dot {{ animation:none; }} }}

/* insight cards in the Insights tab */
.qb-finding {{ font-family:'Bricolage Grotesque',sans-serif; font-weight:700; font-size:1.18rem; color:{INK}; line-height:1.3; margin:0 0 2px; }}
.qb-finding-sub {{ color:{MUTED}; font-size:.92rem; margin-bottom:4px; line-height:1.45; }}
.qb-k {{ color:{MUTED}; font-size:.85rem; margin-bottom:3px; }}
.qb-v {{ color:{INK}; font-weight:600; }}
.qb-pill {{ display:inline-block; padding:3px 11px; border-radius:999px; font-weight:700; font-size:.85rem; color:#fff; }}
.qb-chip {{ display:inline-block; background:{PAPER}; border:1px solid {LINE}; border-radius:999px; padding:4px 11px;
           margin:0 6px 6px 0; font-size:.92rem; color:{INK}; }}
.qb-action {{ border-left:4px solid; background:#fff; border-radius:0 10px 10px 0; padding:12px 16px; margin-top:14px; color:{INK}; line-height:1.5; }}
.qb-action b {{ font-family:'Bricolage Grotesque',sans-serif; }}
.qb-card {{ background:#fff; border:1px solid {LINE}; border-radius:12px; padding:16px 18px; height:100%; }}
.qb-card h4 {{ margin:0 0 4px; font-size:1.15rem; padding:0; }}
.qb-card .sub {{ color:{MUTED}; font-size:.9rem; margin-bottom:10px; line-height:1.4; }}
.qb-row {{ display:flex; justify-content:space-between; border-top:1px solid {LINE}; padding:6px 0; font-size:.93rem; }}
.qb-row span:first-child {{ color:{MUTED}; }} .qb-row span:last-child {{ font-weight:600; color:{INK}; }}
.qb-insight {{ background:#fff; border:1px solid {LINE}; border-left:4px solid {INK}; border-radius:0 12px 12px 0;
              padding:14px 18px; color:{INK}; margin:6px 0 14px; line-height:1.55; }}
.qb-insight b {{ font-family:'Bricolage Grotesque',sans-serif; }}
@media (max-width: 700px) {{ .qb-ticket .bottom, .qb-ticket .top {{ grid-template-columns:1fr; }} .qb-ticket .prob {{ text-align:left; }} .qb-replay .tag span.full {{ display:none; }} }}
</style>
""", unsafe_allow_html=True)

# ---------- animated background ----------
# Slow-drifting light in the three segment colours. It sits behind everything (z-index 0, no pointer events),
# pauses for people who ask their OS for reduced motion, and can be switched off in the sidebar.
ANIMATED_BG = f"""
<style>
.stApp {{ background:{PAPER}; }}
.stApp::before, .stApp::after {{ content:''; position:fixed; inset:-25%; z-index:0; pointer-events:none; will-change:transform; }}
.stApp::before {{
  background:
    radial-gradient(30% 34% at 18% 22%, rgba(227,155,11,.42), transparent 70%),
    radial-gradient(28% 32% at 82% 30%, rgba(49,89,214,.32), transparent 70%);
  animation:qb-drift-a 18s ease-in-out infinite alternate, qb-breathe 9s ease-in-out infinite; }}
.stApp::after {{
  background:
    radial-gradient(32% 36% at 62% 82%, rgba(106,143,58,.36), transparent 70%),
    radial-gradient(22% 26% at 30% 70%, rgba(215,38,61,.14), transparent 70%);
  animation:qb-drift-b 24s ease-in-out infinite alternate, qb-breathe 11s ease-in-out -4s infinite; }}
@keyframes qb-drift-a {{
  0%   {{ transform:translate3d(0,0,0) rotate(0deg) scale(1); }}
  50%  {{ transform:translate3d(9%,6%,0) rotate(14deg) scale(1.12); }}
  100% {{ transform:translate3d(-4%,7%,0) rotate(-6deg) scale(1.04); }} }}
@keyframes qb-drift-b {{
  0%   {{ transform:translate3d(0,0,0) rotate(0deg) scale(1.05); }}
  50%  {{ transform:translate3d(-10%,-7%,0) rotate(-16deg) scale(1); }}
  100% {{ transform:translate3d(5%,-8%,0) rotate(8deg) scale(1.1); }} }}
@keyframes qb-breathe {{ 0%, 100% {{ opacity:.75; }} 50% {{ opacity:1; }} }}
[data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background:transparent; position:relative; z-index:1; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:rgba(255,255,255,.82); -webkit-backdrop-filter:blur(14px); backdrop-filter:blur(14px); }}
[data-testid="stMetric"], .qb-card, .qb-insight, .qb-action {{ background:rgba(255,255,255,.9);
  -webkit-backdrop-filter:blur(10px); backdrop-filter:blur(10px); }}
@media (prefers-reduced-motion: reduce) {{ .stApp::before, .stApp::after {{ animation:none; }} }}
</style>
"""
if st.session_state.get("animate_bg", True):
    st.markdown(ANIMATED_BG, unsafe_allow_html=True)


def chart_layout(fig, height=320, legend=True):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Figtree, sans-serif", color=INK, size=13),
                      showlegend=legend, legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(gridcolor=LINE, zerolinecolor=LINE)
    fig.update_yaxes(gridcolor=LINE, zerolinecolor=LINE)
    return fig


def show(fig, key):
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False}, key=key)


# ---------- models & data ----------
@st.cache_resource
def load_models():
    return joblib.load(HERE / "food_models.joblib")


M = load_models()
scaler, kmeans = M["scaler"], M["kmeans"]
seg_names, clf, FEATURES, reco = M["segment_names"], M["clf"], M["features"], M["reco"]
CLF_FEATURES = FEATURES + ["age", "total_spend"]
SEGMENTS = [seg_names[k] for k in sorted(seg_names)]


def segment_of(df):
    labels = kmeans.predict(scaler.transform(df[FEATURES]))
    return np.array([seg_names.get(int(c), f"Segment {c}") for c in labels])


def reorder_prob(df):
    return clf.predict_proba(df[CLF_FEATURES])[:, 1]


@st.cache_data
def load_data():
    if not (HERE / "diners.csv").exists() or not (HERE / "orders.csv").exists():
        return None, None
    d = pd.read_csv(HERE / "diners.csv")
    o = pd.read_csv(HERE / "orders.csv")
    d["segment"] = segment_of(d)
    d["p_reorder"] = reorder_prob(d)
    d["monthly_value"] = d.orders_per_month * d.avg_order_value
    d["value_at_risk"] = (1 - d.p_reorder) * d.monthly_value
    o = o.merge(d[["diner_id", "segment"]], on="diner_id", how="left")
    return d, o


@st.cache_data
def basket_stats(o, segment):
    """Pairwise rules + best 2–3 item bundles, computed directly (no mlxtend needed)."""
    sub = o if segment == "All diners" else o[o.segment == segment]
    baskets = sub.groupby("order_id").item.apply(lambda s: tuple(sorted(set(s))))
    n = len(baskets)
    counts = {}
    for b in baskets:
        for k in (1, 2, 3):
            for combo in itertools.combinations(b, k):
                counts[combo] = counts.get(combo, 0) + 1
    sup = {c: v / n for c, v in counts.items()}
    rules = []
    for c, s in sup.items():
        if len(c) == 2:
            for a, b in (c, c[::-1]):
                conf = s / sup[(a,)]
                rules.append(dict(antecedent=a, consequent=b, support=s, confidence=conf, lift=conf / sup[(b,)]))
    rules = pd.DataFrame(rules)
    min_sup = 0.02 if segment == "All diners" else 0.08
    cand = [dict(items=c, support=s, lift=s / np.prod([sup[(i,)] for i in c]),
                 together=s / max(sup[(i,)] for i in c))   # all-confidence: how often the full set travels together
            for c, s in sup.items() if len(c) >= 2 and s >= min_sup]
    bundles = pd.DataFrame(cand)
    if len(bundles):
        rank = "together" if segment == "All diners" else "lift"
        bundles = bundles[bundles.lift > 1].sort_values([rank, "support"], ascending=False)
        keep, seen = [], []
        for _, r in bundles.iterrows():          # skip bundles that overlap one already shown
            if not any(set(r["items"]) & s for s in seen):
                keep.append(r)
                seen.append(set(r["items"]))
        bundles = pd.DataFrame(keep)
        bundles = bundles[(bundles.lift >= 1.5) | (bundles.index == bundles.index[0])]   # drop weak bundles, keep at least one
    items = sorted({c[0] for c in sup if len(c) == 1})
    return n, rules, bundles, items


def recency_curve(profile, days=range(0, 61)):
    rows = pd.DataFrame([{**profile, "days_since_last_order": d} for d in days])
    return np.array(list(days)), reorder_prob(rows)


def deadline(profile, thr):
    x, y = recency_curve(profile)
    below = x[y < thr]
    return int(below[0]) if len(below) else None


def drivers(profile):
    """Per-feature push toward reorder (+) or churn (−) for a scaled linear model."""
    try:
        est = clf[-1]
        z = clf[:-1].transform(pd.DataFrame([profile])[CLF_FEATURES])[0]
        return pd.Series(est.coef_[0] * z, index=CLF_FEATURES)
    except Exception:
        return None


diners, orders = load_data()
seg_bundle = {}
if orders is not None:
    for s in SEGMENTS:
        b = basket_stats(orders, s)[2]
        if len(b):
            seg_bundle[s] = b.iloc[0]["items"]


@st.cache_data
def replay_strip(o, n=36, seed=7):
    """A looping strip of real baskets from orders.csv. Built once and cached, so it doesn't restart on every rerun."""
    ids = pd.Series(o.order_id.unique()).sample(n, random_state=seed)
    chips = []
    for oid in ids:
        b = o[o.order_id == oid]
        seg_ = b.segment.iloc[0] if "segment" in b and pd.notna(b.segment.iloc[0]) else ""
        items = ", ".join(i.replace("_", " ") for i in b["item"])
        icons = "".join(ITEM_EMOJI.get(i, "🍽️") for i in b["item"])
        col = seg_color(seg_, SEGMENTS) if seg_ in SEGMENTS else LINE
        chips.append(f'<span class="ord" style="border-left-color:{col}">{icons} {items} <span>{seg_}</span></span>')
    track = "".join(chips) * 2          # doubled so the loop is seamless
    return f"""<div class="qb-replay qb-body" title="Replaying real baskets from orders.csv. Hover to pause.">
  <div class="tag"><span class="dot"></span>Order replay</div>
  <div class="window"><div class="track">{track}</div></div></div>"""


@st.cache_data
def cv_scores(d):
    """Out-of-fold reorder probabilities: each diner is scored by a model that never saw them."""
    if "reordered" not in d:
        return None
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return cross_val_predict(clone(clf), d[CLF_FEATURES], d.reordered, cv=cv, method="predict_proba")[:, 1]


def bundle_text(seg):
    b = seg_bundle.get(seg)
    return " + ".join(i.replace("_", " ") for i in b) if b is not None else None


def next_best_action(seg, p, thr, days_left):
    bundle = bundle_text(seg) or "favourite-items"
    top = reco.get(seg, ["a favourite"])[0].replace("_", " ") if reco.get(seg) else "a favourite"
    if p < thr * 0.67:
        return RISK, "Win back now", f"Send a {bundle} bundle with a time-limited discount within 48 hours. Most diners like this one don't come back on their own."
    if p < thr:
        return WATCH, "Nudge this week", f"Send a reminder featuring {top}. A small incentive now is cheaper than a win-back later."
    if seg == "Budget Students":
        return SAFE, "Keep, don't discount", f"Already loyal. Skip discounts and cross-sell the {bundle} combo instead."
    msg = f"Upsell the {bundle} bundle at full price."
    if days_left is not None and days_left > 0:
        msg += f" Re-engage within {days_left} days, before their reorder chance drops below {thr:.0%}."
    return SAFE, "Grow this diner", msg


# ---------- sidebar ----------
PRESETS = {
    "Student":        dict(age=21, orders_per_month=14, avg_order_value=40,  avg_items_per_order=2, days_since_last_order=3,  discount_rate=0.70, total_spend=550),
    "Quiet family":   dict(age=42, orders_per_month=3,  avg_order_value=180, avg_items_per_order=6, days_since_last_order=20, discount_rate=0.30, total_spend=600),
    "Foodie":         dict(age=33, orders_per_month=9,  avg_order_value=100, avg_items_per_order=3, days_since_last_order=6,  discount_rate=0.05, total_spend=900),
    "Drifter":        dict(age=34, orders_per_month=6,  avg_order_value=110, avg_items_per_order=3, days_since_last_order=16, discount_rate=0.10, total_spend=750),
}
DEFAULT = dict(age=28, orders_per_month=8, avg_order_value=90, avg_items_per_order=3, days_since_last_order=10, discount_rate=0.30, total_spend=800)
for k, v in DEFAULT.items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### Diner profile")
    st.caption("Load an example diner, or set the sliders yourself.")
    bcols = st.columns(2)
    for i, name in enumerate(PRESETS):
        bcols[i % 2].button(name, on_click=apply_preset, args=(name,), width="stretch")
    st.slider("Age", 18, 70, key="age")
    st.slider("Orders per month", 1, 20, key="orders_per_month")
    st.slider("Average order value (AED)", 15, 250, key="avg_order_value")
    st.slider("Average items per order", 1, 10, key="avg_items_per_order")
    st.slider("Days since last order", 0, 60, key="days_since_last_order")
    st.slider("Discount usage rate", 0.0, 1.0, step=0.05, key="discount_rate")
    st.number_input("Total spend so far (AED)", 50, 20000, step=50, key="total_spend")
    st.divider()
    thr = st.slider("At-risk threshold", 0.3, 0.8, 0.6, 0.05,
                    help="Diners below this reorder chance are flagged. 0.6 caught 44 of 48 churners in testing vs 40 at 0.5, with the same accuracy.")
    st.toggle("Animated background", value=True, key="animate_bg",
              help="Turn off for screenshots or slower laptops. Motion also stops if your device asks for reduced motion.")

profile = {k: st.session_state[k] for k in DEFAULT}
row = pd.DataFrame([profile])
seg = segment_of(row)[0]
p = float(reorder_prob(row)[0])
color = seg_color(seg, SEGMENTS)
pcol = prob_color(p, thr)
dl = deadline(profile, thr)
days_left = None if dl is None else dl - profile["days_since_last_order"]

# ---------- header ----------
st.markdown("""
<div class="qb-head qb-body">
  <h1>QuickBite diner intelligence</h1>
  <p>Who your diners are, what they order together, and who is about to stop ordering.</p>
</div>""", unsafe_allow_html=True)
if orders is not None:
    st.markdown(replay_strip(orders), unsafe_allow_html=True)

(tab_lookup, tab_seg, tab_basket, tab_insight, tab_plan, tab_watch, tab_model) = st.tabs(
    ["🔎 Diner lookup", "👥 Segments", "🛒 Baskets", "💡 Insights", "🎯 Campaign planner", "⚠️ Churn watchlist", "🧪 Model check"])

# ================= TAB 1: DINER LOOKUP =================
with tab_lookup:
    status = "Likely to reorder" if p >= thr else "At risk of churning"
    if dl is None:
        clock = "Stays above the at-risk line even after 60 quiet days"
    elif days_left > 0:
        clock = f"{days_left} days left before the at-risk line (day {dl})"
    elif dl == 0:
        clock = "Below the at-risk line even after a fresh order"
    else:
        clock = f"Crossed the at-risk line {-days_left} days ago (day {dl})"
    monthly = profile["orders_per_month"] * profile["avg_order_value"]
    chips = "".join(f'<span class="qb-chip">{item_name(i)}</span>' for i in reco.get(seg, []))
    meta = SEG_META.get(seg, {})
    ac, atitle, amsg = next_best_action(seg, p, thr, days_left)

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown(f"""
<div class="qb-ticket qb-body">
 <div class="upper">
  <div class="strip" style="background:{color}"></div>
  <div class="top">
    <div>
      <div class="qb-k">Segment</div>
      <div class="seg">{meta.get('emoji', '🍽️')} {seg}</div>
      <div class="blurb">{meta.get('blurb', '')}</div>
    </div>
    <div class="prob">
      <div class="num" style="color:{pcol}">{p:.0%}</div>
      <div class="lab">chance of reordering</div>
      <div style="margin-top:8px"><span class="qb-pill" style="background:{pcol}">{status}</span></div>
    </div>
  </div>
 </div>
 <div class="lower">
  <div class="bottom">
    <div><div class="qb-k">Reorder clock</div><div class="qb-v">{clock}</div></div>
    <div><div class="qb-k">Monthly order value</div><div class="qb-v">AED {monthly:,.0f}, of which AED {(1-p)*monthly:,.0f} at risk</div></div>
    <div style="grid-column:1/-1"><div class="qb-k">Recommended items</div>{chips or '—'}</div>
  </div>
 </div>
</div>
<div class="qb-action qb-body" style="border-color:{ac}"><b>{atitle}.</b> {amsg}</div>
""", unsafe_allow_html=True)

    with right:
        st.markdown("#### What if they stay quiet?")
        x, y = recency_curve(profile)
        fig = go.Figure()
        fig.add_hrect(y0=0, y1=thr, fillcolor=RISK, opacity=0.06, line_width=0)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=INK, width=3),
                                 hovertemplate="Day %{x}: %{y:.0%}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[profile["days_since_last_order"]], y=[p], mode="markers",
                                 marker=dict(size=15, color=pcol, line=dict(color="#fff", width=2)),
                                 hovertemplate="Today: %{y:.0%}<extra></extra>"))
        fig.add_hline(y=thr, line_dash="dot", line_color=RISK,
                      annotation_text=f"At-risk line ({thr:.0%})", annotation_position="top right",
                      annotation_font_color=RISK)
        fig.update_yaxes(tickformat=".0%", range=[0, 1.04])
        fig.update_xaxes(title="Days since last order")
        show(chart_layout(fig, 290, legend=False), "whatif")
        st.caption("Same diner, every other input held fixed. Days since last order is the strongest churn signal in the model.")

    st.markdown("#### Why the model thinks so")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dr = drivers(profile)
        if dr is not None:
            dr = dr.sort_values()
            fig = go.Figure(go.Bar(x=dr.values, y=[LABELS[i] for i in dr.index], orientation="h",
                                   marker_color=[SAFE if v > 0 else RISK for v in dr.values],
                                   hovertemplate="%{y}: %{x:+.2f}<extra></extra>"))
            fig.update_xaxes(title="← pushes toward churn  |  pushes toward reorder →", zeroline=True,
                             zerolinewidth=2, zerolinecolor=INK)
            show(chart_layout(fig, 300, legend=False), "drivers")
            st.caption("How much each input moved this prediction, compared with an average diner.")
        else:
            st.info("The driver breakdown needs a scaled linear classifier (a scikit-learn pipeline ending in LogisticRegression).")
    with c2:
        if diners is not None:
            seg_df = diners[diners.segment == seg]
            feats = FEATURES + ["total_spend"]
            pct_d = [(diners[f] < profile[f]).mean() * 100 for f in feats]
            pct_s = [(diners[f] < seg_df[f].median()).mean() * 100 for f in feats]
            ys = list(range(len(feats)))
            fig = go.Figure()
            for i in ys:
                fig.add_shape(type="line", x0=0, x1=100, y0=i, y1=i, line=dict(color=LINE, width=6), layer="below")
            fig.add_trace(go.Scatter(x=pct_s, y=ys, mode="markers", name=f"Typical {seg.lower()}",
                                     marker=dict(symbol="line-ns", size=20, line=dict(width=4, color=color)),
                                     hovertemplate="Segment median: higher than %{x:.0f}% of diners<extra></extra>"))
            fig.add_trace(go.Scatter(x=pct_d, y=ys, mode="markers", name="This diner",
                                     marker=dict(size=14, color=INK, line=dict(color="#fff", width=2)),
                                     hovertemplate="This diner: higher than %{x:.0f}% of diners<extra></extra>"))
            fig.update_yaxes(tickvals=ys, ticktext=[LABELS[f] for f in feats], showgrid=False)
            fig.update_xaxes(range=[-3, 103], title="Percentile among all diners", ticksuffix="%")
            show(chart_layout(fig, 300), "percentiles")
            st.caption(f"Where this diner sits among all {len(diners)} diners, next to a typical member of {seg}.")

# ================= TAB 2: SEGMENTS =================
with tab_seg:
    if diners is None:
        st.warning("Add diners.csv and orders.csv next to app.py to see segment insights.")
    else:
        flagged = diners.p_reorder < thr
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Diners", f"{len(diners):,}")
        k2.metric("Monthly order value", f"AED {diners.monthly_value.sum()/1000:,.0f}K")
        k3.metric("Value at risk", f"AED {diners.value_at_risk.sum()/1000:,.0f}K",
                  f"{diners.value_at_risk.sum()/diners.monthly_value.sum():.0%} of total", delta_color="off", delta_arrow="off")
        k4.metric("Diners flagged at risk", f"{flagged.sum()}", f"{flagged.mean():.0%} of base", delta_color="off", delta_arrow="off")

        g = diners.groupby("segment")
        summary = pd.DataFrame({
            "n": g.size(), "aov": g.avg_order_value.mean(), "opm": g.orders_per_month.mean(),
            "reorder": g.reordered.mean() if "reordered" in diners else g.p_reorder.mean(),
            "value": g.monthly_value.sum(), "risk": g.value_at_risk.sum(), "age": g.age.mean(),
            "disc": g.discount_rate.mean()}).reindex(SEGMENTS)
        worst = summary.risk.idxmax()
        seg_dl = {s: deadline(diners[diners.segment == s][CLF_FEATURES].median().to_dict(), thr) for s in SEGMENTS}
        safest = summary.reorder.idxmax()
        wd = seg_dl[worst]

        st.markdown(f"""
<div class="qb-insight qb-body"><b>Where the money leaks.</b> {worst} account for AED {summary.loc[worst,'risk']/1000:,.0f}K
of the AED {summary.risk.sum()/1000:,.0f}K monthly value at risk ({summary.loc[worst,'risk']/summary.risk.sum():.0%}).
A typical diner in this segment falls below the at-risk line after only <b>{wd if wd is not None else '60+'} days</b> without an order,
so their win-back offers need to go out sooner than anyone else's. {safest} reorder {summary.loc[safest,'reorder']:.0%} of the time,
so discounts spent on them are mostly wasted.</div>
""", unsafe_allow_html=True)

        cols = st.columns(len(SEGMENTS), gap="medium")
        for c, s in zip(cols, SEGMENTS):
            r = summary.loc[s]
            items = "".join(f'<span class="qb-chip">{item_name(i)}</span>' for i in reco.get(s, [])[:4])
            dd = seg_dl[s]
            c.markdown(f"""
<div class="qb-card qb-body" style="border-top:6px solid {seg_color(s, SEGMENTS)}">
  <h4>{SEG_META.get(s, {}).get('emoji', '')} {s}</h4>
  <div class="sub">{SEG_META.get(s, {}).get('blurb', '')}</div>
  <div class="qb-row"><span>Diners</span><span>{int(r.n)} ({r.n/len(diners):.0%})</span></div>
  <div class="qb-row"><span>Average age</span><span>{r.age:.0f}</span></div>
  <div class="qb-row"><span>Orders per month</span><span>{r.opm:.1f}</span></div>
  <div class="qb-row"><span>Average order value</span><span>AED {r.aov:,.0f}</span></div>
  <div class="qb-row"><span>Discount usage</span><span>{r.disc:.0%}</span></div>
  <div class="qb-row"><span>Reorder rate</span><span>{r.reorder:.0%}</span></div>
  <div class="qb-row"><span>Win-back deadline</span><span>{'day ' + str(dd) if dd is not None else '60+ days'}</span></div>
  <div style="margin-top:10px">{items}</div>
</div>""", unsafe_allow_html=True)

        st.write("")
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("#### Order value at risk, by segment")
            fig = go.Figure()
            fig.add_trace(go.Bar(y=SEGMENTS, x=summary.value - summary.risk, orientation="h", name="Expected to stay",
                                 marker_color=SAFE, hovertemplate="AED %{x:,.0f}<extra>Expected to stay</extra>"))
            fig.add_trace(go.Bar(y=SEGMENTS, x=summary.risk, orientation="h", name="At risk",
                                 marker_color=RISK, hovertemplate="AED %{x:,.0f}<extra>At risk</extra>"))
            fig.update_layout(barmode="stack")
            fig.update_xaxes(tickprefix="AED ", tickformat=",.0f")
            show(chart_layout(fig, 300), "leak")
        with b:
            st.markdown("#### How fast each segment goes cold")
            fig = go.Figure()
            for s in SEGMENTS:
                x, y = recency_curve(diners[diners.segment == s][CLF_FEATURES].median().to_dict())
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=s, line=dict(width=3, color=seg_color(s, SEGMENTS)),
                                         hovertemplate="Day %{x}: %{y:.0%}<extra>" + s + "</extra>"))
            fig.add_hline(y=thr, line_dash="dot", line_color=RISK)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.04])
            fig.update_xaxes(title="Days since last order")
            show(chart_layout(fig, 300), "cold")
            st.caption("Reorder chance for each segment's typical diner as their silence grows.")

        st.markdown("#### The diner map")
        rng = np.random.default_rng(0)
        fig = go.Figure()
        for s in SEGMENTS:
            sd = diners[diners.segment == s]
            fig.add_trace(go.Scatter(x=sd.orders_per_month + rng.uniform(-.25, .25, len(sd)),
                                     y=sd.avg_order_value, mode="markers", name=s,
                                     marker=dict(color=seg_color(s, SEGMENTS), size=8, opacity=.65),
                                     customdata=np.c_[sd.diner_id, sd.p_reorder.round(2)],
                                     hovertemplate="%{customdata[0]}<br>%{y:.0f} AED per order<br>Reorder chance %{customdata[1]}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[profile["orders_per_month"]], y=[profile["avg_order_value"]], mode="markers",
                                 name="Diner in sidebar", marker=dict(symbol="star", size=24, color=INK, line=dict(color="#fff", width=2))))
        fig.update_xaxes(title="Orders per month")
        fig.update_yaxes(title="Average order value (AED)")
        show(chart_layout(fig, 380), "map")
        st.caption("Each dot is a diner. The segments separate cleanly on order frequency and basket value alone.")

# ================= TAB 3: BASKETS =================
with tab_basket:
    if orders is None:
        st.warning("Add orders.csv next to app.py to see basket insights.")
    else:
        f1, f2 = st.columns([2, 1])
        scope = f1.radio("Show baskets for", ["All diners"] + SEGMENTS, horizontal=True)
        min_conf = f2.slider("Minimum confidence", 0.3, 0.95, 0.5, 0.05)
        n, rules, bundles, items = basket_stats(orders, scope)

        st.markdown(f"#### Bundles worth promoting ({n:,} orders)")
        if len(bundles):
            bc = st.columns(3, gap="medium")
            for c, (_, b) in zip(bc, bundles.head(3).iterrows()):
                c.markdown(f"""
<div class="qb-card qb-body">
  <h4>{' + '.join(item_name(i) for i in b['items'])}</h4>
  <div class="sub">When any one of these is ordered, the whole set comes with it {b['together']:.0%} of the time.</div>
  <div class="qb-row"><span>Share of all orders</span><span>{b['support']:.0%}</span></div>
  {'' if scope == 'All diners' else f'<div class="qb-row"><span>Lift</span><span>{b["lift"]:.1f}×</span></div>'}
</div>""", unsafe_allow_html=True)
        else:
            st.info("No bundle clears the support threshold for this group.")

        notes = {
            "Budget Students": "<b>Students already buy a fixed combo.</b> Cola is in nearly every student order, so cola rules have a lift close to 1 and say nothing new. Wrap and fries is the only real cross-sell, which makes a wrap meal deal the lever here.",
            "All diners": "<b>Read overall lifts with care.</b> Items like sushi or biryani are bought by only one segment, which inflates lift across the whole base. Pick a segment to see what really drives each group.",
            "Families": "<b>Families order meals, not items.</b> Pizza nights (pizza, garlic bread, dessert, cola) and biryani nights (biryani, raita, salad) are two distinct occasions. Promote them as two named family bundles.",
            "Premium Foodies": "<b>Foodies order cuisines.</b> A Japanese set (sushi, miso soup, green tea) and a steak dinner (steak, salad, wine) dominate. Sell them as chef's sets at full price; this segment barely uses discounts.",
        }
        if scope in notes:
            st.markdown(f'<div class="qb-insight qb-body" style="margin-top:14px">{notes[scope]}</div>', unsafe_allow_html=True)

        a, b = st.columns([1.2, 1], gap="large")
        with a:
            st.markdown("#### If they order X, they also order Y")
            r = rules[(rules.confidence >= min_conf) & (rules.lift > 1.05) & (rules.support >= 0.02)].sort_values("lift", ascending=False)
            r = r.assign(x=r.antecedent.map(item_name), y=r.consequent.map(item_name), confidence=r.confidence * 100)
            st.dataframe(r[["x", "y", "confidence", "lift", "support"]].head(25),
                         hide_index=True, width="stretch", height=390,
                         column_config={
                             "x": "If they order", "y": "They also order",
                             "confidence": st.column_config.ProgressColumn("Confidence", format="%.0f%%", min_value=0, max_value=100),
                             "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
                             "support": st.column_config.NumberColumn("Support", format="%.3f")})
            st.caption("Confidence is the share of X orders that also include Y. Lift above 1 means they're ordered together more than chance.")
        with b:
            st.markdown("#### Pairing strength (lift)")
            mat = pd.DataFrame(np.nan, index=items, columns=items)
            for _, rr in rules.iterrows():
                if rr.support >= 0.01:
                    mat.loc[rr.antecedent, rr.consequent] = rr.lift
            labels = [i.replace("_", " ") for i in items]
            fig = go.Figure(go.Heatmap(z=mat.values, x=labels, y=labels,
                                       colorscale=[[0, "#FFFFFF"], [0.15, "#F6D9A8"], [1, "#B5341F"]],
                                       hovertemplate="%{y} + %{x}<br>Lift %{z:.2f}<extra></extra>",
                                       colorbar=dict(thickness=10, outlinewidth=0)))
            fig.update_xaxes(tickangle=-45, showgrid=False)
            fig.update_yaxes(autorange="reversed", showgrid=False)
            show(chart_layout(fig, 430, legend=False), "heat")

# ================= TAB 4: CHURN WATCHLIST =================
with tab_watch:
    if diners is None:
        st.warning("Add diners.csv next to app.py to score your diner base.")
    else:
        f1, f2 = st.columns([2, 1])
        pick = f1.multiselect("Segments", SEGMENTS, default=SEGMENTS)
        top_n = f2.slider("Rows to show", 10, 100, 25, 5)
        w = diners[(diners.p_reorder < thr) & (diners.segment.isin(pick))].sort_values("value_at_risk", ascending=False)

        k1, k2, k3 = st.columns(3)
        k1.metric("Diners to contact", f"{len(w)}")
        k2.metric("Monthly value at risk", f"AED {w.value_at_risk.sum():,.0f}")
        share = w.value_at_risk.head(top_n).sum() / w.value_at_risk.sum() if len(w) else 0
        k3.metric(f"Value covered by the top {top_n}", f"{share:.0%}")

        if len(w):
            w = w.assign(p_pct=w.p_reorder * 100, action=[next_best_action(s, pp, thr, None)[1] for s, pp in zip(w.segment, w.p_reorder)],
                         offer=[bundle_text(s) or "" for s in w.segment])
            cols_ = ["diner_id", "segment", "p_pct", "days_since_last_order", "monthly_value", "value_at_risk", "action", "offer"]
            st.dataframe(w[cols_].head(top_n), hide_index=True, width="stretch", height=430,
                         column_config={
                             "diner_id": "Diner", "segment": "Segment",
                             "p_pct": st.column_config.ProgressColumn("Reorder chance", format="%.0f%%", min_value=0, max_value=100),
                             "days_since_last_order": st.column_config.NumberColumn("Days quiet"),
                             "monthly_value": st.column_config.NumberColumn("Monthly value", format="AED %.0f"),
                             "value_at_risk": st.column_config.NumberColumn("Value at risk", format="AED %.0f"),
                             "action": "Next step", "offer": "Bundle to offer"})
            st.download_button("Download watchlist (CSV)", w[cols_].round(1).rename(columns={"p_pct": "reorder_chance_pct", "action": "next_step", "offer": "bundle_to_offer"}).to_csv(index=False).encode(),
                               file_name="quickbite_churn_watchlist.csv", mime="text/csv")
            st.caption("Ranked by value at risk = (1 − reorder chance) × orders per month × average order value. "
                       "Start at the top: these diners cost QuickBite the most if they leave.")
        else:
            st.success("No diners in the selected segments fall below the at-risk threshold.")

# ================= TAB: INSIGHTS =================
with tab_insight:
    if diners is None or "reordered" not in diners:
        st.warning("Add diners.csv (with the reordered column) next to app.py to see these insights.")
    else:
        d = diners
        top20 = d.monthly_value.sort_values(ascending=False).head(int(len(d) * .2)).sum() / d.monthly_value.sum()
        quiet10 = (d.days_since_last_order > 10).mean()
        k1, k2, k3 = st.columns(3)
        k1.metric("Value held by the top 20% of diners", f"{top20:.0%}",
                  help="Order value is spread across the base, so broad campaigns matter more than a VIP list.")
        k2.metric("Diners quiet for more than 10 days", f"{quiet10:.0%}")
        k3.metric("Items per order", f"{orders.groupby('order_id').size().mean():.1f}" if orders is not None else "—")
        st.caption("Each chart below uses the observed reorder outcome in diners.csv, not a model prediction.")

        a, b = st.columns(2, gap="large")

        # 1. The recency cliff, straight from the data
        with a:
            bins, labs = [-1, 5, 10, 15, 20, 30, 1000], ["0–5", "6–10", "11–15", "16–20", "21–30", "31+"]
            rc = d.groupby(pd.cut(d.days_since_last_order, bins, labels=labs), observed=True).reordered.agg(["mean", "size"])
            drops = rc["mean"].diff().dropna()
            steep = drops.idxmin()
            prev = rc.index[list(rc.index).index(steep) - 1]
            late = d[d.days_since_last_order > 20].reordered
            late_txt = "no diner came back" if len(late) and late.mean() == 0 else f"only {late.mean():.0%} came back"
            st.markdown(f'<div class="qb-finding">After 20 quiet days, {late_txt}</div>'
                        f'<div class="qb-finding-sub">{rc["mean"].iloc[0]:.0%} of diners seen in the last 5 days reordered. '
                        f'The biggest drop is from {prev} to {steep} days ({rc.loc[prev, "mean"]:.0%} to {rc.loc[steep, "mean"]:.0%}), '
                        f'so that is when a reminder pays off most.</div>', unsafe_allow_html=True)
            fig = go.Figure(go.Bar(x=list(rc.index), y=rc["mean"],
                                   marker=dict(color=[prob_color(v, thr) for v in rc["mean"]], cornerradius=4),
                                   text=[f"{v:.0%}" for v in rc["mean"]], textposition="outside", cliponaxis=False,
                                   customdata=rc["size"], hovertemplate="%{x} days: %{y:.0%} reordered<br>%{customdata} diners<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Reordered")
            fig.update_xaxes(title="Days since last order")
            show(chart_layout(fig, 300, legend=False), "ins_recency")

        # 2. Do discounts buy loyalty?
        with b:
            dbins, dlabs = [0, .2, .4, .6, 1.0001], ["0–20%", "20–40%", "40–60%", "60–100%"]
            t = (d.assign(band=pd.cut(d.discount_rate, dbins, labels=dlabs, include_lowest=True))
                  .groupby(["segment", "band"], observed=True).reordered.agg(["mean", "size"]).reset_index())
            t = t[t["size"] >= 10]
            spread = t.groupby("segment")["mean"].agg(lambda s: s.iloc[-1] - s.iloc[0] if len(s) > 1 else np.nan)
            mover = spread.idxmax() if spread.notna().any() else None
            if mover is not None:
                mt = t[t.segment == mover]
                loyal = d.groupby("segment").reordered.mean().idxmax()
                single = [s for s in SEGMENTS if (t.segment == s).sum() == 1]
                extra = f" {single[0]} almost never use discounts, so there is no evidence either way for them." if single else ""
                st.markdown(f'<div class="qb-finding">Discounts only seem to move {mover}</div>'
                            f'<div class="qb-finding-sub">{mover} who use discounts on {mt.band.iloc[-1]} of orders reordered '
                            f'{mt["mean"].iloc[-1]:.0%} of the time, against {mt["mean"].iloc[0]:.0%} at {mt.band.iloc[0]}. '
                            f'{loyal} reorder at every discount level, so offers to them buy little.{extra}</div>',
                            unsafe_allow_html=True)
            fig = go.Figure()
            for s in SEGMENTS:
                ts = t[t.segment == s]
                if len(ts):
                    fig.add_trace(go.Scatter(x=ts.band.astype(str), y=ts["mean"], name=s, mode="lines+markers+text",
                                             line=dict(width=2, color=seg_color(s, SEGMENTS)),
                                             marker=dict(size=10, line=dict(color="#fff", width=2)),
                                             text=[f"{v:.0%}" for v in ts["mean"]], textposition="top center",
                                             textfont=dict(color=INK), customdata=ts["size"],
                                             hovertemplate="%{x} discount use: %{y:.0%} reordered<br>%{customdata} diners<extra>" + s + "</extra>"))
            fig.update_xaxes(categoryorder="array", categoryarray=dlabs, title="Share of orders using a discount")
            fig.update_yaxes(tickformat=".0%", range=[-0.02, 1.12], title="Reordered")
            show(chart_layout(fig, 300), "ins_discount")
        st.caption("Groups with fewer than 10 diners are hidden. These are patterns in past behaviour, not a tested offer; "
                   "an A/B test on a discount would confirm cause and effect.")

        c, e = st.columns(2, gap="large")

        # 3. Value × risk: who to rescue first
        with c:
            hi = d.monthly_value >= d.monthly_value.median()
            rk = d.p_reorder < thr
            quad = np.select([hi & rk, hi & ~rk, ~hi & rk], ["Rescue first", "Protect", "Cheap nudge"], "Leave be")
            res = quad == "Rescue first"
            share = d.value_at_risk[res].sum() / d.value_at_risk.sum()
            st.markdown(f'<div class="qb-finding">{res.sum()} diners hold {share:.0%} of the value at risk</div>'
                        f'<div class="qb-finding-sub">They spend more than the median diner each month and are below the '
                        f'{thr:.0%} at-risk line. Start every win-back campaign with this group.</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for s in SEGMENTS:
                sd = d[d.segment == s]
                fig.add_trace(go.Scatter(x=sd.p_reorder, y=sd.monthly_value, mode="markers", name=s,
                                         marker=dict(size=8, opacity=.7, color=seg_color(s, SEGMENTS), line=dict(color="#fff", width=1)),
                                         customdata=np.c_[sd.diner_id, quad[d.segment == s]],
                                         hovertemplate="%{customdata[0]}: %{customdata[1]}<br>Reorder chance %{x:.0%}<br>AED %{y:,.0f} a month<extra></extra>"))
            ymax = d.monthly_value.max() * 1.08
            fig.add_vline(x=thr, line_dash="dot", line_color=RISK)
            fig.add_hline(y=d.monthly_value.median(), line_dash="dot", line_color=MUTED)
            for name, x, y, anchor in [("Rescue first", 0.01, ymax, "left"), ("Protect", 0.99, ymax, "right"),
                                       ("Cheap nudge", 0.01, 0, "left"), ("Leave be", 0.99, 0, "right")]:
                n_ = (quad == name).sum()
                fig.add_annotation(x=x, y=y, text=f"<b>{name}</b> · {n_}", showarrow=False, xanchor=anchor,
                                   yanchor="top" if y else "bottom", font=dict(color=RISK if name == "Rescue first" else MUTED, size=12))
            fig.update_xaxes(tickformat=".0%", range=[-0.02, 1.02], title="Reorder chance (model)")
            fig.update_yaxes(title="Monthly order value (AED)", range=[0, ymax])
            show(chart_layout(fig, 340), "ins_quadrant")

        # 4. Age: loyalty and basket value move in opposite directions (two panels, one scale each)
        with e:
            abins, alabs = [17, 24, 34, 44, 200], ["18–24", "25–34", "35–44", "45+"]
            ag = d.groupby(pd.cut(d.age, abins, labels=alabs), observed=True).agg(
                rate=("reordered", "mean"), aov=("avg_order_value", "mean"), n=("diner_id", "size"))
            young_students = (d[d.age <= 24].segment == "Budget Students").mean()
            st.markdown(f'<div class="qb-finding">Loyalty falls with age while baskets grow {ag.aov.iloc[-1]/ag.aov.iloc[0]:.0f}×</div>'
                        f'<div class="qb-finding-sub">{ag.rate.iloc[0]:.0%} of under-25s reorder against {ag.rate.iloc[-1]:.0%} of over-44s, '
                        f'but older diners spend far more per order. Age largely tracks segment: {young_students:.0%} of under-25s are Budget Students.</div>',
                        unsafe_allow_html=True)
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Reordered", "Average order value"), horizontal_spacing=.12)
            fig.add_trace(go.Bar(x=list(ag.index), y=ag.rate, marker=dict(color=INK, cornerradius=4),
                                 text=[f"{v:.0%}" for v in ag.rate], textposition="outside", cliponaxis=False, customdata=ag.n,
                                 hovertemplate="Age %{x}: %{y:.0%} reordered<br>%{customdata} diners<extra></extra>"), 1, 1)
            fig.add_trace(go.Bar(x=list(ag.index), y=ag.aov, marker=dict(color=MUTED, cornerradius=4),
                                 text=[f"{v:.0f}" for v in ag.aov], textposition="outside", cliponaxis=False,
                                 hovertemplate="Age %{x}: AED %{y:.0f} per order<extra></extra>"), 1, 2)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.15], row=1, col=1)
            fig.update_yaxes(tickprefix="AED ", range=[0, ag.aov.max() * 1.2], row=1, col=2)
            fig.update_annotations(font=dict(size=13, color=MUTED), yshift=8)
            chart_layout(fig, 340, legend=False).update_layout(margin=dict(t=36))
            show(fig, "ins_age")
        st.caption("The 45+ group is small, so treat its numbers as indicative.")

# ================= TAB: CAMPAIGN PLANNER =================
with tab_plan:
    if diners is None:
        st.warning("Add diners.csv next to app.py to plan a campaign.")
    else:
        st.markdown('<div class="qb-insight qb-body"><b>Plan a win-back campaign.</b> Pick who to target and what you '
                    'expect an offer to do. The lift is your assumption: the data shows who is at risk, not how they '
                    'respond to an offer, so try a low and a high value.</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
        targets = c1.multiselect("Segments to target", SEGMENTS, default=SEGMENTS, key="plan_segs")
        uplift = c2.slider("Lift in reorder chance", 5, 50, 10, 5, format="%d pts",
                           help="How many percentage points an offer adds to a diner's reorder chance.") / 100
        cost = c3.number_input("Cost per offer (AED)", 0, 300, 50, 5,
                               help="Discount given plus messaging cost, per diner contacted.")
        months = c4.select_slider("Value counted over", [1, 3, 6, 12], value=1, format_func=lambda m: f"{m} month{'s' * (m > 1)}")

        pool = diners[(diners.p_reorder < thr) & diners.segment.isin(targets)].copy()
        pool["gain"] = np.minimum(uplift, 1 - pool.p_reorder) * pool.monthly_value * months
        pool["net"] = pool.gain - cost
        pool = pool.sort_values("net", ascending=False)
        worth = pool[pool.net > 0]

        if not len(pool):
            st.success("No at-risk diners in the selected segments.")
        else:
            rec, spend = worth.gain.sum(), len(worth) * cost
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("At-risk diners", f"{len(pool)}")
            k2.metric("Worth contacting", f"{len(worth)}", f"{len(worth)/len(pool):.0%} of them", delta_arrow="off", delta_color="off")
            k3.metric("Expected value recovered", f"AED {rec:,.0f}")
            k4.metric("Net return", f"AED {rec - spend:,.0f}",
                      f"{rec/spend:.1f}× the AED {spend:,.0f} spent" if spend else "no spend", delta_arrow="off", delta_color="off")

            a, b = st.columns([1.1, 1], gap="large")
            with a:
                st.markdown("#### How far down the list to go")
                cum = pool.net.cumsum().values
                x = np.arange(1, len(pool) + 1)
                best = int(np.argmax(cum))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=cum, mode="lines", line=dict(color=INK, width=2),
                                         hovertemplate="Contact the top %{x}: net AED %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[best + 1], y=[cum[best]], mode="markers+text", marker=dict(size=12, color=SAFE, line=dict(color="#fff", width=2)),
                                         text=[f"Best: top {best + 1}"], textposition="top center", textfont=dict(color=INK),
                                         hovertemplate="Best stopping point: top %{x}<br>Net AED %{y:,.0f}<extra></extra>"))
                fig.add_trace(go.Scatter(x=[len(pool)], y=[cum[-1]], mode="markers+text", marker=dict(size=10, color=MUTED, line=dict(color="#fff", width=2)),
                                         text=["Everyone"], textposition="bottom left", textfont=dict(color=INK),
                                         hovertemplate="Contact all %{x}: net AED %{y:,.0f}<extra></extra>"))
                fig.add_hline(y=0, line_color=LINE, line_width=2)
                fig.update_xaxes(title="Diners contacted, best prospects first")
                fig.update_yaxes(title="Cumulative net return (AED)", tickformat=",.0f", range=[min(0, cum.min()) * 1.1, max(cum.max(), 1) * 1.18])
                show(chart_layout(fig, 320, legend=False), "plan_curve")
                lost = cum[best] - cum[-1]
                st.caption(f"Contacting everyone at risk would return AED {lost:,.0f} less than stopping at the top {best + 1}. "
                           "Past that point an offer costs more than the value it is likely to save."
                           if lost > 1 else "With these settings every at-risk diner is worth contacting.")
            with b:
                st.markdown("#### By segment")
                g = pool.groupby("segment")
                tbl = pd.DataFrame({
                    "At risk": g.size(),
                    "Contact": g.net.apply(lambda s: (s > 0).sum()),
                    "Recovered": g.apply(lambda x: x.gain[x.net > 0].sum(), include_groups=False),
                    "Cost": g.net.apply(lambda s: (s > 0).sum() * cost),
                }).reindex([s for s in SEGMENTS if s in targets]).dropna()
                tbl["Net"] = tbl["Recovered"] - tbl["Cost"]
                st.dataframe(tbl.reset_index().rename(columns={"segment": "Segment", "index": "Segment"}), hide_index=True, width="stretch",
                             column_config={c_: st.column_config.NumberColumn(format="%,.0f") for c_ in ["Recovered", "Cost", "Net"]})
                missing = [s for s in targets if s not in tbl.index]
                st.caption("All money in AED. Recovered value = extra reorder chance × monthly order value × months counted, "
                           "capped so no diner goes above 100%."
                           + (f" {', '.join(missing)} have no diners below the at-risk line." if missing else ""))
            st.download_button("Download contact list (CSV)",
                               worth[["diner_id", "segment", "p_reorder", "monthly_value", "gain", "net"]].round(2)
                               .rename(columns={"p_reorder": "reorder_chance", "gain": "expected_recovered_aed", "net": "expected_net_aed"})
                               .to_csv(index=False).encode(), file_name="quickbite_campaign_contacts.csv", mime="text/csv")

# ================= TAB: MODEL CHECK =================
with tab_model:
    oof = cv_scores(diners) if diners is not None else None
    if oof is None:
        st.warning("Add diners.csv (with the reordered column) next to app.py to check the model.")
    else:
        y = diners.reordered.values
        pred = (oof >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()   # 0 = churned, 1 = reordered
        churners, loyal = tn + fp, fn + tp
        st.caption("Every diner is scored by a model trained without them (5-fold cross-validation), so these are honest "
                   "out-of-sample numbers. They move with the at-risk threshold in the sidebar.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Accuracy", f"{(tn + tp) / len(y):.1%}", f"vs {max(y.mean(), 1 - y.mean()):.0%} by always guessing", delta_arrow="off", delta_color="off")
        k2.metric("Churners caught", f"{tn} of {churners}", f"{tn / churners:.0%}", delta_arrow="off", delta_color="off")
        k3.metric("Loyal diners flagged by mistake", f"{fn} of {loyal}", f"{fn / loyal:.0%}", delta_arrow="off", delta_color="off")
        k4.metric("Ranking quality (AUC)", f"{roc_auc_score(y, oof):.2f}", "1.00 is perfect, 0.50 is a coin flip", delta_arrow="off", delta_color="off")

        a, b = st.columns([1, 1.25], gap="large")
        with a:
            st.markdown(f"#### Results at the {thr:.0%} line")
            cell = lambda n, lab, good: (f'<div style="background:{"rgba(15,157,122,.12)" if good else "rgba(215,38,61,.10)"};'
                                         f'border-radius:10px;padding:14px 12px;text-align:center">'
                                         f'<div style="font-family:Bricolage Grotesque,sans-serif;font-size:2rem;font-weight:800;color:{INK}">{n}</div>'
                                         f'<div style="color:{MUTED};font-size:.88rem">{lab}</div></div>')
            st.markdown(f"""
<div class="qb-card qb-body">
  <div style="display:grid;grid-template-columns:auto 1fr 1fr;gap:8px;align-items:center">
    <div></div><div class="qb-k" style="text-align:center">Flagged at risk</div><div class="qb-k" style="text-align:center">Predicted to reorder</div>
    <div class="qb-k">Actually churned</div>{cell(tn, "✓ caught", True)}{cell(fp, "✗ missed churner", False)}
    <div class="qb-k">Actually reordered</div>{cell(fn, "✗ false alarm", False)}{cell(tp, "✓ correctly left alone", True)}
  </div>
</div>""", unsafe_allow_html=True)
            st.caption("A missed churner costs a customer. A false alarm costs one voucher, which is why the default line sits at 60%, not 50%.")
        with b:
            st.markdown("#### What moving the line trades off")
            ts = np.round(np.arange(0.2, 0.91, 0.05), 2)
            caught = [((oof < t) & (y == 0)).sum() / churners for t in ts]
            flagged = [((oof < t) & (y == 1)).sum() / loyal for t in ts]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=caught, name="Churners caught", mode="lines+markers",
                                     line=dict(color=SAFE, width=2), marker=dict(size=8, line=dict(color="#fff", width=2)),
                                     hovertemplate="Line at %{x:.0%}: %{y:.0%} of churners caught<extra></extra>"))
            fig.add_trace(go.Scatter(x=ts, y=flagged, name="Loyal diners flagged by mistake", mode="lines+markers",
                                     line=dict(color=RISK, width=2, dash="dash"), marker=dict(size=8, symbol="square", line=dict(color="#fff", width=2)),
                                     hovertemplate="Line at %{x:.0%}: %{y:.0%} of loyal diners flagged<extra></extra>"))
            fig.add_vline(x=thr, line_dash="dot", line_color=INK, annotation_text="Current line", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", title="At-risk threshold")
            fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
            chart_layout(fig, 300).update_layout(margin=dict(t=34))
            show(fig, "model_tradeoff")

        c, e = st.columns(2, gap="large")
        with c:
            st.markdown("#### Can the percentages be trusted?")
            bins = pd.cut(oof, [0, .2, .4, .6, .8, 1.0001], labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"], include_lowest=True)
            cal = pd.DataFrame({"band": bins, "pred": oof, "y": y}).groupby("band", observed=True).agg(pred=("pred", "mean"), y=("y", "mean"), n=("y", "size"))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, width=1.5, dash="dot"),
                                     name="Perfectly honest", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=cal.pred, y=cal.y, mode="lines+markers+text", name="QuickBite model",
                                     line=dict(color=INK, width=2), marker=dict(size=10, color=INK, line=dict(color="#fff", width=2)),
                                     text=[f"{n} diners" for n in cal.n], textposition=["bottom right"] * (len(cal) - 1) + ["bottom left"], textfont=dict(color=MUTED, size=11),
                                     hovertemplate="Model said %{x:.0%} on average<br>%{y:.0%} actually reordered<extra></extra>"))
            fig.update_xaxes(tickformat=".0%", range=[-0.02, 1.02], title="Predicted reorder chance")
            fig.update_yaxes(tickformat=".0%", range=[-0.02, 1.05], title="Actually reordered")
            show(chart_layout(fig, 300), "model_calibration")
            gap = (cal.pred - cal.y).abs().max()
            st.caption(f"Each point groups diners by their predicted chance. Points near the dotted line mean the percentages "
                       f"can be read literally; the largest gap here is {gap * 100:.0f} percentage points.")
        with e:
            st.markdown("#### What the model leans on overall")
            try:
                coef = pd.Series(clf[-1].coef_[0], index=CLF_FEATURES).sort_values(key=np.abs)
                fig = go.Figure(go.Bar(x=coef.values, y=[LABELS[i] for i in coef.index], orientation="h",
                                       marker=dict(color=[SAFE if v > 0 else RISK for v in coef.values], cornerradius=4),
                                       text=[f"{v:+.2f}" for v in coef.values], textposition="outside", cliponaxis=False,
                                       textfont=dict(color=INK),
                                       hovertemplate="%{y}: %{x:+.2f} per standard deviation<extra></extra>"))
                lim = coef.abs().max() * 1.35
                fig.update_xaxes(range=[-lim, lim], title="← toward churn  |  toward reorder →", zeroline=True, zerolinewidth=2, zerolinecolor=INK)
                show(chart_layout(fig, 300, legend=False), "model_coef")
                st.caption("Effect of a one-standard-deviation increase in each input, holding the others fixed.")
            except Exception:
                st.info("Overall weights are shown when the saved classifier is a scaled linear model.")
