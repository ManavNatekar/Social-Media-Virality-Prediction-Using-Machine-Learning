"""Streamlit dashboard for predicting social media post virality."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import FEATURE_COLUMNS, MODEL_PATH, train_and_save_model

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"


def compute_ratios(views: float, likes: float, comments: float, shares: float) -> dict:
    """Calculate safe ratios and engagement metrics. This avoids division-by-zero."""
    views_value = max(float(views), 0.0)
    likes_value = max(float(likes), 0.0)
    comments_value = max(float(comments), 0.0)
    shares_value = max(float(shares), 0.0)

    if views_value > 0:
        like_view_ratio = likes_value / views_value
        comment_view_ratio = comments_value / views_value
        share_view_ratio = shares_value / views_value
        engagement_rate = ((likes_value + comments_value + shares_value) / views_value) * 100
    else:
        like_view_ratio = 0.0
        comment_view_ratio = 0.0
        share_view_ratio = 0.0
        engagement_rate = 0.0

    return {
        "views": views_value,
        "likes": likes_value,
        "comments": comments_value,
        "shares": shares_value,
        "like_view_ratio": like_view_ratio,
        "comment_view_ratio": comment_view_ratio,
        "share_view_ratio": share_view_ratio,
        "engagement_rate": engagement_rate,
    }


def determine_category(score: float) -> str:
    """Map the prediction score to a virality category."""
    if 0 <= score <= 20:
        return "Low"
    if 21 <= score <= 40:
        return "Moderate"
    if 41 <= score <= 60:
        return "Good"
    if 61 <= score <= 80:
        return "High"
    return "Viral"


def generate_explanation(score: float, engagement_rate: float, like_ratio: float, share_ratio: float) -> str:
    """Generate a human-readable explanation for the virality score."""
    category = determine_category(score)

    if score >= 80:
        return (
            "This post shows a very strong engagement pattern. The combination of high viewer interest, "
            "strong likes-to-view ratio, and meaningful sharing suggests a very high chance of viral reach."
        )
    if score >= 60:
        return (
            "This post has a healthy level of interaction. The engagement rate and share activity indicate "
            "strong audience interest and above-average discoverability."
        )
    if score >= 40:
        return (
            "This post is performing reasonably well. It has a good mix of interaction and reach, but stronger "
            "engagement or additional shares would improve its viral potential."
        )
    if score >= 20:
        return (
            "This post is showing moderate traction. It is generating some discussion and audience response, "
            "but the engagement pattern is still below the typical viral threshold."
        )
    if engagement_rate == 0:
        return (
            "No meaningful engagement was detected because the post received zero views. "
            "With more visibility, the content could gain stronger traction."
        )
    return (
        "This post is currently underperforming relative to its potential reach. The engagement pattern suggests "
        "limited visibility or content resonance, so more compelling incentivization may be needed."
    )


def load_model():
    """Load a trained model from disk, creating one if it does not exist."""
    if not MODEL_PATH.exists():
        st.warning("Model not found. Training a new model using the sample dataset...")
        train_and_save_model()

    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as exc:  # pragma: no cover - defensive UI code
        st.error(f"Unable to load the model: {exc}")
        return None


def predict_virality(model, views: float, likes: float, comments: float, shares: float) -> tuple[float, dict, str]:
    """Run model prediction and return the final score and supporting metrics."""
    metrics = compute_ratios(views, likes, comments, shares)
    feature_row = pd.DataFrame([metrics])[FEATURE_COLUMNS]
    predicted_score = float(model.predict(feature_row)[0])
    virality_percentage = float(np.clip(predicted_score, 0.0, 100.0))
    category = determine_category(virality_percentage)
    explanation = generate_explanation(
        virality_percentage,
        metrics["engagement_rate"],
        metrics["like_view_ratio"],
        metrics["share_view_ratio"],
    )
    return virality_percentage, metrics, explanation


st.set_page_config(page_title="Social Media Virality Predictor", page_icon="🚀", layout="wide")

st.title("🚀 Social Media Virality Predictor")
st.caption(
    "A practical machine learning demo for estimating how likely a social media post is to go viral. "
    "The sample dataset is for demonstration only and should be replaced by real historical post data."
)

with st.form("virality_form"):
    st.subheader("Enter post metrics")
    col1, col2 = st.columns(2)

    with col1:
        views = st.number_input("Views", min_value=0, step=1, value=50000)
        likes = st.number_input("Likes", min_value=0, step=1, value=4200)

    with col2:
        comments = st.number_input("Comments", min_value=0, step=1, value=600)
        shares = st.number_input("Shares", min_value=0, step=1, value=1200)

    submitted = st.form_submit_button("🔮 Predict Virality", use_container_width=True)

if submitted:
    model = load_model()

    if model is not None:
        score, metrics, explanation = predict_virality(model, views, likes, comments, shares)
        category = determine_category(score)

        st.markdown("---")
        st.subheader("🔥 Virality Score")
        st.markdown(f"<h2 style='margin:0; text-align:center; color:#ff6b6b;'>{score:.1f}%</h2>", unsafe_allow_html=True)
        st.write(f"Category: **{category}**")
        st.write(f"Engagement Rate: **{metrics['engagement_rate']:.2f}%**")
        st.progress(min(100, max(0, score)) / 100)

        st.markdown("### Prediction")
        st.info(explanation)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Views", f"{int(metrics['views']):,}")
        metric_cols[1].metric("Likes", f"{int(metrics['likes']):,}")
        metric_cols[2].metric("Comments", f"{int(metrics['comments']):,}")
        metric_cols[3].metric("Shares", f"{int(metrics['shares']):,}")

        st.markdown("### Engagement Breakdown")
        breakdown_df = pd.DataFrame(
            {
                "Metric": ["Likes", "Comments", "Shares"],
                "Value": [metrics["likes"], metrics["comments"], metrics["shares"]],
            }
        )
        st.bar_chart(breakdown_df.set_index("Metric"), use_container_width=True)

        st.markdown("### Ratio Summary")
        ratio_df = pd.DataFrame(
            {
                "Ratio Type": ["Like/View", "Comment/View", "Share/View"],
                "Value": [
                    metrics["like_view_ratio"],
                    metrics["comment_view_ratio"],
                    metrics["share_view_ratio"],
                ],
            }
        )
        st.dataframe(ratio_df, use_container_width=True)

        st.caption(
            "Note: This is a demonstration model trained on synthetic post data. Replace the sample dataset "
            "with historical social media data for production-quality predictions."
        )

    else:
        st.error("The model could not be loaded. Please run the training script first.")
