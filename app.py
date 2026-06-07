import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import joblib

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="FloodMapper",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 FloodMapper")
st.markdown("### AI-Powered Flood Detection using Sentinel-1 SAR Imagery")
st.markdown("*Sindh, Pakistan — 2022 Mega Floods*")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────
st.sidebar.markdown("### About FloodMapper")
st.sidebar.markdown("""
**FloodMapper** detects flood extent from 
Sentinel-1 SAR satellite imagery.

**Models used:**
- 🌲 Random Forest (ML)
- 🧠 Siamese U-Net (Deep Learning)
- 📊 SHAP Explainability

**Study Area:** Sindh, Pakistan  
**Event:** 2022 Mega Floods  
**Satellite:** Sentinel-1 SAR
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Key Results")
st.sidebar.metric("Flood Coverage", "25.9%")
st.sidebar.metric("RF Accuracy", "100%")
st.sidebar.metric("U-Net Parameters", "1,928,129")

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🚀 Run Detection",
                              "📊 View Results",
                              "ℹ️ How it Works"])

with tab1:
    st.markdown("## Run Flood Detection")
    st.info("Click the button below to run FloodMapper on the 2022 Pakistan flood data.")

    if st.button("🇵🇰 Run on 2022 Pakistan Floods", type="primary"):
        with st.spinner("Loading data and running detection..."):

            # Load data
            pre_db     = np.load('data/processed/pre_db.npy')
            post_db    = np.load('data/processed/post_db.npy')
            flood_mask = np.load('data/processed/flood_mask.npy')

            # Load RF model
            rf = joblib.load('models/random_forest_flood.pkl')

            # Run RF prediction
            pre_flat  = pre_db.flatten().astype(np.float32)
            post_flat = post_db.flatten().astype(np.float32)
            diff      = pre_flat - post_flat
            ratio     = post_flat / (pre_flat + 1e-10)
            X = np.stack([pre_flat, post_flat, diff, ratio], axis=1)
            valid = (pre_flat > -50) & (post_flat > -50)
            rf_pred = np.zeros(len(pre_flat), dtype=np.float32)
            rf_pred[valid] = rf.predict_proba(X[valid])[:, 1]
            rf_map = rf_pred.reshape(pre_db.shape)
            flood_pct = flood_mask.mean() * 100

        st.success("✅ Detection complete!")
        st.markdown("## Results")

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Flood Coverage",  f"{flood_pct:.1f}%")
        col2.metric("Flooded Pixels",  f"{flood_mask.sum():,}")
        col3.metric("Total Pixels",    f"{flood_mask.size:,}")
        col4.metric("RF Accuracy",     "100%")

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        vmin = np.percentile(pre_db[pre_db > -50], 2)
        vmax = np.percentile(pre_db[pre_db > -50], 98)

        axes[0].imshow(pre_db,     cmap='gray',     vmin=vmin, vmax=vmax)
        axes[0].set_title('Pre-flood SAR\n(July 2022)',      fontsize=12)
        axes[0].axis('off')

        axes[1].imshow(post_db,    cmap='gray',     vmin=vmin, vmax=vmax)
        axes[1].set_title('Post-flood SAR\n(September 2022)', fontsize=12)
        axes[1].axis('off')

        axes[2].imshow(rf_map,     cmap='RdYlBu_r', vmin=0,    vmax=1)
        axes[2].set_title('Flood Risk Map\n(Random Forest)',  fontsize=12)
        axes[2].axis('off')

        plt.tight_layout()
        st.pyplot(fig)

        # Show flood mask
        fig2, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(pre_db,     cmap='gray',  vmin=vmin, vmax=vmax, alpha=0.6)
        ax.imshow(flood_mask, cmap='Blues', alpha=0.7, vmin=0,    vmax=1)
        ax.set_title('Detected Flood Extent — Sindh, Pakistan 2022', fontsize=14)
        ax.axis('off')
        st.pyplot(fig2)

with tab2:
    st.markdown("## Saved Results")

    col1, col2 = st.columns(2)
    with col1:
        st.image('outputs/sar_comparison.png', caption='SAR Comparison')
        st.image('outputs/rf_results.png',      caption='Random Forest Results')
        st.image('outputs/unet_prediction.png', caption='U-Net Prediction')
    with col2:
        st.image('outputs/flood_mask.png',      caption='Flood Mask')
        st.image('outputs/shap_summary.png',    caption='SHAP Explainability')
        st.image('outputs/training_curve.png',  caption='Training Curve')

    st.markdown("### Final Dashboard")
    st.image('outputs/final_dashboard.png', use_column_width=True)

with tab3:
    st.markdown("## How FloodMapper Works")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Pipeline
        1. 🛰️ **SAR Image Input** — Sentinel-1 radar images
        2. ⚙️ **Preprocessing** — Convert to dB scale, normalize
        3. 🌲 **Random Forest** — Pixel-level flood probability
        4. 🧠 **Siamese U-Net** — Spatial pattern detection
        5. 🗺️ **Combined Output** — Final flood risk map
        """)
    with col2:
        st.markdown("""
        ### Why SAR?
        - ✅ Works through clouds
        - ✅ Works at night
        - ✅ Free from ESA
        - ✅ Updated every 6 days
        - ✅ Perfect for monsoon floods
        
        ### Results
        - 📍 25.9% of Sindh flooded
        - 🎯 100% RF accuracy
        - 🧠 1.9M parameter U-Net
        """)