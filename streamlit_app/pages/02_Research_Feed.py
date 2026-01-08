"""
WeatherFlow Research Feed

Stay up to date with the latest weather AI research.
Curated papers, model releases, and community updates.

Features:
- Latest papers from arXiv and top venues
- Model release announcements
- Community highlights
- Research opportunities
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(
    page_title="Research Feed - WeatherFlow",
    page_icon="📰",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .paper-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #1e88e5;
    }
    .paper-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #1e88e5;
        margin-bottom: 5px;
    }
    .paper-authors {
        color: #666;
        font-size: 0.9em;
        margin-bottom: 10px;
    }
    .paper-venue {
        background: #1e88e5;
        color: white;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.8em;
        display: inline-block;
    }
    .tag {
        display: inline-block;
        background: #e0e0e0;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        margin: 2px;
    }
    .model-release {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #28a745;
    }
    .community-highlight {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 15px 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 Weather AI Research Feed")

st.markdown("""
Stay informed about the latest developments in AI-powered weather prediction.
Curated content from top venues, arXiv, and the research community.
""")

# Filter options
col_filter1, col_filter2, col_filter3 = st.columns(3)

with col_filter1:
    category_filter = st.multiselect(
        "Categories",
        ["Foundation Models", "Graph Neural Networks", "Diffusion Models",
         "Physics-Informed", "Ensemble Methods", "Extreme Events", "Climate"],
        default=["Foundation Models", "Graph Neural Networks", "Diffusion Models"]
    )

with col_filter2:
    time_filter = st.selectbox(
        "Time Range",
        ["Last 7 days", "Last 30 days", "Last 3 months", "Last year", "All time"],
        index=1
    )

with col_filter3:
    venue_filter = st.multiselect(
        "Venues",
        ["arXiv", "Nature", "Science", "NeurIPS", "ICML", "ICLR", "AGU", "EGU"],
        default=["arXiv", "Nature", "Science", "NeurIPS", "ICML"]
    )

# Tabs for different content types
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄 Latest Papers",
    "🚀 Model Releases",
    "🌟 Community",
    "📚 Reading List",
    "🎓 Learning Resources"
])

# =================== TAB 1: Latest Papers ===================
with tab1:
    st.header("Recent Publications")

    # Curated paper database (in production, this would be fetched from an API)
    PAPERS = [
        {
            "title": "Aurora: A Foundation Model of the Atmosphere",
            "authors": "Bodnar et al.",
            "venue": "arXiv",
            "date": "2024-05-21",
            "abstract": "We introduce Aurora, a large-scale foundation model of the atmosphere trained on over a million hours of diverse weather and climate data...",
            "arxiv": "https://arxiv.org/abs/2405.13063",
            "tags": ["Foundation Models", "Transformers"],
            "citations": 45,
            "organization": "Microsoft Research",
        },
        {
            "title": "Neural General Circulation Models for Weather and Climate",
            "authors": "Kochkov et al.",
            "venue": "Nature",
            "date": "2024-07-22",
            "abstract": "We present NeuralGCM, a hybrid approach that combines the interpretability and physical guarantees of traditional GCMs with the data-driven power of neural networks...",
            "arxiv": "https://arxiv.org/abs/2311.07222",
            "doi": "10.1038/s41586-024-07744-y",
            "tags": ["Physics-Informed", "Hybrid Models", "Climate"],
            "citations": 89,
            "organization": "Google Research",
        },
        {
            "title": "GenCast: Diffusion-based ensemble forecasting for medium-range weather",
            "authors": "Price et al.",
            "venue": "arXiv",
            "date": "2023-12-24",
            "abstract": "We introduce GenCast, a probabilistic weather model using diffusion that generates diverse, skillful ensemble forecasts...",
            "arxiv": "https://arxiv.org/abs/2312.15796",
            "tags": ["Diffusion Models", "Ensemble Methods"],
            "citations": 156,
            "organization": "Google DeepMind",
        },
        {
            "title": "Learning skillful medium-range global weather forecasting",
            "authors": "Lam et al.",
            "venue": "Science",
            "date": "2023-11-14",
            "abstract": "We introduce GraphCast, a machine learning-based weather prediction method that outperforms the best operational deterministic systems...",
            "arxiv": "https://arxiv.org/abs/2212.12794",
            "doi": "10.1126/science.adi2336",
            "tags": ["Graph Neural Networks", "Global Forecasting"],
            "citations": 892,
            "organization": "Google DeepMind",
        },
        {
            "title": "Accurate medium-range global weather forecasting with 3D neural networks",
            "authors": "Bi et al.",
            "venue": "Nature",
            "date": "2023-07-05",
            "abstract": "We present Pangu-Weather, a 3D high-resolution model for deterministic global weather forecasting using Earth-specific transformers...",
            "arxiv": "https://arxiv.org/abs/2211.02556",
            "doi": "10.1038/s41586-023-06185-3",
            "tags": ["3D Transformers", "Global Forecasting"],
            "citations": 567,
            "organization": "Huawei Cloud",
        },
        {
            "title": "FourCastNet: A Global Data-driven High-resolution Weather Model using Adaptive Fourier Neural Operators",
            "authors": "Pathak et al.",
            "venue": "arXiv",
            "date": "2022-02-22",
            "abstract": "We present FourCastNet, a Fourier neural operator-based model that achieves state-of-the-art accuracy with 45,000x speedup...",
            "arxiv": "https://arxiv.org/abs/2202.11214",
            "tags": ["Fourier Methods", "Speed"],
            "citations": 423,
            "organization": "NVIDIA",
        },
        {
            "title": "ClimaX: A foundation model for weather and climate",
            "authors": "Nguyen et al.",
            "venue": "ICML",
            "date": "2023-07-03",
            "abstract": "We introduce ClimaX, the first foundation model for weather and climate that can be fine-tuned for various downstream tasks...",
            "arxiv": "https://arxiv.org/abs/2301.10343",
            "tags": ["Foundation Models", "Transfer Learning"],
            "citations": 234,
            "organization": "Microsoft Research",
        },
        {
            "title": "Stormer: Simple Scalable Spatio-temporal Transformer for Global Weather Forecasting",
            "authors": "Nguyen et al.",
            "venue": "arXiv",
            "date": "2024-03-14",
            "abstract": "We present Stormer, a simple yet effective transformer architecture that achieves competitive results with fewer parameters...",
            "arxiv": "https://arxiv.org/abs/2312.03876",
            "tags": ["Transformers", "Efficiency"],
            "citations": 28,
            "organization": "Microsoft Research",
        },
        {
            "title": "FuXi: A cascade machine learning forecasting system for 15-day global weather forecast",
            "authors": "Chen et al.",
            "venue": "Nature Climate",
            "date": "2023-08-21",
            "abstract": "FuXi is a cascade ML system that achieves reliable 15-day forecasts, extending the useful prediction range...",
            "arxiv": "https://arxiv.org/abs/2306.12873",
            "tags": ["Long-range Forecasting", "Cascade Models"],
            "citations": 112,
            "organization": "Fudan University",
        },
        {
            "title": "AtmoRep: A stochastic model of atmosphere dynamics using large scale representation learning",
            "authors": "Lessig et al.",
            "venue": "arXiv",
            "date": "2023-08-03",
            "abstract": "We present AtmoRep, a novel approach to atmospheric modeling using large-scale self-supervised representation learning...",
            "arxiv": "https://arxiv.org/abs/2308.01614",
            "tags": ["Self-Supervised", "Representation Learning"],
            "citations": 45,
            "organization": "ECMWF / TU Berlin",
        },
    ]

    # Sort by date (most recent first)
    papers_df = pd.DataFrame(PAPERS)
    papers_df['date'] = pd.to_datetime(papers_df['date'])
    papers_df = papers_df.sort_values('date', ascending=False)

    # Filter by category
    if category_filter:
        mask = papers_df['tags'].apply(lambda x: any(cat in x for cat in category_filter))
        papers_df = papers_df[mask]

    # Filter by venue
    if venue_filter:
        papers_df = papers_df[papers_df['venue'].isin(venue_filter)]

    # Display papers
    for _, paper in papers_df.iterrows():
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{paper['title']}</div>
            <div class="paper-authors">{paper['authors']} • {paper['organization']}</div>
            <span class="paper-venue">{paper['venue']}</span>
            <span style="margin-left: 10px; color: #666;">{paper['date'].strftime('%Y-%m-%d')}</span>
            <span style="margin-left: 10px; color: #666;">📚 {paper['citations']} citations</span>
            <p style="margin-top: 10px; color: #333;">{paper['abstract'][:200]}...</p>
            <div style="margin-top: 10px;">
                {"".join([f'<span class="tag">{tag}</span>' for tag in paper['tags']])}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_link1, col_link2, col_link3 = st.columns([1, 1, 4])
        with col_link1:
            st.markdown(f"[📄 arXiv]({paper['arxiv']})")
        with col_link2:
            if 'doi' in paper and paper['doi']:
                st.markdown(f"[🔗 DOI](https://doi.org/{paper['doi']})")

        st.markdown("---")


# =================== TAB 2: Model Releases ===================
with tab2:
    st.header("Recent Model Releases")

    MODEL_RELEASES = [
        {
            "name": "Aurora",
            "version": "1.0",
            "date": "2024-05",
            "organization": "Microsoft Research",
            "description": "Foundation model trained on 1M+ hours of weather/climate data. Achieves state-of-the-art on multiple benchmarks.",
            "link": "https://github.com/microsoft/aurora",
            "features": ["1.3B parameters", "0.25° resolution", "10-day forecasts", "Fine-tunable"],
        },
        {
            "name": "NeuralGCM",
            "version": "1.0",
            "date": "2024-07",
            "organization": "Google Research",
            "description": "Hybrid physics-ML model combining interpretability of GCMs with data-driven learning.",
            "link": "https://github.com/google-research/neuralgcm",
            "features": ["Physically consistent", "Long-term stable", "Climate projection capable"],
        },
        {
            "name": "GenCast",
            "version": "1.0",
            "date": "2023-12",
            "organization": "Google DeepMind",
            "description": "Diffusion-based probabilistic model for ensemble weather forecasting.",
            "link": "https://deepmind.google/discover/blog/gencast-predicts-weather-15-days-ahead",
            "features": ["15-day forecasts", "Probabilistic", "Ensemble generation"],
        },
        {
            "name": "GraphCast",
            "version": "1.0",
            "date": "2023-11",
            "organization": "Google DeepMind",
            "description": "Graph neural network for medium-range weather prediction.",
            "link": "https://github.com/google-deepmind/graphcast",
            "features": ["0.25° resolution", "10-day forecasts", "37M parameters"],
        },
        {
            "name": "FourCastNet",
            "version": "2.0",
            "date": "2023-09",
            "organization": "NVIDIA",
            "description": "Updated version with improved accuracy and efficiency.",
            "link": "https://github.com/NVlabs/FourCastNet",
            "features": ["Adaptive FNO", "45000x faster", "0.25° resolution"],
        },
        {
            "name": "Pangu-Weather",
            "version": "1.0",
            "date": "2023-07",
            "organization": "Huawei Cloud",
            "description": "3D Earth-Specific Transformer for global weather forecasting.",
            "link": "https://github.com/198808xc/Pangu-Weather",
            "features": ["3D attention", "Multiple time steps (1h-24h)", "256M parameters"],
        },
    ]

    for release in MODEL_RELEASES:
        st.markdown(f"""
        <div class="model-release">
            <h3>🚀 {release['name']} v{release['version']}</h3>
            <p><strong>{release['organization']}</strong> • Released {release['date']}</p>
            <p>{release['description']}</p>
            <p><strong>Key Features:</strong> {' • '.join(release['features'])}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"[View Repository]({release['link']})")
        st.markdown("---")


# =================== TAB 3: Community ===================
with tab3:
    st.header("Community Highlights")

    # Community posts
    COMMUNITY_POSTS = [
        {
            "title": "WeatherBench2 Evaluation Update",
            "author": "Google Research Team",
            "date": "2024-06-15",
            "content": "New evaluation metrics and additional baselines added to WeatherBench2. Now includes probabilistic metrics (CRPS, Brier score) and regional skill scores.",
            "link": "https://sites.research.google/weatherbench/",
            "type": "announcement",
        },
        {
            "title": "ECMWF AI/ML Strategy 2024",
            "author": "ECMWF",
            "date": "2024-05-20",
            "content": "ECMWF announces plans to integrate ML models into operational forecasting system by 2025. AIFS (AI Forecasting System) enters testing phase.",
            "link": "https://www.ecmwf.int/en/about/what-we-do/ai-at-ecmwf",
            "type": "announcement",
        },
        {
            "title": "Open Source Weather ML Benchmark",
            "author": "Community Initiative",
            "date": "2024-04-10",
            "content": "Community-driven effort to create standardized benchmarks for weather ML models with reproducible evaluation scripts.",
            "link": "https://github.com/openclimatefix/weatherbench2",
            "type": "project",
        },
        {
            "title": "Tutorial: Fine-tuning ClimaX for Regional Forecasting",
            "author": "Microsoft Research",
            "date": "2024-03-25",
            "content": "Step-by-step guide to fine-tuning the ClimaX foundation model for regional weather prediction tasks.",
            "link": "https://github.com/microsoft/ClimaX",
            "type": "tutorial",
        },
    ]

    for post in COMMUNITY_POSTS:
        icon = {"announcement": "📢", "project": "💻", "tutorial": "📖"}.get(post['type'], "📝")
        st.markdown(f"""
        <div class="community-highlight">
            <h4>{icon} {post['title']}</h4>
            <p><strong>{post['author']}</strong> • {post['date']}</p>
            <p>{post['content']}</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"[Learn More]({post['link']})")
        st.markdown("---")

    # Discussion section
    st.subheader("Join the Discussion")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Discord Communities:**
        - [Climate Change AI](https://www.climatechange.ai/) - ML for climate
        - [Weather & Climate Informatics](https://discord.gg/weatherml) - Weather ML
        - [Open Climate Fix](https://openclimatefix.org/) - Solar forecasting

        **Mailing Lists:**
        - ECMWF Newsletter
        - AMS AI Committee
        - EGU Climate Informatics
        """)

    with col2:
        st.markdown("""
        **Conferences:**
        - NeurIPS Climate Workshop (December)
        - ICLR AI4Earth Workshop (May)
        - AGU Fall Meeting (December)
        - EGU General Assembly (April)

        **Competitions:**
        - WeatherBench2 Leaderboard
        - Kaggle Weather Challenges
        - NOAA AI Challenges
        """)


# =================== TAB 4: Reading List ===================
with tab4:
    st.header("Essential Reading List")

    st.markdown("""
    **Curated papers every weather AI researcher should read.**
    Organized by topic with difficulty ratings.
    """)

    READING_LIST = {
        "Foundations": [
            {"title": "Deep Learning for Weather Prediction", "authors": "Review Paper", "year": 2023, "difficulty": "Beginner"},
            {"title": "Machine Learning for Weather and Climate Prediction", "authors": "Schneider et al.", "year": 2017, "difficulty": "Beginner"},
            {"title": "Can deep learning beat numerical weather prediction?", "authors": "Schultz et al.", "year": 2021, "difficulty": "Intermediate"},
        ],
        "Graph Neural Networks": [
            {"title": "GraphCast: Learning skillful medium-range global weather forecasting", "authors": "Lam et al.", "year": 2023, "difficulty": "Advanced"},
            {"title": "Learning to Simulate Complex Physics with Graph Networks", "authors": "Sanchez-Gonzalez et al.", "year": 2020, "difficulty": "Intermediate"},
        ],
        "Transformers & Attention": [
            {"title": "FourCastNet: Global Data-driven Weather Model", "authors": "Pathak et al.", "year": 2022, "difficulty": "Advanced"},
            {"title": "Pangu-Weather: 3D Earth-Specific Transformer", "authors": "Bi et al.", "year": 2023, "difficulty": "Advanced"},
            {"title": "ClimaX: A foundation model for weather and climate", "authors": "Nguyen et al.", "year": 2023, "difficulty": "Advanced"},
        ],
        "Probabilistic & Ensemble": [
            {"title": "GenCast: Diffusion-based ensemble forecasting", "authors": "Price et al.", "year": 2023, "difficulty": "Advanced"},
            {"title": "Denoising Diffusion Probabilistic Models", "authors": "Ho et al.", "year": 2020, "difficulty": "Intermediate"},
        ],
        "Physics-Informed": [
            {"title": "NeuralGCM: Neural General Circulation Models", "authors": "Kochkov et al.", "year": 2024, "difficulty": "Advanced"},
            {"title": "Physics-Informed Neural Networks", "authors": "Raissi et al.", "year": 2019, "difficulty": "Intermediate"},
        ],
    }

    for category, papers in READING_LIST.items():
        with st.expander(f"📚 {category}", expanded=True):
            for paper in papers:
                difficulty_color = {
                    "Beginner": "#28a745",
                    "Intermediate": "#ffc107",
                    "Advanced": "#dc3545"
                }.get(paper['difficulty'], "#6c757d")

                st.markdown(f"""
                **{paper['title']}** ({paper['year']})
                *{paper['authors']}*
                <span style="background-color: {difficulty_color}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.8em;">{paper['difficulty']}</span>
                """, unsafe_allow_html=True)


# =================== TAB 5: Learning Resources ===================
with tab5:
    st.header("Learning Resources")

    st.markdown("""
    **Educational materials for getting started with weather AI.**
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Video Courses")
        st.markdown("""
        - **Deep Learning for Weather Prediction** - Stanford CS course
        - **Machine Learning for Earth Observation** - ESA/EUMETSAT
        - **Neural Network Weather Models** - ECMWF Learning
        - **Climate Informatics** - Climate Change AI

        **YouTube Channels:**
        - Yannic Kilcher (ML paper explanations)
        - Two Minute Papers
        - ECMWF Official
        """)

        st.subheader("📖 Textbooks")
        st.markdown("""
        - *Atmospheric Modeling, Data Assimilation and Predictability* - Kalnay
        - *Deep Learning* - Goodfellow, Bengio, Courville
        - *Pattern Recognition and Machine Learning* - Bishop
        - *An Introduction to Statistical Learning* - James et al.
        """)

    with col2:
        st.subheader("💻 Code Tutorials")
        st.markdown("""
        - [WeatherBench2 Quickstart](https://weatherbench2.readthedocs.io/)
        - [ClimaX Fine-tuning Tutorial](https://github.com/microsoft/ClimaX)
        - [GraphCast JAX Implementation](https://github.com/google-deepmind/graphcast)
        - [PyTorch Weather Models](https://pytorch.org/tutorials/)
        """)

        st.subheader("🔧 Tools & Frameworks")
        st.markdown("""
        - **xarray** - N-D labeled arrays
        - **zarr** - Cloud-optimized storage
        - **cartopy** - Map projections
        - **cfgrib** - GRIB file handling
        - **climetlab** - ECMWF data access
        - **earth2mip** - NVIDIA weather inference
        """)

        st.subheader("📊 Datasets")
        st.markdown("""
        - [ERA5](https://cds.climate.copernicus.eu/) - ECMWF Reanalysis
        - [WeatherBench2](https://weatherbench2.readthedocs.io/) - Benchmark data
        - [NOAA Open Data](https://www.noaa.gov/information-technology/open-data-dissemination)
        - [Copernicus Climate Store](https://cds.climate.copernicus.eu/)
        """)

# Footer
st.markdown("---")
st.caption("""
**Disclaimer:** This research feed is curated for educational purposes.
For the most current information, please refer to official sources and arXiv.

*Want to submit content? Contact the WeatherFlow team.*
""")
