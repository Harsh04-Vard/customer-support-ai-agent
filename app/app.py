import sys
from pathlib import Path

import streamlit as st


# ---------------------------------------------------------
# Project path setup
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_pipeline


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Apple Support AI Agent",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>

        /* Main background */
        .stApp {
            background-color: #f5f7fb;
        }

        /* Main content */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }

        /* Header */
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.8rem;
        }

        /* Cards */
        .card {
            background: white;
            border-radius: 14px;
            padding: 1.3rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            margin-bottom: 1rem;
        }

        .card-title {
            font-size: 1rem;
            font-weight: 650;
            color: #111827;
            margin-bottom: 0.7rem;
        }

        /* Decision badges */
        .auto-badge {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-weight: 700;
            font-size: 0.85rem;
        }

        .escalate-badge {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: #fee2e2;
            color: #991b1b;
            font-weight: 700;
            font-size: 0.85rem;
        }

        /* Intent badge */
        .intent-badge {
            display: inline-block;
            padding: 0.45rem 0.9rem;
            border-radius: 999px;
            background: #e0edff;
            color: #1d4ed8;
            font-weight: 600;
            font-size: 0.85rem;
        }

        /* Similarity */
        .similarity {
            color: #6b7280;
            font-size: 0.85rem;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #ffffff;
            border-right: 1px solid #e5e7eb;
        }

        /* Text area */
        textarea {
            border-radius: 10px !important;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 9px;
            font-weight: 600;
            height: 2.8rem;
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## 🍎 Support Agent")

    st.markdown(
        """
        <div style="
            color:#6b7280;
            font-size:0.9rem;
            line-height:1.5;
        ">
            AI-powered customer support assistant built using
            historical AppleSupport conversations.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### System")

    st.markdown(
        """
        **Brand:** AppleSupport

        **Intent Model:** TF-IDF + Logistic Regression

        **Retrieval:** FAISS + Sentence Transformers

        **Generation:** Groq LLM

        **Cases:** 105K+ historical conversations
        """
    )

    st.divider()

    st.markdown("### Pipeline")

    st.markdown(
        """
        1. Customer message
        2. Intent classification
        3. Similar-case retrieval
        4. Grounded response generation
        5. Auto-handle / escalation
        """
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">Customer Support AI Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Classify, retrieve, and generate grounded support responses."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Input section
# ---------------------------------------------------------
st.markdown(
    """
    <div class="card">
        <div class="card-title">Customer Message</div>
    </div>
    """,
    unsafe_allow_html=True,
)

customer_message = st.text_area(
    "Enter the customer's issue",
    placeholder=(
        "Example: My iPhone keeps freezing after the latest iOS update."
    ),
    height=130,
    label_visibility="collapsed",
)


run_button = st.button(
    "Run Support Agent",
    type="primary",
    use_container_width=True,
)


# ---------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------
if run_button:

    if not customer_message.strip():

        st.warning("Please enter a customer message first.")

    else:

        with st.spinner("Analyzing customer message..."):

            try:
                result = run_pipeline(customer_message.strip())

            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        st.divider()

        # -------------------------------------------------
        # Decision + Intent + Similarity
        # -------------------------------------------------
        col1, col2, col3 = st.columns(3)

        # Decision
        with col1:

            st.markdown("**Decision**")

            if result["decision"] == "AUTO-HANDLE":

                st.markdown(
                    '<span class="auto-badge">✓ AUTO-HANDLE</span>',
                    unsafe_allow_html=True,
                )

            else:

                st.markdown(
                    '<span class="escalate-badge">⚠ ESCALATE</span>',
                    unsafe_allow_html=True,
                )

        # Intent
        with col2:

            st.markdown("**Predicted Intent**")

            st.markdown(
                f'<span class="intent-badge">{result["intent"]}</span>',
                unsafe_allow_html=True,
            )

        # Similarity
        with col3:

            st.markdown("**Top Retrieval Similarity**")

            st.markdown(
                f"""
                <div style="
                    font-size:1.35rem;
                    font-weight:700;
                    color:#111827;
                    margin-top:0.35rem;
                ">
                    {result["top_similarity"]:.4f}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # Escalation reason
        # -------------------------------------------------
        if result["decision"] == "ESCALATE":

            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">
                        Escalation Reason
                    </div>

                    <div style="color:#991b1b;">
                        {result["reason"]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # Generated response
        # -------------------------------------------------
        st.markdown(
            '<h3 style="color:#111827; margin-top:1.5rem; margin-bottom:0.8rem;">'
            'Generated Response'
            '</h3>',
            unsafe_allow_html=True,
        )

        response_text = result["response"]

        st.markdown(
            '<div style="background:white; border:1px solid #e5e7eb; '
            'border-radius:14px; padding:1.3rem; '
            'box-shadow:0 2px 8px rgba(0,0,0,0.04);">'
            '<div style="font-size:0.85rem; color:#6b7280; '
            'margin-bottom:0.6rem;">'
            'Suggested customer-facing response'
            '</div>'
            '<div style="font-size:1rem; line-height:1.65; color:#111827;">'
            f'{response_text}'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # -------------------------------------------------
        # Retrieved cases
        # -------------------------------------------------
        st.markdown("### Retrieved Historical Cases")

        st.caption(
            "The cases below are the historical examples used "
            "to ground the generated response."
        )

        for i, case in enumerate(
            result["retrieved_cases"],
            start=1,
        ):

            with st.expander(f"Case {i}"):

                st.write(case)