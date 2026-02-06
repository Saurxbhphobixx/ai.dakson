import os
import io
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Page config
st.set_page_config(page_title="AI Data Chatbot", layout="wide")

# Sidebar
st.sidebar.title("Settings")
api_key = os.getenv("OPENAI_API_KEY")
model = st.sidebar.selectbox("Model", ["gpt-4.1-mini", "gpt-4o-mini"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 100, 2000, 800, 50)
st.sidebar.markdown("---")
st.sidebar.success("API key loaded from environment" if api_key else "Set OPENAI_API_KEY env var")
st.sidebar.markdown("Run: `python -m streamlit run app.py`")

# OpenAI client
client = OpenAI(api_key=api_key) if api_key else None

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant for data analysis and Python code."}]
if "df" not in st.session_state:
    st.session_state.df = None
if "eda_code" not in st.session_state:
    st.session_state.eda_code = ""

st.title("AI Data Analyst Chatbot")

# Layout
left, right = st.columns([2, 3])

# Left: Data
with left:
    st.subheader("Upload Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            df = pd.read_csv(io.BytesIO(uploaded.read()))
            st.session_state.df = df
            st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")

    if st.session_state.df is not None:
        st.write("Preview:")
        st.dataframe(st.session_state.df.head(100))

        if st.button("Show Summary"):
            st.write(st.session_state.df.describe(include="all").transpose())

        # Quick Charts
        st.subheader("Quick Charts")
        numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            col_x = st.selectbox("Select numeric column", numeric_cols)
            fig = px.histogram(st.session_state.df, x=col_x, title=f"Distribution of {col_x}")
            st.plotly_chart(fig, use_container_width=True)

            if len(numeric_cols) > 1:
                corr = st.session_state.df[numeric_cols].corr()
                fig2 = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No numeric columns to plot.")

# Right: Chat + AI actions
with right:
    st.subheader("Chat with AI")

    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

    prompt = st.text_area("Message (try: 'plot', 'suggest EDA', 'generate eda code')", height=100)
    col_send, col_clear = st.columns(2)

    send = col_send.button("Send")
    clear = col_clear.button("Clear Chat")

    if clear:
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant for data analysis and Python code."}]
        st.session_state.eda_code = ""
        st.rerun()

    # 🔹 NEW: EDA Code Generator
    st.markdown("### 🧪 EDA Code Generator")
    if st.button("Generate EDA Code"):
        if not client:
            st.error("No OpenAI API key configured.")
        elif st.session_state.df is None:
            st.warning("Upload a CSV first.")
        else:
            cols = ", ".join(st.session_state.df.columns.tolist())
            prompt_eda = f"""
            Generate clean, well-commented Python EDA code using pandas and plotly for a dataset with columns:
            {cols}
            The code should:
            - Load a CSV file
            - Show basic info and missing values
            - Plot distributions for numeric columns
            - Plot correlation heatmap
            - Be ready to run in Jupyter or a .py file
            """
            with st.spinner("Generating EDA code..."):
                resp = client.responses.create(
                    model=model,
                    input=prompt_eda,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                st.session_state.eda_code = resp.output_text

    if st.session_state.eda_code:
        st.code(st.session_state.eda_code, language="python")

    # Chat logic
    if send and prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})

        reply = ""
        df = st.session_state.df

        if "plot" in prompt.lower() and df is not None:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                fig = px.line(df, y=numeric_cols[0], title=f"Auto plot of {numeric_cols[0]}")
                st.plotly_chart(fig, use_container_width=True)
                reply = f"I plotted **{numeric_cols[0]}** for you."
            else:
                reply = "Your dataset has no numeric columns to plot."

        else:
            if not client:
                reply = "No OpenAI API key configured."
            else:
                try:
                    response = client.responses.create(
                        model=model,
                        input=prompt,
                        temperature=temperature,
                        max_output_tokens=max_tokens,
                    )
                    reply = response.output_text
                except Exception as e:
                    reply = f"[Error] {e}"

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# Footer
st.markdown("---")
st.caption("Step 1 done: AI-generated EDA code. Next: Auto ML + PDF report.")
