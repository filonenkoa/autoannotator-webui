import streamlit as st

st.set_page_config(
    page_title="Autoannotator WebUI",
    page_icon="😸",
)

st.write("# Welcome to Autoannotator WebUI")

st.sidebar.success("Select a task above.")

st.markdown(
    """
    **Autoannotator** is an extendable tool for automatic annotation of image data by a combination of deep neural networks.
    
    The primary objective of this annotator is to prioritize the accuracy and quality of predictions over speed.
    The autoannotator has been specifically designed to surpass the precision offered by most publicly available tools.
    It leverages ensembles of deep neural models to ensure the utmost quality in its predictions.
    It is important to note that neural networks trained on clean datasets tend to yield superior results compared to those trained on larger but noisier datasets.
"""
)

st.image("https://github.com/CatsWhoTrain/autoannotator/raw/main/readme_files/auto-annotate_logo.jpg")

st.markdown("## Implemented tasks")

st.page_link("pages/1_Human_Detection.py", label="Human Detection", icon="1️⃣")