# streamlit_app.py
import os
import subprocess
import tempfile
from pathlib import Path
import streamlit as st
from huggingface_hub import HfApi, whoami

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

HF_TOKEN = st.secrets.get("HF_TOKEN")  # Optional: set in Streamlit secrets

st.set_page_config(page_title="GGUF-my-repo Streamlit", layout="wide")

st.title("Create your own GGUF Quants ⚡")

# User inputs
model_id = st.text_input("HuggingFace Model Repo ID", placeholder="username/model-name")
q_method = st.selectbox(
    "Quantization Method",
    ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_K_S", "Q4_K_M", "Q5_0", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0"],
    index=5
)
use_imatrix = st.checkbox("Use Imatrix Quantization", value=False)
imatrix_q_method = st.selectbox(
    "Imatrix Quantization Method",
    ["IQ3_M", "IQ3_XXS", "Q4_K_M", "Q4_K_S", "IQ4_NL", "IQ4_XS", "Q5_K_M", "Q5_K_S"],
    index=4
)
private_repo = st.checkbox("Create Private Repo", value=False)
train_data_file = st.file_uploader("Training Data File (.txt)", type=["txt"])
split_model = st.checkbox("Split Model into Shards")
split_max_tensors = st.number_input("Max Tensors per File", value=256)
split_max_size = st.text_input("Max File Size (e.g., 256M, 5G)", value="")

def escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
         .replace("\n", "<br/>")
    )

def run_process(cmd_list, timeout=None):
    try:
        result = subprocess.run(cmd_list, shell=False, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Process timed out."

def process_model():
    if not model_id:
        st.error("Please provide a HuggingFace Model Repo ID.")
        return
    
    st.info(f"Processing model: {model_id}")
    
    try:
        api = HfApi(token=HF_TOKEN)
        whoami(HF_TOKEN)  # Validate token
        
        with st.spinner("Downloading model..."):
            tmp_dir = tempfile.TemporaryDirectory()
            local_dir = Path(tmp_dir.name)/Path(model_id).name
            api.snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
            st.success("Model downloaded!")
        
        st.info("Ready for conversion and quantization.")
        st.warning("Conversion and quantization scripts should be available in llama.cpp folder.")

        st.success("✅ Processing finished! (Demo version does not run full quantization on Cloud)")

    except Exception as e:
        st.error(f"❌ ERROR: {escape(str(e))}")

if st.button("Run GGUF Processing"):
    process_model()
