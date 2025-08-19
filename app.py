import os
import subprocess
import tempfile
from pathlib import Path
from textwrap import dedent

import streamlit as st
from huggingface_hub import HfApi, whoami, ModelCard

HF_TOKEN = os.environ.get("HF_TOKEN")  # Set in Streamlit secrets
CONVERSION_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"
LLAMA_CPP_PATH = "./llama.cpp"

st.set_page_config(page_title="GGUF Quantizer", layout="wide")
st.title("GGUF Quantizer ⚡")
st.write("Convert Hugging Face models to GGUF format, optionally apply imatrix quantization, split and upload.")

# --- User Inputs ---
model_id = st.text_input("Hub Model ID", placeholder="e.g., tloen/alpaca-lora-7b")
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
private_repo = st.checkbox("Private Repo", value=False)
train_data_file = st.file_uploader("Training Data File (txt)", type=["txt"])
split_model = st.checkbox("Split Model", value=False)
split_max_tensors = st.number_input("Max Tensors per File", value=256)
split_max_size = st.text_input("Max File Size", placeholder="Example: 256M or 5G")

st.markdown("---")

# --- Helper Functions ---
def escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("\n", "<br/>")

def run_command(cmd, desc="Running"):
    st.info(f"🔹 {desc}: `{' '.join(cmd)}`")
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    output = []
    for line in iter(process.stdout.readline, ""):
        st.text(line.strip())
        output.append(line)
    process.wait()
    return process.returncode, "".join(output)

def generate_importance_matrix(model_path: str, train_data_path: str, imatrix_path: str):
    imatrix_cmd = [
        f"{LLAMA_CPP_PATH}/llama-imatrix",
        "-m", model_path,
        "-f", train_data_path,
        "-ngl", "99",
        "--output-frequency", "10",
        "-o", imatrix_path
    ]
    run_command(imatrix_cmd, "Generating Importance Matrix")

def split_upload_model(model_path: str, outdir: str, repo_id: str, oauth_token, split_max_tensors=256, split_max_size=None):
    split_cmd = [f"{LLAMA_CPP_PATH}/llama-gguf-split", "--split"]
    if split_max_size:
        split_cmd += ["--split-max-size", split_max_size]
    else:
        split_cmd += ["--split-max-tensors", str(split_max_tensors)]
    prefix = '.'.join(model_path.split('.')[:-1])
    split_cmd += [model_path, prefix]
    run_command(split_cmd, "Splitting Model")
    
    api = HfApi(token=oauth_token)
    sharded_files = [f for f in os.listdir(outdir) if f.startswith(Path(model_path).stem) and f.endswith(".gguf")]
    for f in sharded_files:
        file_path = os.path.join(outdir, f)
        api.upload_file(file_path, path_in_repo=f, repo_id=repo_id)
        st.success(f"Uploaded: {f}")

# --- Main Process ---
def process_model():
    if HF_TOKEN is None:
        st.error("HF_TOKEN not set! Add it in Streamlit secrets.")
        return

    api = HfApi(token=HF_TOKEN)
    username = whoami(HF_TOKEN)["name"]
    model_name = model_id.split("/")[-1]

    st.info(f"Downloading model {model_id} ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        local_dir = Path(tmpdir)/model_name
        api.snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
        st.success(f"Model downloaded to {local_dir}")

        # Convert
        fp16_path = Path(tmpdir)/f"{model_name}.fp16.gguf"
        ret, out = run_command(["python", CONVERSION_SCRIPT, str(local_dir), "--outtype", "f16", "--outfile", str(fp16_path)], "Converting model")
        if ret != 0:
            st.error(f"Conversion failed:\n{out}")
            return
        st.success(f"Model converted: {fp16_path}")

        # Imatrix
        imatrix_path = Path(tmpdir)/"imatrix.dat"
        if use_imatrix:
            if train_data_file:
                train_path = Path(tmpdir)/train_data_file.name
                train_data_file.seek(0)
                train_path.write_bytes(train_data_file.read())
            else:
                st.warning("No training file provided. Using default fallback dataset.")
                train_path = Path(f"{LLAMA_CPP_PATH}/groups_merged.txt")
            generate_importance_matrix(fp16_path, str(train_path), str(imatrix_path))

        # Quantized GGUF naming
        quant_name = f"{model_name.lower()}-{imatrix_q_method if use_imatrix else q_method}.gguf"
        quant_path = Path(tmpdir)/quant_name

        # Quantization command
        quant_cmd = [f"{LLAMA_CPP_PATH}/llama-quantize"]
        if use_imatrix:
            quant_cmd += ["--imatrix", str(imatrix_path)]
        quant_cmd += [str(fp16_path), str(quant_path), imatrix_q_method if use_imatrix else q_method]
        ret, out = run_command(quant_cmd, "Quantizing model")
        if ret != 0:
            st.error(f"Quantization failed:\n{out}")
            return
        st.success(f"Quantized model: {quant_path}")

        # Create repo
        repo_name = f"{username}/{model_name}-{imatrix_q_method if use_imatrix else q_method}-GGUF"
        api.create_repo(repo_id=repo_name, exist_ok=True, private=private_repo)
        st.success(f"Repo created: {repo_name}")

        # Upload GGUF
        api.upload_file(str(quant_path), path_in_repo=quant_name, repo_id=repo_name)
        st.success(f"Uploaded GGUF: {quant_name}")

        # Upload imatrix if exists
        if imatrix_path.exists():
            api.upload_file(str(imatrix_path), path_in_repo="imatrix.dat", repo_id=repo_name)
            st.success("Uploaded imatrix.dat")

        # Optional split
        if split_model:
            split_upload_model(str(quant_path), tmpdir, repo_name, HF_TOKEN, split_max_tensors, split_max_size)

        st.markdown(f"✅ **Done!** View repo: [https://huggingface.co/{repo_name}](https://huggingface.co/{repo_name})")

if st.button("Run Quantization"):
    process_model()
