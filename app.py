# streamlit_app.py
import os
import subprocess
import signal
import tempfile
from pathlib import Path
from textwrap import dedent

import streamlit as st
from huggingface_hub import HfApi, ModelCard, whoami

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

HF_TOKEN = st.secrets.get("HF_TOKEN")  # Add your HF_TOKEN in Streamlit secrets
CONVERSION_SCRIPT = "./llama.cpp/convert_hf_to_gguf.py"


def escape(s: str) -> str:
    """Escape HTML for logging"""
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("\n", "<br/>")
    return s


def generate_importance_matrix(model_path: str, train_data_path: str, output_path: str):
    imatrix_command = [
        "./llama.cpp/llama-imatrix",
        "-m", model_path,
        "-f", train_data_path,
        "-ngl", "99",
        "--output-frequency", "10",
        "-o", output_path,
    ]
    if not os.path.isfile(model_path):
        raise Exception(f"Model file not found: {model_path}")

    process = subprocess.Popen(imatrix_command, shell=False)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    st.success("Importance matrix generated!")


def split_upload_model(model_path: str, outdir: str, repo_id: str, oauth_token, split_max_tensors=256, split_max_size=None):
    split_cmd = ["./llama.cpp/llama-gguf-split", "--split"]
    if split_max_size:
        split_cmd += ["--split-max-size", split_max_size]
    else:
        split_cmd += ["--split-max-tensors", str(split_max_tensors)]

    model_path_prefix = '.'.join(model_path.split('.')[:-1])
    split_cmd += [model_path, model_path_prefix]
    result = subprocess.run(split_cmd, shell=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Error splitting the model: {result.stderr}")
    st.success("Model split successfully!")

    api = HfApi(token=oauth_token)
    sharded_files = [f for f in os.listdir(outdir) if f.startswith(Path(model_path).stem) and f.endswith(".gguf")]
    uploaded_files = []
    for file in sharded_files:
        file_path = os.path.join(outdir, file)
        api.upload_file(path_or_fileobj=file_path, path_in_repo=file, repo_id=repo_id)
        uploaded_files.append(file_path)
    return uploaded_files


def process_model(model_id, q_method, use_imatrix, imatrix_q_method, private_repo, train_data_file, split_model, split_max_tensors, split_max_size):
    if not HF_TOKEN:
        st.error("HF_TOKEN is required in Streamlit secrets")
        return

    api = HfApi(token=HF_TOKEN)
    username = whoami(HF_TOKEN)["name"]
    model_name = model_id.split("/")[-1]

    with tempfile.TemporaryDirectory() as tmp_downloads, tempfile.TemporaryDirectory() as tmp_output:
        local_dir = Path(tmp_downloads)/model_name
        api.snapshot_download(repo_id=model_id, local_dir=local_dir, allow_patterns=["*.bin", "*.safetensors"])

        # Convert to FP16 GGUF
        fp16_path = Path(tmp_output)/f"{model_name}.fp16.gguf"
        result = subprocess.run(
            ["python", CONVERSION_SCRIPT, str(local_dir), "--outtype", "f16", "--outfile", str(fp16_path)],
            shell=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            st.error(result.stderr)
            return
        st.success(f"Converted to FP16: {fp16_path.name}")

        # Generate imatrix if requested
        if use_imatrix:
            train_path = train_data_file.name if train_data_file else "llama.cpp/groups_merged.txt"
            generate_importance_matrix(fp16_path, train_path, Path(tmp_output)/"imatrix.dat")

        # Quantization
        quant_name = f"{model_name.lower()}-{imatrix_q_method.lower()}-imat.gguf" if use_imatrix else f"{model_name.lower()}-{q_method.lower()}.gguf"
        quant_path = Path(tmp_output)/quant_name
        quant_cmd = ["./llama.cpp/llama-quantize"]
        if use_imatrix:
            quant_cmd += ["--imatrix", str(Path(tmp_output)/"imatrix.dat"), str(fp16_path), str(quant_path), imatrix_q_method]
        else:
            quant_cmd += [str(fp16_path), str(quant_path), q_method]

        result = subprocess.run(quant_cmd, shell=False, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(result.stderr)
            return
        st.success(f"Quantized model: {quant_path.name}")

        # Create repo
        new_repo_url = api.create_repo(repo_id=f"{username}/{quant_name}-GGUF", exist_ok=True, private=private_repo)
        uploaded_files = []
        if split_model:
            uploaded_files = split_upload_model(str(quant_path), tmp_output, new_repo_url.repo_id, HF_TOKEN, split_max_tensors, split_max_size)
        else:
            api.upload_file(str(quant_path), quant_path.name, new_repo_url.repo_id)
            uploaded_files.append(str(quant_path))

        # Upload imatrix if exists
        imatrix_file = Path(tmp_output)/"imatrix.dat"
        if imatrix_file.exists():
            api.upload_file(str(imatrix_file), "imatrix.dat", new_repo_url.repo_id)
            uploaded_files.append(str(imatrix_file))

        # Upload README
        card = ModelCard.load(model_id, token=HF_TOKEN)
        if card.data.tags is None:
            card.data.tags = []
        card.data.tags += ["llama-cpp", "gguf-my-repo"]
        card.text = dedent(f"# {new_repo_url.repo_id}\nConverted GGUF model from {model_id}")
        readme_path = Path(tmp_output)/"README.md"
        card.save(readme_path)
        api.upload_file(str(readme_path), "README.md", new_repo_url.repo_id)
        uploaded_files.append(str(readme_path))

        st.success(f"✅ Done! Repo: {new_repo_url.repo_id}")

        st.subheader("Download Uploaded Files")
        for f in uploaded_files:
            f_path = Path(f)
            if f_path.exists():
                with open(f_path, "rb") as file:
                    st.download_button(label=f"Download {f_path.name}", data=file, file_name=f_path.name)
