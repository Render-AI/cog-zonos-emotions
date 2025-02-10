from huggingface_hub import hf_hub_download

config_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-transformer", filename="config.json")
model_path = hf_hub_download(repo_id="Zyphra/Zonos-v0.1-transformer", filename="model.safetensors")

# move into local directory
import os
os.makedirs("zonos-v0.1", exist_ok=True)
os.rename(config_path, "zonos-v0.1/config.json")
os.rename(model_path, "zonos-v0.1/model.safetensors")

