import os
from huggingface_hub import hf_hub_download

def download_models():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download both versions
    models = {
        "transformer": "Zyphra/Zonos-v0.1-transformer",
        "hybrid": "Zyphra/Zonos-v0.1-hybrid"
    }
    
    for model_type, repo_id in models.items():
        print(f"Downloading {model_type} model...")
        save_dir = f"models/{model_type}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Download both config and model files
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        
        # Copy files to our models directory
        os.system(f"cp {config_path} {save_dir}/config.json")
        os.system(f"cp {model_path} {save_dir}/model.safetensors")
        
        print(f"Saved {model_type} model to {save_dir}")

if __name__ == "__main__":
    download_models() 