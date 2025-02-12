from cog import BasePredictor, Input, Path
import torch
import torchaudio
from pget import pget_manifest, pget_url
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from typing import Optional

class Predictor(BasePredictor):

  def setup(self, weights: Optional[Path] = None):
    pget_manifest('manifest.pget')
    # Load the default transformer model
    self.current_model_dir = "./models/transformer"
    self.model = Zonos.from_local(
        f"{self.current_model_dir}/config.json",
        f"{self.current_model_dir}/model.safetensors",
        device="cuda"
    )
    self.model.bfloat16()
    
    if weights is not None:
        # Get the actual URL from the URLFile object
        weights_url = weights.url if hasattr(weights, 'url') else str(weights)
        pget_url("https://replicate.delivery/" + weights_url, "spk_embedding.pt")
        with open("spk_embedding.pt", "rb") as f:
            self.speaker_embedding = torch.load(f)
    else:
        self.speaker_embedding = None

  def predict(
      self,
      text: str = Input(description="Text to speak!"),
      audio: Path = Input(description="(Optional) Audio with voice to mimic", default=None),
      model_type: str = Input(description="Model type to use ('transformer' or 'hybrid')", default="transformer")
  ) -> Path:
    # Only initialize a new model if switching to a different type
    model_dir = f"./models/{model_type.lower()}"
    if model_dir != self.current_model_dir:
        self.model = Zonos.from_local(
            f"{model_dir}/config.json",
            f"{model_dir}/model.safetensors",
            device="cuda"
        )
        self.model.bfloat16()
        self.current_model_dir = model_dir

    if self.speaker_embedding is None:
      wav, sampling_rate = torchaudio.load(audio)
      spk_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
    else:
      spk_embedding = self.speaker_embedding

    torch.manual_seed(421)

    cond_dict = make_cond_dict(
        text=text,
        speaker=spk_embedding.to(torch.bfloat16),
        language="en-us",
    )
    conditioning = self.model.prepare_conditioning(cond_dict)

    codes = self.model.generate(conditioning)

    wavs = self.model.autoencoder.decode(codes).cpu()
    torchaudio.save("sample.wav", wavs[0], self.model.autoencoder.sampling_rate)
    return Path("sample.wav")

