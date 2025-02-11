from cog import Input, Path, BaseModel
import torch
import torchaudio
from zonos.model import Zonos


class TrainingOutput(BaseModel):
  weights: Path

def train(audio: Path = Input(description="Audio with voice to mimic", default=None), weights: Path = Input(description="Existing weights to absorb", default=None)) -> TrainingOutput:
  if weights:
    return TrainingOutput(weights=weights)
  else:
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    model.bfloat16()
    wav, sampling_rate = torchaudio.load(audio)
    spk_embedding = model.make_speaker_embedding(wav, sampling_rate)
    torch.save(spk_embedding, "spk_embedding.pt")
    return TrainingOutput(weights=Path("spk_embedding.pt"))
