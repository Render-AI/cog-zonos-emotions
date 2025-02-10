from cog import BasePredictor, Input, Path
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict


class Predictor(BasePredictor):

  def setup(self):
    """Load the model into memory"""
    self.model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device="cuda")
    self.model.bfloat16()

  def predict(self, text: str = Input(description="Text to speak"), audio: Path = Input(description="Audio with voice to mimic")):
    wav, sampling_rate = torchaudio.load(audio)
    spk_embedding = self.model.embed_spk_audio(wav, sampling_rate)

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

