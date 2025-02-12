from cog import BasePredictor, Input, Path
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict


class Predictor(BasePredictor):
    def setup(self):
        # Load a default transformer model
        self.current_model_dir = "./models/transformer"
        self.model = Zonos.from_local(
            f"{self.current_model_dir}/config.json",
            f"{self.current_model_dir}/model.safetensors",
            device="cuda",
        )
        self.model.bfloat16()

    def predict(
        self,
        text: str = Input(description="Text to speak!"),
        audio: Path = Input(
            description="Optional Path to audio file to derive speaker embedding",
            default=None,
        ),
        language: str = Input(
            description="Language code (e.g. 'en-us', 'fr-ca', etc.)",
            default="en-us",
        ),
        model_type: str = Input(
            description="Model type to use ('transformer' or 'hybrid')",
            default="transformer",
        ),
    ) -> Path:
        """Generate speech from text with an optional speaker embedding from audio."""
        # If switching model type, load a new model
        model_dir = f"./models/{model_type.lower()}"
        if model_dir != self.current_model_dir:
            self.model = Zonos.from_local(
                f"{model_dir}/config.json",
                f"{model_dir}/model.safetensors",
                device="cuda",
            )
            self.model.bfloat16()
            self.current_model_dir = model_dir

        if audio is not None:
            # Load audio for speaker embedding
            wav, sampling_rate = torchaudio.load(audio)
            spk_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
        else:
            spk_embedding = None

        torch.manual_seed(421)

        cond_dict = make_cond_dict(
            text=text,
            speaker=spk_embedding.to(torch.bfloat16) if spk_embedding is not None else None,
            language=language,
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()

        out_path = Path("sample.wav")
        torchaudio.save(str(out_path), wavs[0], self.model.autoencoder.sampling_rate)
        return out_path