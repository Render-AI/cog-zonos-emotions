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
        text: str = Input(description="Text to generate speech from"),
        audio: Path = Input(
            description="Path to audio file for voice cloning (optional)",
            default=None,
        ),
        language: str = Input(
            description="Language code for speech generation",
            default="en-us",
            choices=['en-us', 'en-gb', 'ja', 'cmn', 'yue', 'fr-fr', 'de']
        ),
        model_type: str = Input(
            description="Model type to use",
            default="transformer",
            choices=['transformer', 'hybrid']
        ),
        emotion: str = Input(
            description=(
                "Optionally pass a comma-separated list of 8 floats for your desired emotion vector\n"
                "in the order [Happiness, Sadness, Disgust, Fear, Surprise, Anger, Other, Neutral].\n"
                "For example: '0.5,0.2,0.0,0.0,0.3,0.1,0.0,0.0'.\n"
                "If empty or invalid, defaults to the built-in neutralish emotion."
            ),
            default="",  # Empty string => use the default get_cond_dict behavior
        ),
        speaking_rate: float = Input(
            description=(
                "Speaking rate in phonemes per second. Default is 15.0.\n"
                "10-12 is slow and clear, 15-17 is natural conversational,\n"
                "20+ is fast. Values above 30 may produce artifacts."
            ),
            default=15.0,
            ge=5.0,
            le=30.0,
        ),
        seed: int = Input(
            description="Seed for reproducibility (optional)",
            default=None,
        ),
    ) -> Path:
        """
        Generate speech from text with an optional speaker embedding from audio,
        and an optional emotion vector.
        """
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

        # If provided audio, extract speaker embedding
        if audio is not None:
            wav, sampling_rate = torchaudio.load(audio)
            spk_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
        else:
            spk_embedding = None

        # Parse custom emotion vector if given
        emotion_tensor = None
        if emotion.strip():
            try:
                # e.g. "0.5,0.1,0.0,0.0,0.3,0.1,0.0,0.0"
                emo_vals = [float(x.strip()) for x in emotion.split(",")]
                if len(emo_vals) != 8:
                    raise ValueError("Emotion must be exactly 8 floats.")
                # Convert to a [1 x 8] (or [1, 8]) which make_cond_dict then unsqueezes further
                emotion_tensor = torch.tensor([emo_vals], dtype=torch.float32, device="cuda").bfloat16()
            except Exception as e:
                print(f"[WARN] Could not parse emotion vector: {e}")
                emotion_tensor = None

        # Create the conditioning dictionary
        if seed:
            torch.manual_seed(seed)
        else:
            # Generate random seed for reproducibility
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            print(f"Using random seed: {seed}")

        cond_dict = make_cond_dict(
            text=text,
            speaker=spk_embedding.to(torch.bfloat16) if spk_embedding is not None else None,
            language=language,
            emotion=emotion_tensor,  # If None, make_cond_dict will handle defaults
            speaking_rate=speaking_rate,  # Add speaking rate to conditioning
        )
        conditioning = self.model.prepare_conditioning(cond_dict)

        # Generate codes and decode to waveform
        codes = self.model.generate(conditioning)
        wavs = self.model.autoencoder.decode(codes).cpu()

        out_path = Path("sample.wav")
        torchaudio.save(str(out_path), wavs[0], self.model.autoencoder.sampling_rate)
        return out_path