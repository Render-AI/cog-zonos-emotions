from cog import BasePredictor, Input, Path
import torch
import torchaudio
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from pydub import AudioSegment

def approximate_phoneme_count(text: str) -> int:
    """
    A very rough approximation that assumes character count ~ phoneme count.
    Feel free to adjust as needed or implement a more sophisticated method.
    """
    return len(text)

def group_sentences(text, max_length=400):
    """
    Provided splitting logic, with minor edits to ensure it runs in this script.
    Splits text into chunks that do not exceed max_length. 
    """
    sentences = sent_tokenize(text)

    # Debug: print the length of the longest sentence
    print(f"Longest sentence: {max([len(sent) for sent in sentences]) if sentences else 0}")

    # While biggest sentence is bigger than max_length, try splitting by periods, then by commas
    while sentences and max([len(sent) for sent in sentences]) > max_length:
        max_sent = max(sentences, key=len)
        max_idx = sentences.index(max_sent)

        sentences_before = sentences[:max_idx]
        sentences_after = sentences[max_idx+1:]
        
        new_sentences = max_sent.split(".")
        new_sentences = [sent.strip() for sent in new_sentences if sent.strip() != ""]
        
        # check if a split sentence is still too big
        if new_sentences and max([len(sent) for sent in new_sentences]) > max_length:
            biggest_new_sent = max(new_sentences, key=len)
            bn_idx = new_sentences.index(biggest_new_sent)
            new_senteces_before = new_sentences[:bn_idx]
            new_senteces_after = new_sentences[bn_idx+1:]

            new_sentence_parts = biggest_new_sent.split(",")
            new_sentence_parts = [sent.strip() for sent in new_sentence_parts if sent.strip() != ""]
            new_sentences = new_senteces_before + new_sentence_parts + new_senteces_after
        
        sentences = sentences_before + new_sentences + sentences_after

    # Debug: print the length of the longest sentence after splitting
    if sentences:
        print(f"Longest sentence after split: {max([len(sent) for sent in sentences])}")

    # Merge sentences until we can't merge further without exceeding max_length
    while True:
        if len(sentences) <= 1:
            break

        # Find the shortest sentence
        min_index = min(range(len(sentences)), key=lambda i: len(sentences[i]))
        min_length = len(sentences[min_index])

        # Determine the nearest neighbor that is shorter (or equally short)
        if min_index == 0:
            next_index = min_index + 1
        elif min_index == len(sentences) - 1:
            next_index = min_index - 1
        else:
            left_length = len(sentences[min_index - 1])
            right_length = len(sentences[min_index + 1])
            next_index = (min_index - 1) if left_length <= right_length else (min_index + 1)

        # Check if merging would exceed the maximum length
        if min_length + len(sentences[next_index]) + 1 > max_length:
            break

        # Merge sentences
        if next_index > min_index:
            sentences[min_index] = f"{sentences[min_index]} {sentences[next_index]}"
            del sentences[next_index]
        else:
            sentences[next_index] = f"{sentences[next_index]} {sentences[min_index]}"
            del sentences[min_index]

    return sentences

class Predictor(BasePredictor):
    def setup(self):
        # Load the default transformer model from local
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
        and an optional emotion vector. Now includes logic to split text into multiple
        chunks if we exceed the maximum recommended generation length based on
        speaking_rate and reference audio length (up to 30 seconds).
        """

        # If switching model type, load appropriate model
        if model_type == "hybrid" and self.current_model_dir != "hybrid":
            self.model = Zonos.from_pretrained(
                repo_id="Zyphra/Zonos-v0.1-hybrid",
                device="cuda"
            )
            self.model.bfloat16()
            self.current_model_dir = "hybrid"
        elif model_type == "transformer" and self.current_model_dir != "./models/transformer":
            self.model = Zonos.from_local(
                "./models/transformer/config.json",
                "./models/transformer/model.safetensors",
                device="cuda",
            )
            self.model.bfloat16()
            self.current_model_dir = "./models/transformer"

        # If provided audio, extract speaker embedding and set reference length
        ref_seconds = 30.0
        spk_embedding = None
        if audio is not None:
            wav, sampling_rate = torchaudio.load(audio)
            spk_embedding = self.model.make_speaker_embedding(wav, sampling_rate)
            ref_seconds = wav.size(-1) / sampling_rate

        # Parse custom emotion vector if given
        emotion_tensor = None
        if emotion.strip():
            try:
                emo_vals = [float(x.strip()) for x in emotion.split(",")]
                if len(emo_vals) != 8:
                    raise ValueError("Emotion must be exactly 8 floats.")
                emotion_tensor = torch.tensor([emo_vals], dtype=torch.float32, device="cuda").bfloat16()
            except Exception as e:
                print(f"[WARN] Could not parse emotion vector: {e}")
                emotion_tensor = None

        # Seed
        if seed:
            torch.manual_seed(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            torch.manual_seed(seed)
            print(f"Using random seed: {seed}")

        # --------------------------------------
        # Split text if it exceeds max phonemes
        # --------------------------------------
        # Roughly estimate how many phonemes we can fit given the reference (or default) length
        # and the user-provided speaking_rate
        max_phonemes = int(ref_seconds * speaking_rate)
        total_phonemes_estimate = approximate_phoneme_count(text)

        # We'll do splitting only if total phonemes exceed the max recommended
        if total_phonemes_estimate > max_phonemes:
            # Convert that phoneme fraction to a proportional max_length for text splitting
            ratio = max_phonemes / float(total_phonemes_estimate)
            chunk_char_length = max(50, int(len(text) * ratio))
            print(
                f"Text likely too long ({total_phonemes_estimate} phonemes) for "
                f"{ref_seconds:.2f}s reference at {speaking_rate} phonemes/s. "
                f"Splitting into chunks of ~{chunk_char_length} characters."
            )

            text_chunks = group_sentences(text, max_length=chunk_char_length)
        else:
            print(f"Text is short enough ({total_phonemes_estimate} phonemes) for {ref_seconds:.2f}s reference at {speaking_rate} phonemes/s.")
            text_chunks = [text]

        # --------------------------------------
        # Perform generation chunk by chunk
        # --------------------------------------
        wav_files_list = []
        for chunk_index, chunk_text in enumerate(text_chunks):
            # Create the conditioning dictionary for this chunk
            print(f"Doing inference for chunk {chunk_index+1}/{len(text_chunks)}: {chunk_text}")
            cond_dict = make_cond_dict(
                text=chunk_text,
                speaker=spk_embedding.to(torch.bfloat16) if spk_embedding is not None else None,
                language=language,
                emotion=emotion_tensor,  # If None, make_cond_dict will handle defaults
                speaking_rate=float(speaking_rate),  # Add speaking rate to conditioning
            )
            conditioning = self.model.prepare_conditioning(cond_dict)

            # Generate codes and decode to waveform
            codes = self.model.generate(conditioning)
            wavs = self.model.autoencoder.decode(codes).cpu()

            out_path = f"sample_{chunk_index}.wav"
            wav_files_list.append(out_path)
            torchaudio.save(out_path, wavs[0], self.model.autoencoder.sampling_rate)

        print(f"Concatenating {len(wav_files_list)} chunks: {wav_files_list}")
        try:
            # Concatenate all chunks using pydub
            final_wav = AudioSegment.empty()
            for wav_file in wav_files_list:
                wav = AudioSegment.from_wav(str(wav_file))
                print(f"Adding chunk {wav_file} to final wav")
                final_wav += wav

            # Save the final concatenated wav
            final_wav.export("output.wav", format="wav")
            print(f"Saved final wav to output.wav")
            return Path("output.wav")
        except Exception as e:
            print(f"Error processing audio files: {str(e)}")
            return None