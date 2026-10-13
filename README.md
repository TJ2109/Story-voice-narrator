# Voice-Personalized Story Narration using a Fine-Tuned LLM and Zero-Shot Voice Cloning

An AI system that writes a short children's story from a few user choices, then narrates it **in the voice of a person you choose** from a short audio clip. Everything runs on a free Google Colab T4 GPU.

> Only use the voice of someone who has given you permission. Narration is AI-generated.

---

## 1. Overview

1. The user picks a **genre, theme, characters, and length**.
2. A **Qwen2.5-1.5B-Instruct** model, fine-tuned with **QLoRA** on TinyStories, writes the story.
3. The user can read and edit the story.
4. The user uploads or records a **10 to 30 second voice clip**. It is cleaned automatically.
5. **XTTS-v2** clones the voice and narrates the story segment by segment.
6. A **Gradio** app plays the narrated story.

An optional **RAG** module retrieves passages from a fairy tale library to inspire the story.

## 2. Architecture

```
            +----------------------+
 User ----> | genre/theme/characters| ----> prompt ----> Fine-tuned LLM (QLoRA) ----> story
            +----------------------+          ^                                        |
                                              | (optional) fairy tale RAG              | user edits
                                              |                                        v
 Voice clip --> preprocessing --> XTTS-v2 voice cloning --> segment-by-segment narration --> audio
                                                                         |
                                                              Gradio app (consent, no storage)
```

## 3. Repository structure

```
.
├── 01_AI_Story_Narrator_Main.ipynb     # full pipeline: Steps 1-8
├── 02_RAG_and_UI_Upgrade.ipynb         # RAG fairy tale library + improved interface
├── README.md
├── requirements.txt
└── .gitignore
```

Not stored in this repo: voice recordings, generated audio, and the LoRA adapter (`story_lora`). Voices are private, and the adapter is about 70 MB.

## 4. Environment and requirements

| Item | Value |
|---|---|
| Platform | Google Colab (free tier) |
| GPU | NVIDIA T4 (about 15 GB) |
| Python | 3.13 (Colab default at the time of the project) |
| Storage | Google Drive folder `story_narrator` with `data/`, `models/`, `audio/`, `outputs/` |
| Key pin | `transformers<5` (the installed `coqui-tts` needs it) |

Install used in the notebooks:

```
pip install -q coqui-tts "transformers<5" peft datasets accelerate bitsandbytes \
  noisereduce librosa soundfile jiwer speechbrain gradio
pip install -q sentence-transformers faiss-cpu      # RAG notebook only
```

## 5. Step-by-step description and configuration

### Step 1. Workspace setup

- **What:** switch Colab to a T4 GPU, mount Google Drive, and create the project folders.
- **Why:** free Colab sessions disconnect, so checkpoints and data are saved to Drive.
- **Config:** `BASE = /content/drive/MyDrive/story_narrator` with subfolders `data`, `models`, `audio`, `outputs`.

### Step 2. Voice cloning test (done first, because it is the riskiest part)

- **What:** clone a voice from one reference clip and synthesize a test sentence.
- **Model:** XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`), zero-shot cloning, no training.

| Setting | Value |
|---|---|
| Language | `en` |
| Reference clip | 15 to 30 seconds, one speaker, quiet room |
| License acceptance | `COQUI_TOS_AGREED=1` (non-commercial model license) |

- **Output:** a cloned-voice test WAV used to judge whether the approach works.

### Step 3. Audio preprocessing

- **What:** clean any uploaded clip before cloning, and reject unusable clips with a clear message.
- **Tools:** librosa, noisereduce, soundfile.

| Setting | Value |
|---|---|
| Sample rate / channels | 22,050 Hz, mono |
| Silence trimming | `top_db = 30` |
| Minimum speech length | 6 s (shorter clips are rejected) |
| Maximum length used | 30 s (longer clips are cut) |
| Noise reduction | stationary, `prop_decrease = 0.8` |
| Normalization | peak normalized to 0.95 |
| Noise check | warns if loud-to-quiet frame ratio is under 10 dB |

### Step 4. Data preparation and LLM fine-tuning (QLoRA)

- **What:** teach a small pretrained LLM to write a story from a structured prompt.
- **Data:** TinyStories, first 30,000 stories (29,500 train, 500 held-out evaluation).
- **Auto-labels:** TinyStories has no genre, theme, or character labels, so each story is labelled with keyword rules (themes such as kindness or bravery, genres such as animal tale or adventure, characters from frequent capitalized names, length from word count). These labels are rough.

Prompt format:

```
### Instruction:
Write a {length} {genre} for children.
Characters: {characters}
Theme: {theme}

### Story:
{story}
```

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Quantization | 4-bit NF4, double quantization, fp16 compute |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 18,464,768 of 1,562,179,072 (1.18%) |
| Max sequence length | 384 tokens |
| Batch size | 4 per device, gradient accumulation 4 (effective 16) |
| Learning rate / schedule | 2e-4, cosine, 30 warmup steps |
| Training length | 500 steps (about 8,000 sequences, under one epoch of the subset) |
| Optimizer | `paged_adamw_8bit` |
| Precision / memory | fp16, gradient checkpointing |
| Checkpointing | every 100 steps to Drive, last 2 kept |

- **Result:** held-out eval loss 1.167, **perplexity 3.21**. The adapter is saved to `models/story_lora` on Drive.

### Step 5. Story generator

- **What:** load the base model with the saved adapter, generate a story, and split it into short segments for TTS.

| Setting | Value |
|---|---|
| Sampling | temperature 0.8 (user slider 0.5 to 1.1), top-p 0.9 |
| Repetition penalty | 1.1 |
| Max new tokens | 350 (first app), 250 short / 400 medium (RAG notebook) |
| Cleanup | an unfinished last sentence is dropped |
| Segmenting | sentences grouped up to 220 characters each |

### Step 6. Full narration pipeline

- **What:** story, then segments, then voice-cloned speech for each segment, then one audio file.

| Setting | Value |
|---|---|
| TTS | XTTS-v2, `language = en` |
| Pause between segments | 0.4 s |
| Output rate | the model's native output rate |
| Multiple reference clips | supported (a list is passed to XTTS to help keep accent) |

### Step 7. Evaluation

| Metric | Method | Result |
|---|---|---|
| LLM perplexity | exp(eval loss) on 500 held-out stories | **3.21** |
| Speaker similarity | cosine similarity of SpeechBrain ECAPA-VoxCeleb embeddings (16 kHz) between the reference and the narration | **0.627** |
| Word error rate | Whisper-small transcription of the narration vs the story text (lowercased, punctuation removed, 30 s chunks) | **0.029** |
| Listening test | 5 to 10 listeners rate naturalness, similarity, and story quality (1 to 5) | _add your scores here_ |

Speaker similarity mainly reflects timbre, so it does not show accent drift. Accent similarity is judged in the listening test.

### Step 8. Gradio app

- **What:** a step-by-step web interface: choose a story, read and edit it, then choose the voice.

| Feature | Detail |
|---|---|
| Inputs | genre, theme (custom allowed), characters, length, creativity slider |
| Voice input | upload or microphone, plus an optional second clip |
| Safety | consent checkbox is required before narration |
| Privacy | cleaned voice clips are deleted right after use, nothing is stored |
| Feedback | progress messages, friendly errors, voice quality warnings |
| Launch | `demo.launch(share=True)` gives a temporary public link while Colab runs |

## 6. Optional RAG module (notebook 02)

- **What:** retrieve a passage from a fairy tale library and add it to the prompt as inspiration.

| Setting | Value |
|---|---|
| Corpus | Grimm's and Andersen's fairy tales, Project Gutenberg (public domain) |
| Chunking | paragraphs merged to 300 to 800 characters, giving 988 chunks |
| Embeddings | `all-MiniLM-L6-v2`, 384 dimensions, normalized |
| Index | FAISS `IndexFlatIP` (cosine similarity), NumPy fallback |
| Retrieval | top 1 passage, trimmed to about 300 characters |
| Prompt change | extra line: `Inspiration (borrow the mood and style, do not copy the words): ...` |

- **Finding:** in an A/B test, the story without RAG followed the requested characters (a dragon and a baker), but the story with RAG drifted to the retrieved passage's mother and child. The fine-tuned model was never trained with an inspiration field, and the library's old-fashioned language also pulls the style. RAG stays as an optional toggle, off by default.
- **Better version (future work):** fine-tune again with retrieved passages inside the training prompts.

## 7. Findings and limitations

- **Accent:** XTTS-v2 copies timbre well, but pulls English narration toward a standard US or UK accent. Two or more reference clips help.
- **Story quality:** a 1.5B model trained on simple stories writes coherent but simple stories, and some end with odd morals.
- **Training length:** the model saw about 8,000 sequences (under one epoch of the subset), a trade-off for free GPU time.
- **Labels:** the genre and theme labels are keyword-based and rough.
- **Language:** English only.

## 8. Ethics and privacy

- A consent checkbox is required, and the app text asks users to use only voices they have permission to use.
- Uploaded clips are deleted after each narration and are not stored.
- The audio is labelled as AI-generated.
- Voice cloning can be misused, so never clone a voice without permission.

## 9. Troubleshooting (issues met on Colab)

| Problem | Fix |
|---|---|
| `ImportError: isin_mps_friendly` when importing TTS | install `"transformers<5"`, restart the session, then re-run setup cells |
| `No package metadata was found for bitsandbytes` | install `bitsandbytes`, then restart the session |
| `Failed to assign a backend` | free GPU quota is used up, so retry later or use a CPU runtime for the non-GPU steps |
| Warnings about `gradio` or `diffusers` and `huggingface-hub` | harmless unless Gradio fails to import |
| `HF_TOKEN` warning | harmless, a token is optional for public models |
| `NameError` for variables after a restart | re-run the Drive mount and the earlier setup cells (a restart clears memory) |

## 10. How to reproduce

1. Open `01_AI_Story_Narrator_Main.ipynb` in Colab and select a **T4 GPU**.
2. Run the cells in order. Record a clean 15 to 30 second voice clip when asked.
3. For RAG and the improved interface, run `02_RAG_and_UI_Upgrade.ipynb` in the same session afterwards.

## 11. Future work

- Fine-tune with retrieved passages in the prompt so RAG actually helps.
- Train for more steps and on more data for better stories.
- Fine-tune XTTS on a speaker's recordings to match accent more closely.
- Add more languages and a deployed demo on a GPU Space.

## 12. Licenses and credits

Check each license before reuse. XTTS-v2 uses the Coqui Public Model License (non-commercial). Qwen2.5, TinyStories, sentence-transformers, and SpeechBrain have their own licenses on their model and dataset pages. The fairy tales come from Project Gutenberg (public domain).
