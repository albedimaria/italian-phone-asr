# italian-phone-asr

Fine-tuning Whisper-large-v3-turbo (encoder LoRA) for **Italian phone speech** (8 kHz, G.711 μ-law), with a
focus on the entities a phone agent must get right: prices, dates, times, car models, names, phone numbers.
The fine-tuned model is benchmarked against Deepgram Nova-3, ElevenLabs Scribe v2, OpenAI `gpt-transcribe`,
NVIDIA Parakeet-TDT-0.6B-v3 and base Whisper-large-v3-turbo, on real human calls held out from training.

> Status: **work in progress** — metrics, normalizer and recording pipeline are done; data collection and
> training are under way. Results will appear in [`results/table.md`](results/table.md).

## Why
Phone ASR fails on exactly the words that matter: a car-dealer agent heard *"dodicimilaquattrocento"* and
the ASR wrote `12000.400`. This project measures those errors with a shared normalizer and an entity
metric, then tries to fix them with a small, cheap fine-tune.

## Method
- **Audio chain**: every training clip goes 8 kHz → G.711 μ-law encode/decode (ffmpeg) → 16 kHz for Whisper.
- **Training data**: a Common Voice Italian subset plus entity-dense synthetic sentences (open TTS voices),
  kept only if Parakeet transcribes them back within 10% WER.
- **Model**: LoRA (r 32) on the attention and feed-forward layers of the encoder only. The problem is
  acoustic; adapting the decoder too (Whisper-small rounds 1–3) drifted the spelling of common words.
- **Training**: Kaggle, two T4s in DataParallel, batch 4 per GPU × 2 GPUs × 4 accumulation steps =
  effective batch 32, fp16, lr 2e-4, 3 epochs (1,116 steps); checkpoint chosen by WER on the dev set only.
  Round 4 stopped on Kaggle's time limit at step 600 (1.6 epochs); round 5 resumes it from that checkpoint
  with optimizer and scheduler state. The data ablation (Common Voice only / synthetic only / both) was
  not run: at ~55 s per step each arm takes about 17 GPU hours, and three arms exceed Kaggle's 30 h weekly quota.
- **Decoding**: the same for base and fine-tuned, with Whisper's guards against repetition loops
  (temperature fallback, compression-ratio and log-prob thresholds). Without them the base model looped
  "Grazie a tutti." for 267 words on one clip, and that single clip moved its WER from 6.4% to 10.7%.
- **Test data**
  - `test_human`: 5–8 volunteers read a 108-line script over a real Twilio call (browser, codec forced to
    PCMU). Never used for training or model selection, never published. Every take is checked by
    listening (`bench record review`): the reference is what was said, not what was shown.
  - `test_cv` and `test_cv_g711`: 661 clips of the Common Voice Italian test split, clean and through the
    G.711 chain. **Used to choose between training rounds**, so they are not clean held-out sets: the
    unbiased number is `test_human`.
  - `test_collaudio`: caller channel of real phone calls with TTS voices.
  - `test_synth_entities` (in progress): 200 held-out entity sentences (templates, car models and names never
    used in training) read by a TTS voice that is not in the training set. Synthetic, declared as such.
    The TTS can skip or misread a word, so its references go through the same listening check
    (`bench record review --set test_synth_entities`), worst clips first by the highest WER of any system.
- **Metrics**
  - WER after one Italian normalizer for reference and hypothesis (numbers to words with `num2words`,
    prices, times, dates, phone numbers digit by digit). Unit-tested, because this is where systems that
    write digits and references in words would otherwise diverge.
  - Entity accuracy by type: an entity is correct if its normalized form appears contiguously in the
    normalized hypothesis.
  - Relative WER reduction with a 95% CI from a paired bootstrap (10,000 resamples). On `test_human`
    every speaker reads the same script, so the bootstrap resamples speakers and sentences together:
    resampling utterances alone would give too narrow an interval.
- **Leak guard**: test-script entity values and phrasing never appear in the synthetic training set
  (`tests/test_leak.py`).

## Reproduce
```bash
uv sync
uv run pytest                 # normalizer, metrics, bootstrap, leak guard, report gate
uv run bench report           # rebuild results/table.md from cached transcripts (no network)
```
`bench report` exits non-zero unless `test_human` has ≥ 5 speakers and ≥ 40 minutes and all six systems
have complete transcripts on every test set: an empty benchmark is not a passing one.

Audio and transcripts live outside the repo, under `IPASR_DATA` (see `src/iphasr/paths.py`).

## What the Whisper-small rounds taught
Rounds 1–3 fine-tuned Whisper-small with LoRA on encoder and decoder. Common Voice improved (13.5% →
11.3% WER in round 1), but the phone set got worse: the model wrote "bongiorno" for "buongiorno", the most
frequent word on a phone line, because the synthetic sentences had no greetings. Adding phone openers fixed
the word but not the trade-off, and on G.711 audio round 3 was within noise of the base (−3.8%, CI
[−2.6%, +9.7%]). The next rounds moved to a stronger base (turbo) and adapted the encoder only.

## Costs
Every paid call is logged in [`COSTS.md`](COSTS.md), with a $25 cap for the whole project.

## License
Code: MIT. Model weights and card: see the Hugging Face model page (to be published).
