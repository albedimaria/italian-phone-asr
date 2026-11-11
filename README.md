# italian-phone-asr

Fine-tuning Whisper-small (LoRA) for **Italian phone speech** (8 kHz, G.711 μ-law), with a focus on the
entities a phone agent must get right: prices, dates, times, car models, names, phone numbers.
The fine-tuned model is benchmarked against Deepgram Nova-3, ElevenLabs Scribe v2, OpenAI `gpt-transcribe`,
NVIDIA Parakeet-TDT-0.6B-v3 and base Whisper-small, on real human calls held out from training.

> Status: **work in progress** — metrics, normalizer and recording pipeline are done; data collection and
> training are under way. Results will appear in [`results/table.md`](results/table.md).

## Why
Phone ASR fails on exactly the words that matter: a car-dealer agent heard *"dodicimilaquattrocento"* and
the ASR wrote `12000.400`. This project measures those errors with a shared normalizer and an entity
metric, then tries to fix them with a small, cheap fine-tune.

## Method
- **Audio chain**: every training clip goes 8 kHz → G.711 μ-law encode/decode (ffmpeg) → 16 kHz for Whisper.
- **Training data**: a Common Voice Italian subset plus entity-dense synthetic sentences (open TTS voices).
- **Test data**
  - `test_human`: 5–8 volunteers read a 108-line script over a real Twilio call (browser, codec forced to
    PCMU). Never used for training, never published.
  - `test_cv`: Common Voice Italian official test split (forgetting check).
  - `test_collaudio`: caller channel of real phone calls with TTS voices.
- **Metrics**
  - WER after one Italian normalizer for reference and hypothesis (numbers to words with `num2words`,
    prices, times, dates, phone numbers digit by digit). Unit-tested, because this is where systems that
    write digits and references in words would otherwise diverge.
  - Entity accuracy by type: an entity is correct if its normalized form appears contiguously in the
    normalized hypothesis.
  - Paired bootstrap over utterances (10,000 resamples) for the relative WER reduction, with a 95% CI.
- **Leak guard**: test-script entity values and phrasing never appear in the synthetic training set
  (`tests/test_leak.py`).

## Reproduce
```bash
uv sync
uv run pytest                 # normalizer, metrics, bootstrap, leak guard, report gate
uv run bench report           # rebuild results/table.md from cached transcripts (no network)
```
`bench report` exits non-zero unless `test_human` has ≥ 5 speakers and ≥ 40 minutes and all six systems
have complete transcripts on all three test sets: an empty benchmark is not a passing one.

Audio and transcripts live outside the repo, under `IPASR_DATA` (see `src/iphasr/paths.py`).

## Costs
Every paid call is logged in [`COSTS.md`](COSTS.md), with a $25 cap for the whole project.

## License
Code: MIT. Model weights and card: see the Hugging Face model page (to be published).
