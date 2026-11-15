# Results

> **Interim.** The gate below is not met yet: empty cells are sets or systems still to run.

WER (%) after the shared Italian normalizer (`src/iphasr/normalize.py`). Lower is better.

| System | test_human (0 utts, 0 speakers, 0 min) | test_cv (661 utts) | test_collaudio (132 utts) |
|---|---:|---:|---:|
| Deepgram Nova-3 | — | 6.5 | 1.6 |
| ElevenLabs Scribe v3 | — | 1.8 | 0.3 |
| OpenAI gpt-transcribe | — | 3.1 | 0.2 |
| NVIDIA Parakeet-TDT-0.6B-v3 | — | 3.3 | 0.6 |
| Whisper-small (base) | — | 13.5 | 1.1 |
| Whisper-small + LoRA (ours) | — | 12.7 | 1.3 |

## Speed of the local systems (CPU)

Real-time factor = processing time / audio duration, batch transcription on a laptop CPU, best full run. Below 1 is faster than real time.

| System | RTF |
|---|---:|
| NVIDIA Parakeet-TDT-0.6B-v3 | 0.16 |
| Whisper-small, base and LoRA (same architecture) | 0.26 |

## Entity accuracy on test_human (%)

An entity is correct when its normalized form appears contiguously in the normalized hypothesis.

| System | prezzo | data | orario | auto | nome | telefono | numero | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deepgram Nova-3 | — | — | — | — | — | — | — | — |
| ElevenLabs Scribe v3 | — | — | — | — | — | — | — | — |
| OpenAI gpt-transcribe | — | — | — | — | — | — | — | — |
| NVIDIA Parakeet-TDT-0.6B-v3 | — | — | — | — | — | — | — | — |
| Whisper-small (base) | — | — | — | — | — | — | — | — |
| Whisper-small + LoRA (ours) | — | — | — | — | — | — | — | — |

## Fine-tuned vs base on test_human

- not available (missing transcripts)

## Slices of test_human (WER %)

- not available

## Gate: FAILED

- test_human has 0 speakers (< 5)
- test_human has 0.0 min (< 40)
- test_human: manifest missing or empty (D:\data\italian-phone-asr\manifests\test_human.jsonl)
- deepgram-nova-3 on test_human: all transcripts missing
- elevenlabs-scribe on test_human: all transcripts missing
- openai-gpt-transcribe on test_human: all transcripts missing
- parakeet-tdt-0.6b-v3 on test_human: all transcripts missing
- whisper-small on test_human: all transcripts missing
- whisper-small-it-phone on test_human: all transcripts missing
