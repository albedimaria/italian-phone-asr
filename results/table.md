# Results

> **Interim.** The gate below is not met yet: empty cells are sets or systems still to run.

WER (%) after the shared Italian normalizer (`src/iphasr/normalize.py`). Lower is better.

| System | test_human (216 utts, 2 speakers, 18 min) | test_cv (661 utts) | test_cv_g711 (661 utts) | test_collaudio (132 utts) |
|---|---:|---:|---:|---:|
| Deepgram Nova-3 | — | 6.5 | 7.7 | 1.6 |
| ElevenLabs Scribe v3 | — | 1.8 | 2.3 | 0.3 |
| OpenAI gpt-transcribe | — | 3.1 | 3.6 | 0.2 |
| NVIDIA Parakeet-TDT-0.6B-v3 | — | 3.3 | 4.0 | 0.6 |
| Whisper-large-v3-turbo (base) | — | 6.1 | 6.9 | — |
| Whisper-large-v3-turbo + encoder LoRA (ours) | — | 4.7 | 5.5 | — |

## Speed of the local systems (CPU)

Real-time factor = processing time / audio duration, batch transcription on a laptop CPU, best full run. Below 1 is faster than real time.

| System | RTF |
|---|---:|
| NVIDIA Parakeet-TDT-0.6B-v3 | 0.11 |
| Whisper-large-v3-turbo, base and LoRA (same architecture) | 2.19 |

## Entity accuracy on test_human (%)

An entity is correct when its normalized form appears contiguously in the normalized hypothesis.

| System | prezzo | data | orario | auto | nome | telefono | numero | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deepgram Nova-3 | — | — | — | — | — | — | — | — |
| ElevenLabs Scribe v3 | — | — | — | — | — | — | — | — |
| OpenAI gpt-transcribe | — | — | — | — | — | — | — | — |
| NVIDIA Parakeet-TDT-0.6B-v3 | — | — | — | — | — | — | — | — |
| Whisper-large-v3-turbo (base) | — | — | — | — | — | — | — | — |
| Whisper-large-v3-turbo + encoder LoRA (ours) | — | — | — | — | — | — | — | — |

## Fine-tuned vs base on test_human

- not available (missing transcripts)

## Slices of test_human (WER %)

| System | channel: browser | speaker spk01 | speaker spk02 | uvular /r/ speaker (n=1) |
|---|---:|---:|---:|---:|

## Gate: FAILED

- test_human has 2 speakers (< 5)
- test_human has 18.0 min (< 40)
- deepgram-nova-3 on test_human: all transcripts missing
- elevenlabs-scribe on test_human: all transcripts missing
- openai-gpt-transcribe on test_human: all transcripts missing
- parakeet-tdt-0.6b-v3 on test_human: all transcripts missing
- whisper-large-v3-turbo on test_human: all transcripts missing
- whisper-large-v3-turbo on test_collaudio: all transcripts missing
- whisper-turbo-it-phone on test_human: all transcripts missing
- whisper-turbo-it-phone on test_collaudio: all transcripts missing
