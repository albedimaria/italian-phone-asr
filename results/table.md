# Results

WER (%) after the shared Italian normalizer (`src/iphasr/normalize.py`). Lower is better.

| System | test_human (0 utts, 0 speakers, 0 min) | test_cv (0 utts) | test_collaudio (0 utts) |
|---|---:|---:|---:|
| Deepgram Nova-3 | — | — | — |
| ElevenLabs Scribe v2 | — | — | — |
| OpenAI gpt-4o-transcribe | — | — | — |
| NVIDIA Parakeet-TDT-0.6B-v3 | — | — | — |
| Whisper-small (base) | — | — | — |
| Whisper-small + LoRA (ours) | — | — | — |

## Entity accuracy on test_human (%)

An entity is correct when its normalized form appears contiguously in the normalized hypothesis.

| System | prezzo | data | orario | auto | nome | telefono | numero | all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Deepgram Nova-3 | — | — | — | — | — | — | — | — |
| ElevenLabs Scribe v2 | — | — | — | — | — | — | — | — |
| OpenAI gpt-4o-transcribe | — | — | — | — | — | — | — | — |
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
- test_cv: manifest missing or empty (D:\data\italian-phone-asr\manifests\test_cv.jsonl)
- test_collaudio: manifest missing or empty (D:\data\italian-phone-asr\manifests\test_collaudio.jsonl)
- deepgram-nova-3 on test_human: all transcripts missing
- deepgram-nova-3 on test_cv: all transcripts missing
- deepgram-nova-3 on test_collaudio: all transcripts missing
- elevenlabs-scribe-v2 on test_human: all transcripts missing
- elevenlabs-scribe-v2 on test_cv: all transcripts missing
- elevenlabs-scribe-v2 on test_collaudio: all transcripts missing
- openai-transcribe on test_human: all transcripts missing
- openai-transcribe on test_cv: all transcripts missing
- openai-transcribe on test_collaudio: all transcripts missing
- parakeet-tdt-0.6b-v3 on test_human: all transcripts missing
- parakeet-tdt-0.6b-v3 on test_cv: all transcripts missing
- parakeet-tdt-0.6b-v3 on test_collaudio: all transcripts missing
- whisper-small on test_human: all transcripts missing
- whisper-small on test_cv: all transcripts missing
- whisper-small on test_collaudio: all transcripts missing
- whisper-small-it-phone on test_human: all transcripts missing
- whisper-small-it-phone on test_cv: all transcripts missing
- whisper-small-it-phone on test_collaudio: all transcripts missing
