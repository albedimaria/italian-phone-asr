# Costs

Budget cap: **$25** for the whole project. Every paid call is logged here.
Trial or prepaid credits already owned are logged at $0 out-of-pocket, with the credit used shown for transparency.

| Date | Item | Provider | Qty | Unit price | Cost (USD) | Notes |
|---|---|---|---|---|---:|---|
| 2026-11-11 | Price check (Pricing API, free) | Twilio | — | — | 0.00 | IT mobile from US number $0.0476/min; account is Trial with $4.77 credit |
| 2026-11-12 | Smoke tests (3 clips x 3 providers, Scribe v2/v3 probes) | Deepgram, ElevenLabs, OpenAI | ~3 min | — | 0.02 | estimate |
| 2026-11-12 | test_collaudio, 35.2 min | Deepgram nova-3 | 35.2 min | 0.0043/min | 0.15 | |
| 2026-11-12 | test_collaudio, 35.2 min | ElevenLabs scribe_v3 | 35.2 min | 0.22/h | 0.13 | billed as plan credits (Creator) |
| 2026-11-12 | test_collaudio, 35.2 min | OpenAI gpt-transcribe | 35.2 min | 0.0045/min | 0.16 | |
| 2026-11-12 | test_cv, 68.5 min | Deepgram nova-3 | 68.5 min | 0.0043/min | 0.29 | |
| 2026-11-12 | test_cv, 68.5 min | ElevenLabs scribe_v3 | 68.5 min | 0.22/h | 0.25 | billed as plan credits (Creator) |
| 2026-11-12 | test_cv, 68.5 min | OpenAI gpt-transcribe | 68.5 min | 0.0045/min | 0.31 | |

**Total out-of-pocket: $1.31 / $25.00**

## Price references (checked 2026-11-11)
- Twilio Programmable Voice, outbound to Italy mobile from a US number: $0.0476/min (Pricing API v2, `Voice/Countries/IT`).
- Twilio Voice JS SDK client minutes ≈ $0.004/min; recordings ≈ $0.0025/min (to confirm on first invoice line).
- OpenAI: `gpt-4o-transcribe` shuts down on 2027-02-26 (announced 2026-08-26, OpenAI deprecations page); this bench uses the named successor `gpt-transcribe`.
- NVIDIA Parakeet-TDT-0.6B-v3: CC-BY-4.0, runs locally, $0.
- Kaggle GPU: free tier, $0.
