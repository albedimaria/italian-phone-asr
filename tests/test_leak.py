"""Train/test leak guard: test-script entity values and phrasing never appear in train_synth."""

from iphasr.normalize import DIGIT_WORDS as _DW, normalize
from iphasr.script import load_script
from iphasr.synth import CARS, FIRST, LAST, generate

N_GRAM = 6
DIGIT_WORDS = set(_DW)


def _ngrams(tokens, n):
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _contains(tokens, needle):
    k = len(needle)
    return any(tokens[i:i + k] == needle for i in range(len(tokens) - k + 1))


def test_script_parses_and_entities_are_substrings():
    lines = load_script()
    assert 100 <= len(lines) <= 130
    for line in lines:
        for e in line.entities:
            assert e["text"] in line.text
            assert normalize(e["text"]), e


def test_pools_disjoint_from_script():
    vocab = set(" ".join(normalize(l.text) for l in load_script()).split())
    for name in FIRST + LAST:
        for tok in normalize(name).split():
            assert tok not in vocab, f"synth name token {tok!r} appears in the test script"
    test_cars = {normalize(e["text"]) for l in load_script() for e in l.entities if e["type"] == "auto"}
    assert not test_cars & {normalize(c) for c in CARS}


def test_generated_corpus_has_no_test_entities_or_phrasing():
    synth = generate(4000, seed=0)
    synth_tokens = [normalize(s["text"]).split() for s in synth]
    synth_vocab = {t for toks in synth_tokens for t in toks}
    synth_grams: set = set()
    for toks in synth_tokens:
        synth_grams |= _ngrams(toks, N_GRAM)
    joined = " | ".join(" ".join(t) for t in synth_tokens)  # fast prefilter

    for line in load_script():
        for e in line.entities:
            ent = normalize(e["text"]).split()
            if e["type"] in ("nome", "auto") and len(ent) == 1:
                assert ent[0] not in synth_vocab, e
            if len(ent) >= 2 and e["type"] != "numero" and " ".join(ent) in joined:
                assert not any(_contains(t, ent) for t in synth_tokens), f"test entity leaked into train_synth: {e}"
        # digit-by-digit runs are random phone numbers, checked whole above, not phrasing
        shared = {g for g in _ngrams(normalize(line.text).split(), N_GRAM) & synth_grams
                  if not set(g) <= DIGIT_WORDS}
        assert not shared, f"test phrasing leaked ({line.id}): {shared}"


def test_synth_written_and_spoken_agree():
    for s in generate(500, seed=3):
        assert normalize(s["text"]) == normalize(s["tts_text"]), s
