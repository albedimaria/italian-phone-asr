import numpy as np
import pytest

from iphasr.metrics import (corpus_wer, entity_accuracy, entity_hit, paired_bootstrap, paired_bootstrap_2way,
                            utt_counts)


def test_wer_known_cases():
    assert corpus_wer(["a b c d"], ["a b c d"]) == 0.0
    assert corpus_wer(["a b c d"], ["a x c d"]) == 0.25          # 1 sub
    assert corpus_wer(["a b c d"], ["a c d"]) == 0.25            # 1 del
    assert corpus_wer(["a b c d"], ["a b c d e"]) == 0.25        # 1 ins
    assert corpus_wer(["a b c d"], [""]) == 1.0


def test_corpus_wer_is_pooled_not_averaged():
    # utt1: 1 error / 1 word, utt2: 0 / 9 -> pooled 1/10, a mean of WERs would be 0.5
    assert corpus_wer(["a", "b c d e f g h i j"], ["x", "b c d e f g h i j"]) == pytest.approx(0.1)


def test_wer_after_normalization():
    # formatting differences are not errors
    assert corpus_wer(["costa dodicimilaquattrocento euro"], ["Costa 12.400 €."]) == 0.0
    c = utt_counts("alle quindici e trenta", "alle 15")
    assert (c.dele, c.ref_len) == (2, 4)


def test_corpus_wer_rejects_empty():
    with pytest.raises(ValueError):
        corpus_wer([""], ["x"])


def test_entity_hit():
    assert entity_hit("14.900 euro", "una Fiat Tipo a quattordicimila novecento euro")
    assert not entity_hit("14.900 euro", "una Fiat Tipo a 14.000 euro")
    assert entity_hit("347 281 9063", "il numero è 3472819063")
    assert not entity_hit("347 281 9063", "il numero è 347 281 906")
    assert entity_hit("Peugeot 208", "test drive della peugeot duecentotto")
    assert not entity_hit("Peugeot 208", "test drive della peugeot 2008")


def test_entity_accuracy_per_type():
    items = [
        ([{"type": "prezzo", "text": "12.400 euro"}, {"type": "data", "text": "18 dicembre"}], "12.400 € il 18 dicembre"),
        ([{"type": "prezzo", "text": "60 euro"}], "sessanta"),
    ]
    acc = entity_accuracy(items)
    assert acc["prezzo"] == (1, 2)
    assert acc["data"] == (1, 1)
    assert acc["all"] == (2, 3)


def test_bootstrap_identical_systems():
    e = np.array([1, 0, 2, 1, 0, 3])
    n = np.array([5, 4, 6, 5, 3, 7])
    r = paired_bootstrap(e, e, n, n_resamples=2000)
    assert r.rel_reduction == 0.0 and r.ci_low == 0.0 and r.ci_high == 0.0


def test_bootstrap_clear_improvement_ci_above_zero():
    rng = np.random.default_rng(1)
    n = rng.integers(5, 15, size=400)
    ea = rng.binomial(n, 0.20)
    eb = rng.binomial(n, 0.12)
    r = paired_bootstrap(ea, eb, n, n_resamples=10_000)
    assert r.wer_a == pytest.approx(ea.sum() / n.sum())
    assert 0.25 < r.rel_reduction < 0.55
    assert r.ci_low > 0 and r.ci_low < r.rel_reduction < r.ci_high


def test_bootstrap_noise_ci_spans_zero():
    rng = np.random.default_rng(2)
    n = rng.integers(5, 15, size=60)
    ea = rng.binomial(n, 0.2)
    eb = rng.binomial(n, 0.2)
    r = paired_bootstrap(ea, eb, n, n_resamples=5000)
    assert r.ci_low < 0 < r.ci_high


def test_bootstrap_is_deterministic():
    e1, e2, n = np.array([2, 1, 3]), np.array([1, 1, 2]), np.array([6, 5, 7])
    assert paired_bootstrap(e1, e2, n, seed=7) == paired_bootstrap(e1, e2, n, seed=7)


def _crossed(n_spk=5, n_sen=100, seed=0):
    """Every speaker reads every sentence; B is better than A only for speaker 0."""
    rng = np.random.default_rng(seed)
    spk, sen, ea, eb, n = [], [], [], [], []
    for s in range(n_spk):
        for t in range(n_sen):
            a = int(rng.integers(0, 4))
            spk.append(f"s{s}"); sen.append(f"t{t}"); n.append(10)
            ea.append(a); eb.append(0 if s == 0 else a)
    return np.array(ea), np.array(eb), np.array(n), spk, sen


def test_bootstrap_2way_identical_systems_is_zero():
    ea, _, n, spk, sen = _crossed()
    r = paired_bootstrap_2way(ea, ea, n, spk, sen, n_resamples=2000)
    assert r.rel_reduction == 0 and r.ci_low == 0 and r.ci_high == 0


def test_bootstrap_2way_is_wider_when_one_speaker_drives_the_gain():
    ea, eb, n, spk, sen = _crossed()
    one = paired_bootstrap(ea, eb, n, n_resamples=5000)
    two = paired_bootstrap_2way(ea, eb, n, spk, sen, n_resamples=5000)
    assert abs(one.rel_reduction - two.rel_reduction) < 1e-12
    assert one.ci_low > 0                               # utterance bootstrap: "significant"
    assert two.ci_low <= 0 < two.ci_high                # speaker resampling: the gain may be one speaker
    assert (two.ci_high - two.ci_low) > 2 * (one.ci_high - one.ci_low)


def test_bootstrap_2way_rejects_mismatched_lengths():
    ea, eb, n, spk, sen = _crossed()
    with pytest.raises(ValueError):
        paired_bootstrap_2way(ea, eb, n, spk[:-1], sen)
