"""WER, entity accuracy and paired bootstrap.

All inputs are raw strings; normalization happens here so every system goes
through exactly the same path.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import jiwer
import numpy as np

from .normalize import normalize


@dataclass(frozen=True)
class Counts:
    """Word-level edit counts of one utterance (ref length = hits+sub+del)."""
    sub: int
    dele: int
    ins: int
    ref_len: int

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins


def utt_counts(ref: str, hyp: str) -> Counts:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return Counts(0, 0, len(h.split()), 0)
    if not h:
        n = len(r.split())
        return Counts(0, n, 0, n)
    o = jiwer.process_words(r, h)
    return Counts(o.substitutions, o.deletions, o.insertions, o.hits + o.substitutions + o.deletions)


def corpus_wer(refs: list[str], hyps: list[str]) -> float:
    """Corpus WER = total errors / total reference words (not a mean of per-utterance WERs)."""
    if len(refs) != len(hyps):
        raise ValueError("refs and hyps differ in length")
    cs = [utt_counts(r, h) for r, h in zip(refs, hyps)]
    n = sum(c.ref_len for c in cs)
    if n == 0:
        raise ValueError("empty reference set: WER undefined")
    return sum(c.errors for c in cs) / n


# --- entities ------------------------------------------------------------------

def contains_seq(hay: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    k = len(needle)
    return any(hay[i:i + k] == needle for i in range(len(hay) - k + 1))


def entity_hit(entity_text: str, hyp: str) -> bool:
    """An entity is correct if its normalized form is a contiguous subsequence of the normalized hypothesis."""
    return contains_seq(normalize(hyp).split(), normalize(entity_text).split())


def entity_accuracy(items: list[tuple[list[dict], str]]) -> dict[str, tuple[int, int]]:
    """items: (entities, hyp) per utterance; entity = {"type": str, "text": str}.

    Returns {type: (correct, total)} plus "all".
    """
    acc: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for ents, hyp in items:
        hyp_tokens = normalize(hyp).split()
        for e in ents:
            ok = contains_seq(hyp_tokens, normalize(e["text"]).split())
            for key in (e["type"], "all"):
                acc[key][0] += int(ok)
                acc[key][1] += 1
    return {k: (v[0], v[1]) for k, v in acc.items()}


# --- paired bootstrap ------------------------------------------------------------

@dataclass(frozen=True)
class BootstrapResult:
    wer_a: float
    wer_b: float
    rel_reduction: float  # (A - B) / A, positive = B better
    ci_low: float
    ci_high: float
    n_utts: int
    n_resamples: int


def paired_bootstrap(
    errors_a: np.ndarray, errors_b: np.ndarray, ref_lens: np.ndarray,
    n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05,
) -> BootstrapResult:
    """Paired bootstrap over utterances of the relative WER reduction of B vs A.

    Each resample draws the same utterance indices for both systems, so the
    interval reflects the difference on the same audio, not two independent
    noisy estimates.
    """
    ea, eb, n = (np.asarray(x, dtype=np.int64) for x in (errors_a, errors_b, ref_lens))
    if not (len(ea) == len(eb) == len(n)) or len(n) == 0:
        raise ValueError("arrays must be non-empty and equal length")
    if n.sum() == 0 or ea.sum() == 0:
        raise ValueError("baseline has no errors or no reference words: relative reduction undefined")
    rng = np.random.default_rng(seed)
    rels = np.empty(n_resamples)
    chunk = max(1, 2_000_000 // len(n))
    for start in range(0, n_resamples, chunk):
        stop = min(n_resamples, start + chunk)
        idx = rng.integers(0, len(n), size=(stop - start, len(n)))
        sa, sb = ea[idx].sum(1), eb[idx].sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rels[start:stop] = np.where(sa > 0, (sa - sb) / sa, 0.0)
    wa, wb = ea.sum() / n.sum(), eb.sum() / n.sum()
    lo, hi = np.quantile(rels, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(float(wa), float(wb), float((wa - wb) / wa), float(lo), float(hi), len(n), n_resamples)
