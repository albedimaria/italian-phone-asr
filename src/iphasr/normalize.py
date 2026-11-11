"""Italian text normalizer shared by references and hypotheses.

Every system is scored after the *same* normalization, so the WER measures
recognition errors, not formatting choices (Whisper writes "12.400 €", a
human reference says "dodicimilaquattrocento euro").

Pipeline (order matters, each step consumes its own pattern):
 1. NFKC, lowercase, typographic quotes
 2. currency (€ before/after, cents "12,50 €" -> "dodici euro e cinquanta")
 3. dates "15/03/2026", "15-03" -> "quindici marzo duemilaventisei"
 4. times "15:30", "9.05" -> "quindici e trenta", "nove e cinque"
 5. phone-like digit runs (>= 7 digits, optional spaces/dashes, optional +) -> digit by digit
 6. thousands separators "12.400" -> 12400, decimals "3,5" -> "tre virgola cinque";
    invalid grouping "12000.400" (a real ASR glitch) is read literally as a decimal
 7. percent, ordinals "1°" -> "primo"
 8. letter/digit split "a3" -> "a 3", remaining integers -> words (num2words, it)
 9. punctuation, apostrophes, hyphens -> space; strip diacritics
10. token-level fixes: "dodici mila quattrocento" -> "dodicimilaquattrocento",
    "uno marzo" -> "primo marzo", "e mezza" -> "e trenta", fillers dropped
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from num2words import num2words

MONTHS = [
    "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
    "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
]
FILLERS = {"ehm", "ehmm", "uhm", "uhmm", "mmh", "mhm", "mm", "hmm", "eeh", "ehh"}
DIGIT_WORDS = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"]


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@lru_cache(maxsize=100_000)
def n2w(n: int) -> str:
    """Integer to Italian words, accents stripped ("ventitré" -> "ventitre")."""
    return _strip_accents(num2words(n, lang="it"))


def n2ord(n: int) -> str:
    return _strip_accents(num2words(n, lang="it", to="ordinal"))


def spell_digits(digits: str) -> str:
    return " ".join(DIGIT_WORDS[int(d)] for d in digits if d.isdigit())


@lru_cache(maxsize=1)
def _number_words() -> frozenset[str]:
    """Words num2words produces for 1..999 (used to rejoin "dodici mila quattrocento")."""
    return frozenset(n2w(i) for i in range(1, 1000))


def _int_words(s: str) -> str:
    s = s.replace(".", "")
    if len(s) > 1 and s.startswith("0"):
        return spell_digits(s)
    return n2w(int(s))


# --- regexes -----------------------------------------------------------------
NUM = r"\d{1,3}(?:\.\d{3})+|\d+"  # 12.400 or 12400
_CUR_BEFORE = re.compile(rf"€\s*({NUM})(?:,(\d{{1,2}}))?")
_CUR_AFTER = re.compile(rf"\b({NUM})(?:,(\d{{1,2}}))?\s*(?:€|euro\b)")
_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b")
_TIME = re.compile(r"\b([01]?\d|2[0-4])[:.]([0-5]\d)\b(?!\s*(?:€|euro|%))")
_PHONE = re.compile(r"(?<!\w)(?<!\d[.,])\+?\d(?:[ \-]?\d){6,}(?!\w|[.,]\d)")
_BAD_GROUPING = re.compile(r"\b(\d{4,})\.(\d+)\b")  # "12000.400": not a valid thousands grouping
_THOUSANDS = re.compile(r"\b\d{1,3}(?:\.\d{3})+\b")
_DECIMAL = re.compile(r"\b(\d+),(\d+)\b")
_PERCENT = re.compile(r"(\d+(?:,\d+)?)\s*%")
_ORDINAL = re.compile(r"\b(\d+)\s*[°ºª]")
_ALNUM = re.compile(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])")
_INT = re.compile(r"\d+")
_NONWORD = re.compile(r"[^\w\s]|_")


def _currency(m: re.Match) -> str:
    whole, cents = m.group(1), m.group(2)
    out = f" {_int_words(whole)} euro"
    if cents and int(cents) > 0:
        c = cents if len(cents) == 2 else cents + "0"
        out += f" e {n2w(int(c))}"
    return out + " "


def _date(m: re.Match) -> str:
    d, mo, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return m.group(0)
    out = f" {'primo' if d == 1 else n2w(d)} {MONTHS[mo - 1]}"
    if y:
        yy = int(y) + (2000 if len(y) == 2 else 0)
        out += f" {n2w(yy)}"
    return out + " "


def _time(m: re.Match) -> str:
    h, mi = int(m.group(1)), int(m.group(2))
    return f" {n2w(h) if h else 'zero'}" + (f" e {n2w(mi)}" if mi else "") + " "


def _phone(m: re.Match) -> str:
    raw = m.group(0)
    digits = re.sub(r"\D", "", raw)
    prefix = "piu " if raw.startswith("+") else ""
    return f" {prefix}{spell_digits(digits)} "


def _decimal(m: re.Match) -> str:
    return f" {_int_words(m.group(1))} virgola {_int_words(m.group(2))} "


def _rejoin_thousands(tokens: list[str]) -> list[str]:
    """'dodici mila quattrocento' -> 'dodicimilaquattrocento' (spoken compounds written apart)."""
    nw = _number_words()
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "mila" and out and out[-1] in nw:
            merged = out.pop() + "mila"
            if i + 1 < len(tokens) and tokens[i + 1] in nw:
                merged += tokens[i + 1]
                i += 1
            out.append(merged)
        elif t in nw and out and out[-1].endswith(("mila", "mille")) and out[-1] not in ("mila",):
            out[-1] += t
        else:
            out.append(t)
        i += 1
    return out


def _token_fixes(tokens: list[str]) -> list[str]:
    tokens = [t for t in tokens if t not in FILLERS]
    tokens = _rejoin_thousands(tokens)
    out: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if t in ("uno", "un") and nxt in MONTHS:
            out.append("primo")
        elif t == "e" and nxt == "mezza" and out and out[-1] in _number_words():
            out += ["e", "trenta"]
            i += 1
        elif t == "e" and nxt == "un" and i + 2 < len(tokens) and tokens[i + 2] == "quarto" \
                and out and out[-1] in _number_words():
            out += ["e", "quindici"]
            i += 2
        else:
            out.append(t)
        i += 1
    return out


def normalize(text: str) -> str:
    """Normalize Italian text for WER. Returns space-separated lowercase tokens."""
    s = unicodedata.normalize("NFKC", text).lower()
    s = s.replace("’", "'").replace("‘", "'").replace("`", "'")
    s = s.replace(" ", " ")
    s = _CUR_BEFORE.sub(_currency, s)
    s = _CUR_AFTER.sub(_currency, s)
    s = _DATE.sub(_date, s)
    s = _TIME.sub(_time, s)
    s = _PHONE.sub(_phone, s)
    # read literally as a decimal, so the ASR glitch stays an error
    s = _BAD_GROUPING.sub(lambda m: f" {_int_words(m.group(1))} virgola {_int_words(m.group(2))} ", s)
    s = _THOUSANDS.sub(lambda m: m.group(0).replace(".", ""), s)
    s = _PERCENT.sub(lambda m: f" {m.group(1)} per cento ", s)
    s = _DECIMAL.sub(_decimal, s)
    s = _ORDINAL.sub(lambda m: f" {n2ord(int(m.group(1)))} ", s)
    s = _ALNUM.sub(" ", s)
    s = _INT.sub(lambda m: f" {_int_words(m.group(0))} ", s)
    s = _NONWORD.sub(" ", s)
    s = _strip_accents(s)
    return " ".join(_token_fixes(s.split()))
