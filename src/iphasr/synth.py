"""Template generator for train_synth: entity-dense Italian phone sentences.

Each item has:
- text: written label for Whisper (digits, like Whisper's own style: "12.400 euro")
- tts_text: spoken form fed to the TTS ("dodicimilaquattrocento euro")
- entities: [{"type", "text"}] in written form

Anti-leak: car models and names come from pools disjoint from the test script;
random prices, dates, times and phone numbers skip any value that appears in
the test script. tests/test_leak.py checks the generated corpus independently.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache

from .normalize import MONTHS, normalize
from .script import load_script

CARS = [
    "Fiat Panda", "Fiat Punto", "Fiat 600", "Peugeot 2008", "Peugeot 3008", "Renault Captur",
    "Renault Megane", "Volkswagen T-Roc", "Volkswagen Tiguan", "Toyota Corolla", "Toyota C-HR",
    "Opel Mokka", "Opel Astra", "Ford Fiesta", "Ford Focus", "Ford Kuga", "Dacia Duster",
    "Dacia Spring", "Jeep Compass", "Jeep Avenger", "Citroën C4", "Kia Sportage", "Kia Stonic",
    "Hyundai Tucson", "Hyundai Kona", "Nissan Juke", "Mazda 2", "Skoda Fabia", "Skoda Kamiq",
    "Seat Ibiza", "Cupra Formentor", "Suzuki Swift", "Suzuki Vitara", "Audi A1", "Audi Q3",
    "Mercedes Classe A", "Lancia Delta", "Alfa Romeo Tonale", "Mini Cooper", "Smart Fortwo",
    "Tesla Model 3", "MG ZS", "BYD Atto 3", "Volvo XC40", "Honda Jazz",
]
FIRST = [
    "Marta", "Giorgio", "Valentina", "Simone", "Federica", "Lorenzo", "Silvia", "Nicola", "Alice",
    "Tommaso", "Beatrice", "Emanuele", "Irene", "Fabio", "Martina", "Riccardo", "Laura", "Ettore",
    "Anna", "Giovanni", "Elisa", "Pietro", "Camilla", "Daniele", "Noemi", "Gabriele", "Sofia",
    "Michele", "Ilaria", "Filippo", "Greta", "Vittorio", "Arianna", "Enrico", "Serena", "Claudio",
]
LAST = [
    "Bellini", "Ferri", "Greco", "Ricci", "Conti", "Barbieri", "Mancini", "Rizzo", "Gentile", "Villa",
    "Serra", "Caruso", "Vitale", "Longo", "Mazza", "Pellegrini", "Fabbri", "Testa", "Grasso",
    "Marini", "Ferrara", "Sala", "Basile", "Riva", "Farina", "Monti", "Cattaneo", "Benedetti",
    "Orlando", "Leone", "Bruno", "Galli", "Silvestri", "Sartori", "Morelli", "Neri",
]
WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato"]

# Phrasing deliberately different from the test script (6-gram overlap is tested).
TEMPLATES = [
    "Salve, chiamo per la {auto} che avete pubblicato sul sito.",
    "Quella {auto} a {prezzo} ha già fatto la revisione?",
    "Mi segni pure per {weekday} {data} alle {orario}.",
    "A nome di {nome}, grazie.",
    "Il numero giusto per ricontattarmi è {telefono}.",
    "Sulla {auto} del {anno} quanto mi fate di valutazione?",
    "Il contachilometri segna {km} chilometri, più o meno.",
    "Per la permuta mi avevano offerto {prezzo}, è ancora valido?",
    "Con un anticipo di {prezzo} quanto viene la rata?",
    "Facciamo {data}, la mattina presto se si può.",
    "Arrivo per le {orario}, al massimo dieci minuti dopo.",
    "Le lascio il recapito: {telefono}.",
    "Il signor {nome} ha già firmato il contratto?",
    "Volevo un'informazione sulla {auto} grigia scura.",
    "Il saldo finale sarebbe {prezzo}, mi conferma?",
    "La {auto} me la potete consegnare entro il {data}?",
    "Ho parlato con {nome} la settimana scorsa.",
    "Per me va bene {weekday} alle {orario}.",
    "Alla fine quanto costa la {auto} con tutti gli optional?",
    "Il cliente {nome} ha disdetto l'incontro del {data}.",
    "Il meccanico ha detto {prezzo} per la frizione.",
    "Dovrei ritirarla il {data} verso le {orario}.",
    "Ha percorso {km} chilometri ed è del {anno}.",
    "Il preventivo per la {auto} arriva a {prezzo} chiavi in mano.",
    "Se mi richiamate al {telefono} vi rispondo subito.",
    "Sono {nome} e vorrei un'informazione.",
    "L'officina riapre il {data} alle {orario}.",
    "Con l'incentivo la {auto} scende a {prezzo}?",
    "Per il noleggio a lungo termine pagherei {prezzo} al mese.",
    "Ho prenotato il controllo per il {data}.",
    "La segreteria è aperta fino alle {orario}?",
    "Mi faccia sapere se la {auto} è a benzina o diesel.",
    "Il saldo lo verso il {data} con un bonifico.",
    "Il cellulare di {nome} è {telefono}.",
    "Quanti posti ha la {auto} in versione familiare?",
    "La {auto} usata del {anno} la trovo a {prezzo}?",
    "Ci sentiamo {weekday} {data} dopo le {orario}.",
    "Il ricambio è arrivato, sono {prezzo} in tutto.",
    "Va bene, passo io il {data} a firmare.",
    "Ricordo che l'accordo era di {prezzo} più IVA.",
]
_SLOT = re.compile(r"\{(\w+)\}")


@lru_cache(maxsize=1)
def test_blocklist() -> frozenset[str]:
    """Normalized forms of every annotated entity in the test script."""
    return frozenset(normalize(e["text"]) for line in load_script() for e in line.entities)


def _price(rng: random.Random) -> str:
    kind = rng.random()
    if kind < 0.45:
        v = rng.randrange(60, 460) * 100           # 6.000 .. 45.900
    elif kind < 0.85:
        v = rng.randrange(20, 1500)
    else:
        return f"{rng.randrange(10, 400)},{rng.choice(['50', '90', '99', '40', '20', '80'])} euro"
    return f"{v:,}".replace(",", ".") + " euro"


def _date(rng: random.Random) -> str:
    return f"{rng.randrange(1, 29)} {rng.choice(MONTHS)}"


def _time(rng: random.Random) -> str:
    h = rng.randrange(7, 21)
    m = rng.choice([0, 5, 10, 20, 25, 35, 40, 50, 55])
    return f"{h}:{m:02d}"


def _phone(rng: random.Random) -> str:
    p = rng.choice(["320", "331", "333", "335", "338", "340", "345", "348", "349", "366", "370", "392"])
    return f"{p} {rng.randrange(100, 1000)} {rng.randrange(1000, 10000)}"


def _fill(slot: str, rng: random.Random) -> tuple[str, str | None]:
    """Return (written value, entity type or None)."""
    if slot == "auto":
        return rng.choice(CARS), "auto"
    if slot == "nome":
        return (f"{rng.choice(FIRST)} {rng.choice(LAST)}" if rng.random() < 0.8 else rng.choice(LAST)), "nome"
    if slot == "prezzo":
        return _price(rng), "prezzo"
    if slot == "data":
        return _date(rng), "data"
    if slot == "orario":
        return _time(rng), "orario"
    if slot == "telefono":
        return _phone(rng), "telefono"
    if slot == "km":
        return f"{rng.randrange(5, 220) * 1000:,}".replace(",", "."), "numero"
    if slot == "anno":
        return str(rng.choice([2014, 2015, 2016, 2017, 2020, 2022, 2023, 2024])), "numero"
    if slot == "weekday":
        return rng.choice(WEEKDAYS), None
    raise KeyError(slot)


def generate(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    block = test_blocklist()
    out = []
    while len(out) < n:
        tpl = rng.choice(TEMPLATES)
        written, spoken, ents = tpl, tpl, []
        for slot in _SLOT.findall(tpl):
            for _ in range(50):
                val, typ = _fill(slot, rng)
                nv = normalize(val)
                padded = f" {nv} "
                # multi-token test values must not appear inside a synth value; single tokens only exactly
                if typ is None or (nv not in block and not any(" " in b and f" {b} " in padded for b in block)):
                    break
            else:
                raise RuntimeError(f"cannot fill {slot} outside the test blocklist")
            written = written.replace("{" + slot + "}", val, 1)
            spoken = spoken.replace("{" + slot + "}", normalize(val) if typ else val, 1)
            if typ:
                ents.append({"type": typ, "text": val})
        out.append({"id": f"synth{len(out):05d}", "text": written, "tts_text": spoken, "entities": ents})
    return out
