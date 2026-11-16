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
# --- test_synth_entities (held out): templates, cars and names never used in training --------
# A synthetic entity test set read by a TTS voice that is not in train_synth either (Chatterbox).
# Phrasing, car models and names are disjoint from TEMPLATES/CARS/FIRST/LAST and from the test
# script; tests/test_leak.py checks it.
HELDOUT_CARS = [
    "Fiat Doblò", "Peugeot 308", "Renault Austral", "Renault Twingo", "Toyota RAV4", "Toyota Aygo X",
    "Opel Grandland", "Ford Ranger", "Kia Niro", "Kia Ceed", "Hyundai Bayon", "Nissan Micra",
    "Skoda Karoq", "Seat Arona", "Seat Leon", "Suzuki Ignis", "Audi A3", "BMW Serie 1", "Mercedes GLA",
    "Alfa Romeo Giulia", "Volkswagen Taigo", "Mazda MX-5", "Jeep Wrangler", "Volvo XC60",
    "Citroën C5 Aircross", "Subaru Forester", "Mitsubishi Space Star", "Lexus UX",
]
HELDOUT_FIRST = [
    "Roberta", "Veronica", "Cristina", "Antonio", "Ludovica", "Edoardo", "Rebecca", "Leonardo",
    "Carlotta", "Jacopo", "Margherita", "Samuele", "Viola", "Mattia", "Ginevra", "Raffaele",
]
HELDOUT_LAST = [
    "Santoro", "Giordano", "Lombardi", "Ferraro", "Palumbo", "Valentini", "Guerra", "Bernardi",
    "Caputo", "Messina", "Coppola", "Mariani", "De Santis", "Parisi", "Sorrentino", "Donati",
]
HELDOUT_TEMPLATES = [
    "Avrei visto in vetrina una {auto}, posso venire a provarla?",
    "Mi ha passato il suo contatto {nome}, è lei che segue l'usato?",
    "Me la lascia a {prezzo} se pago in contanti?",
    "Va bene {weekday} {data}, mi raccomando puntuali.",
    "Riesco a esserci verso le {orario}, non prima.",
    "Annoti il mio cellulare, {telefono}, mi chiami pure quando vuole.",
    "La macchina vecchia è una {auto} del {anno} con {km} chilometri.",
    "Per il tagliando mi avete chiesto {prezzo}, mi pare tanto.",
    "Posso spostare l'appuntamento del {data} alle {orario}?",
    "Chieda pure di {nome}, mi trova in amministrazione.",
    "Sulla {auto} il finanziamento parte da {prezzo} al mese?",
    "Il numero fisso dell'ufficio è {telefono}.",
    "L'ho comprata nel {anno} e ha fatto {km} chilometri.",
    "Il carro attrezzi passa il {data} tra le {orario} e le {orario}.",
    "Con la rottamazione la {auto} quanto verrebbe?",
    "Si tratta della pratica a nome {nome}, quella del {data}.",
    "Ho già versato una caparra di {prezzo}.",
    "Mi richiami dopo le {orario} al {telefono}, per favore.",
    "Cercavo una {auto} usata sotto i {prezzo}.",
    "Il ritiro è confermato per {weekday} alle {orario}.",
]
HELDOUT_POOLS = {"auto": HELDOUT_CARS, "first": HELDOUT_FIRST, "last": HELDOUT_LAST}

_SLOT = re.compile(r"\{(\w+)\}")

# Round 3: phone openers/closers. Round 1-2 had none, and the fine-tuned model lost Whisper's
# spelling of the most frequent phone word ("buongiorno" -> "bongiorno"/"bun giorno").
OPENERS = [
    "Buongiorno,", "Buongiorno,", "Buongiorno,", "Sì, buongiorno,", "Pronto, buongiorno,",
    "Buongiorno a lei,", "Buonasera,", "Salve,", "Pronto,", "Ah, buongiorno,",
]
CLOSERS = [
    "Grazie, buongiorno.", "Grazie, buona giornata.", "Va bene, arrivederci.", "La ringrazio, arrivederci.",
    "D'accordo, buonasera.", "Ok, grazie a lei.", "Benissimo, buona giornata.", "Grazie ancora, arrivederci.",
]
WRAP_OPEN_P, WRAP_CLOSE_P = 0.35, 0.25


def _wrap(item: dict, rng: random.Random) -> dict:
    """Add a phone opener and/or closer. Wrapped items get a new id so their audio is re-rendered."""
    op = rng.choice(OPENERS) if rng.random() < WRAP_OPEN_P else None
    cl = rng.choice(CLOSERS) if rng.random() < WRAP_CLOSE_P else None
    if not (op or cl):
        return item
    def w(s: str) -> str:
        if op:
            s = f"{op} {s[0].lower()}{s[1:]}"
        return f"{s} {cl}" if cl else s
    return {**item, "id": item["id"] + "w", "text": w(item["text"]), "tts_text": w(item["tts_text"])}


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


def _fill(slot: str, rng: random.Random, pools: dict | None = None) -> tuple[str, str | None]:
    """Return (written value, entity type or None)."""
    cars, first, last = (pools["auto"], pools["first"], pools["last"]) if pools else (CARS, FIRST, LAST)
    if slot == "auto":
        return rng.choice(cars), "auto"
    if slot == "nome":
        return (f"{rng.choice(first)} {rng.choice(last)}" if rng.random() < 0.8 else rng.choice(last)), "nome"
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


def generate(n: int, seed: int = 0, wrap: bool = True, templates: list[str] | None = None,
             pools: dict | None = None, prefix: str = "synth") -> list[dict]:
    rng = random.Random(seed)
    templates = templates or TEMPLATES
    wrap_rng = random.Random(seed + 10_000)  # separate stream: unwrapped items stay identical to round 1-2
    block = test_blocklist()
    out = []
    while len(out) < n:
        tpl = rng.choice(templates)
        written, spoken, ents = tpl, tpl, []
        for slot in _SLOT.findall(tpl):
            for _ in range(50):
                val, typ = _fill(slot, rng, pools)
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
        item = {"id": f"{prefix}{len(out):05d}", "text": written, "tts_text": spoken, "entities": ents}
        out.append(_wrap(item, wrap_rng) if wrap else item)
    return out


def generate_heldout(n: int = 200, seed: int = 7) -> list[dict]:
    """test_synth_entities sentences: held-out templates and pools, no phone wrapper."""
    return generate(n, seed=seed, wrap=False, templates=HELDOUT_TEMPLATES, pools=HELDOUT_POOLS, prefix="sent")
