import pytest

from iphasr.normalize import normalize


@pytest.mark.parametrize("text, expected", [
    # case, punctuation, apostrophes, accents
    ("Buongiorno, è possibile?", "buongiorno e possibile"),
    ("L'appuntamento", "l appuntamento"),
    ("Sì, perché no!", "si perche no"),
    # plain numbers: digits and words meet in the middle
    ("ho 34 anni", "ho trentaquattro anni"),
    ("ho trentaquattro anni", "ho trentaquattro anni"),
    ("ventitré", "ventitre"),
    ("23", "ventitre"),
    ("del 2019", "del duemiladiciannove"),
    # thousands separator, the Collaudio "dodicimilaquattrocento" case
    ("12.400 euro", "dodicimilaquattrocento euro"),
    ("dodicimilaquattrocento euro", "dodicimilaquattrocento euro"),
    ("dodici mila quattrocento euro", "dodicimilaquattrocento euro"),
    ("12400€", "dodicimilaquattrocento euro"),
    ("€ 12.400", "dodicimilaquattrocento euro"),
    ("140.000 chilometri", "centoquarantamila chilometri"),
    # an ASR glitch stays an error, it is not repaired
    ("12000.400", "dodicimila virgola quattrocento"),
    ("quattordicimila novecento euro", "quattordicimilanovecento euro"),
    # prices with cents
    ("12,50 €", "dodici euro e cinquanta"),
    ("87 euro e 40", "ottantasette euro e quaranta"),
    ("€19,90", "diciannove euro e novanta"),
    ("10,00 euro", "dieci euro"),
    # times
    ("alle 15:30", "alle quindici e trenta"),
    ("alle 15.30", "alle quindici e trenta"),
    ("alle quindici e trenta", "alle quindici e trenta"),
    ("alle 9:00", "alle nove"),
    ("alle 9:05", "alle nove e cinque"),
    ("alle quindici e mezza", "alle quindici e trenta"),
    ("alle otto e un quarto", "alle otto e quindici"),
    # dates
    ("il 18/12/2026", "il diciotto dicembre duemilaventisei"),
    ("il 18 dicembre", "il diciotto dicembre"),
    ("il 1 dicembre", "il primo dicembre"),
    ("il 1° dicembre", "il primo dicembre"),
    ("il 1º orario", "il primo orario"),
    ("il primo dicembre", "il primo dicembre"),
    ("l'11 marzo", "l undici marzo"),
    # phone numbers: digit by digit, whatever the grouping
    ("347 281 9063", "tre quattro sette due otto uno nove zero sei tre"),
    ("3472819063", "tre quattro sette due otto uno nove zero sei tre"),
    ("347-2819063", "tre quattro sette due otto uno nove zero sei tre"),
    ("tre quattro sette due otto uno nove zero sei tre", "tre quattro sette due otto uno nove zero sei tre"),
    ("02 4857 3310", "zero due quattro otto cinque sette tre tre uno zero"),
    # decimals, percent, alphanumeric models
    ("al 5,9%", "al cinque virgola nove per cento"),
    ("il 15 per cento", "il quindici per cento"),
    ("Peugeot 208", "peugeot duecentotto"),
    ("Hyundai i20", "hyundai i venti"),
    ("Citroën C3", "citroen c tre"),
    ("Mazda CX-30", "mazda cx trenta"),
    ("Fiat 500", "fiat cinquecento"),
    # fillers dropped
    ("ehm vorrei uhm prenotare", "vorrei prenotare"),
    ("", ""),
])
def test_normalize(text, expected):
    assert normalize(text) == expected


def test_idempotent_on_script():
    from iphasr.script import load_script
    for line in load_script():
        once = normalize(line.text)
        assert normalize(once) == once, line.text


def test_whisper_style_equals_spoken_style():
    whisper = "Ho visto una Fiat Tipo a 14.900 €, il 18/12 alle 15:30. Il mio numero è 347 2819063."
    spoken = ("ho visto una fiat tipo a quattordicimilanovecento euro il diciotto dicembre alle quindici e trenta "
              "il mio numero e tre quattro sette due otto uno nove zero sei tre")
    assert normalize(whisper) == normalize(spoken)
