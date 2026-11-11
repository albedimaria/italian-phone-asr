# Modulo di consenso — registrazione di frasi lette al telefono

**Progetto**: italian-phone-asr, valutazione e adattamento del riconoscimento vocale (ASR) per l'italiano al telefono.
**Titolare del trattamento**: Alberto Di Maria — contatto: ____________________ (email)

## Cosa ti chiedo
Leggere ad alta voce circa 108 frasi brevi (circa 10 minuti) da una pagina web, in una chiamata fatta dal browser oppure al cellulare. Le frasi sono inventate: prezzi, date, orari, modelli d'auto, nomi e numeri di telefono **di fantasia**. Non devi dire nulla di personale.

## Perché
Misurare quanto sbagliano diversi sistemi di trascrizione automatica sulla voce al telefono, e se un modello adattato da me sbaglia meno. L'audio serve **solo come test**: non viene usato per addestrare nessun modello.

## Quali dati
- La registrazione della tua voce mentre leggi le frasi.
- Un codice al posto del nome (es. `spk03`). Il collegamento fra codice e nome lo conservo solo io, separato dall'audio.

## Chi riceve l'audio
| Servizio | Perché | Dove |
|---|---|---|
| Twilio | trasporta la chiamata e la registra; dopo il download cancello le registrazioni dal loro sistema | USA |
| Deepgram | trascrizione automatica | USA |
| ElevenLabs | trascrizione automatica | USA/UE |
| OpenAI | trascrizione automatica | USA |

I modelli Whisper e NVIDIA Parakeet girano sul mio computer: l'audio non esce. Il trasferimento verso gli USA avviene secondo le clausole contrattuali standard o il Data Privacy Framework adottati da questi fornitori. Dove il fornitore lo permette, chiedo che l'audio non venga usato per migliorare i suoi modelli.

## Cosa NON succede
- L'audio **non viene pubblicato** da nessuna parte (né su GitHub né su Hugging Face) e non viene ceduto a nessuno.
- Pubblico solo **numeri aggregati** (percentuali di errore), anche per parlante ma sotto codice, mai col tuo nome.

## Quanto tempo
Conservo l'audio sul mio computer e in un archivio privato fino alla fine del progetto, e comunque **non oltre 24 mesi**. Poi lo cancello.

## I tuoi diritti
Puoi chiedere in qualsiasi momento di vedere, avere una copia o **cancellare** le tue registrazioni, e puoi **ritirare il consenso** senza dare spiegazioni, scrivendo al contatto sopra. Il ritiro non tocca i risultati aggregati già pubblicati. Puoi anche presentare reclamo al Garante per la protezione dei dati personali.

Base giuridica: il tuo consenso (art. 6.1.a del GDPR). La partecipazione è volontaria e gratuita.

## Consenso
Ho letto queste informazioni e acconsento alla registrazione della mia voce e al trattamento descritto sopra.

Nome e cognome: ______________________________

Codice parlante (lo compila Alberto): `spk___`

Data: ____ / ____ / ________    Firma: ______________________________

Acconsento anche a essere ricontattato/a per eventuali chiarimenti sul progetto:  ☐ sì  ☐ no
