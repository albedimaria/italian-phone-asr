# Sessioni di registrazione (test_human)

Ogni frase del copione è una registrazione Twilio separata: nessun allineamento a posteriori.

## Una volta sola
```bash
uv sync --extra recorder
uv run bench record setup      # crea API key e TwiML App sull'account Twilio
```

## Per ogni sessione
1. Il parlante firma `docs/modulo-consenso.md` (carta o PDF). Il modulo firmato resta fuori dal repo.
2. Crea la sessione (codice pseudonimo, mai il nome):
   ```bash
   uv run bench record new spk03                 # browser
   uv run bench record new spk01 --r-moscia      # Alberto
   uv run bench record new spk01m --channel mobile
   ```
3. Avvia il server (apre ngrok e aggiorna la TwiML App), poi manda il link al parlante:
   ```bash
   uv run bench record serve
   ```
   Il link apre una pagina con le istruzioni; ngrok mostra una volta un avviso "Visit site".
4. Solo per la via mobile, con il server acceso:
   ```bash
   uv run bench record call <token> +39XXXXXXXXXX
   ```
   Sul trial Twilio il numero va prima verificato nella console.
5. A fine giornata scarica le registrazioni (e le cancella da Twilio):
   ```bash
   uv run bench record fetch
   ```

Audio e sessioni stanno in `IPASR_DATA/test_human/` (default `D:/data/italian-phone-asr`), mai nel repo.
