# GARCH Portfolio Optimizer

A small local web app (built with [Streamlit](https://streamlit.io)) that wraps the
GARCH(1,1) mean-variance optimizer. Runs in your browser, but all computation
happens on your own machine — nothing is uploaded anywhere.

In the app you can set:
- **Tickers** — any comma-separated list of Yahoo Finance symbols
- **Date range** — start/end dates for the price history
- **Frequency** — daily / monthly / quarterly / yearly
- **Risk-free rate**
- **Risk aversion factor (λ)**

## One-time setup

You need Python 3.9+ installed.
- **Mac**: check with `python3 --version` in Terminal. If missing, install from [python.org](https://www.python.org/downloads/) or `brew install python`.
- **Windows**: install from [python.org](https://www.python.org/downloads/). During install, **check "Add python.exe to PATH"**.

## Running the app

### Mac
1. Double-click `run_mac.command`.
   - If macOS blocks it the first time (unidentified developer), right-click the file → **Open** → **Open** again.
   - If double-clicking doesn't run it, first make it executable once in Terminal:
     ```
     chmod +x run_mac.command
     ```
2. A browser tab will open automatically at `http://localhost:8501`.

### Windows
1. Double-click `run_windows.bat`.
2. A browser tab will open automatically at `http://localhost:8501`.

The first run installs the required Python packages into a local virtual
environment (`.venv` folder) — this takes a minute or two. Every run after
that starts in a few seconds.

To stop the app, close the terminal/command window that opened, or press
`Ctrl+C` in it.

## Running manually (any OS)

If you prefer the command line directly:

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- Tickers must be valid Yahoo Finance symbols (e.g. `AAPL`, `VWCE.DE`, `CSX5.L`).
- GARCH fitting needs a reasonable amount of history per asset (at least ~10
  data points at the chosen frequency, but far more is recommended,
  especially for daily data where GARCH is most meaningful).
- If an asset's GARCH fit fails (e.g. too little/no data), it's skipped and
  noted in the "GARCH fit log" panel.
