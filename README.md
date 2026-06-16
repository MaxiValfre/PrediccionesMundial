# Predicciones Mundial 2026

World Cup 2026 prediction engine and static analytics dashboard.

## Stack

- Python
- NumPy
- SciPy
- Static HTML dashboard
- GitHub Actions + GitHub Pages for free deployment

## Local usage

```bash
python run_predictions.py
python run_predictions.py --update
```

## Live updates

Automatic updates use TheSportsDB first and then Wikipedia as a fallback for finished matches.

Optional environment variables:

- `THESPORTSDB_API_KEY`: premium key for v2 live scores
- `THESPORTSDB_LEAGUE_ID`: defaults to `4429` (`FIFA World Cup`)
- `THESPORTSDB_REFERENCE_DATE`: overrides the fetch reference date for testing

Manual live/final overrides can still be written in `data/results_update.json`.

## Free deploy

This repo is prepared for:

- GitHub Pages to serve `docs/`
- GitHub Actions scheduled updates every 15 minutes

Workflows:

- `.github/workflows/update-data.yml`
