# Predicciones Mundial 2026

World Cup 2026 prediction engine with a static dashboard ready to open locally or publish via GitHub Pages.

## Quick Path

1. Run `python run_predictions.py`
2. Open `dashboard/index.html`
3. Use `docs/index.html` as the published GitHub Pages copy

`run_predictions.py` generates `dashboard/predictions.json`, embeds the same data directly into `dashboard/index.html`, and mirrors both files into `docs/`.

## Stack

- Python
- NumPy
- SciPy
- Static HTML dashboard
- GitHub Actions + GitHub Pages

## Commands

```bash
python run_predictions.py
python run_predictions.py --update
```

- `python run_predictions.py`: recomputes predictions from current local data and leaves the dashboard ready to open.
- `python run_predictions.py --update`: fetches latest match updates before regenerating the dashboard.

## Live Updates

Automatic refresh uses these sources in order:

1. TheSportsDB for live and final match updates
2. Wikipedia as a fallback for finished matches
3. `data/results_update.json` for manual local overrides

Optional environment variables:

- `THESPORTSDB_API_KEY`: premium key for v2 live scores
- `THESPORTSDB_LEAGUE_ID`: defaults to `4429` (`FIFA World Cup`)
- `THESPORTSDB_REFERENCE_DATE`: overrides the fetch reference date for testing
- `ODDS_API_KEY`: enables bookmaker market odds in the dashboard

## Deploy

This repo is prepared for:

- GitHub Pages serving `docs/`
- GitHub Actions scheduled refreshes every 7 minutes
- `workflow_dispatch` runs for external schedulers if cron reliability is not enough

Workflow:

- `.github/workflows/update-data.yml`
