# IPL Cricket Intelligence

A modular IPL decision-support application for franchise squad planning, playing-XI construction, tournament scenarios, match simulation, player research, auction shortlisting and Orange/Purple Cap watchlists.

## Technology

- **API:** FastAPI, Python, pandas, NumPy
- **Web client:** React, Vite, Recharts, Lucide
- **Deployment:** Render (single web service; frontend compiled and served by FastAPI)
- **Storage:** CSV-backed local data store on a Render persistent disk

## Current modules

1. Data Centre — multi-competition men's T20 ball-by-ball records ship in the repository; upload more supported history, current franchise squads and optional season fixtures.
2. Squad Lab — create an XI from a franchise squad, tune batting/bowling weighting, specify venue/pitch profile and optionally select an opposition.
3. Tournament — Monte Carlo league-table scenarios from the uploaded fixture list.
4. Match Simulator — compare two complete squads, run match scenarios and view direct batter/bowler evidence plus TL-GE-Entropy over-plan recommendations when the uploaded history supports them.
5. Player Intelligence — search player records, season splits, recent batting appearances and direct historical matchups.
6. Auction Room — upload an auction player CSV and rank candidates by squad-role fit and historical performance evidence.
7. Cap Projections — season-filtered Orange/Purple Cap watchlists based on observed records and simple rate scaling.

The existing `legacy_engine.py` is retained in the backend. Both the match and tournament modules call its delivery-level empirical simulator with TL-GE-Entropy bowling policies and historical batter/bowler matchup adjustments. The match module also applies coarse, explicitly labelled pitch and venue priors. Win probabilities and score ranges are model scenarios, not calibrated forecasts; validate them on held-out seasons before operational use.

## Run locally

Requirements: Python 3.11+ and Node.js 20+.

```bash
# Terminal 1 — API
cd backend
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Vite runs at `http://localhost:5173`. The API runs at `http://localhost:8000`. For local development, create `frontend/.env` with:

```env
VITE_API_URL=http://localhost:8000
```

API documentation is available at `http://localhost:8000/docs`.

## Bundled historical data

The repository includes `data/archive/Cricket_History_MultiLeague_2008_2026.zip` (about 66 MB compressed). It contains a normalized 27-column ball-by-ball index with **961,118 deliveries across 4,110 matches**, covering 2008-04-18 through 2026-09-20 and eight men's T20 competitions:

| Competition | Deliveries | Matches |
|---|---:|---:|
| IPL | 295,732 | 1,243 |
| Syed Mushtaq Ali Trophy | 158,839 | 695 |
| Big Bash League | 153,250 | 662 |
| Bangladesh Premier League | 111,232 | 469 |
| Caribbean Premier League | 103,717 | 445 |
| Pakistan Super League | 83,799 | 357 |
| SA20 | 29,020 | 130 |
| Major League Cricket | 25,529 | 109 |

The archive also retains the supplied match-by-match batting and bowling tables, Bengaluru-area hourly weather observations, player `names` / `people` identity files, the IPL 2018–2024 workbook, and source archives. These are retained as reference/source material. **The current player, matchup and simulation engine runs from the normalized ball-by-ball table; it does not yet independently consume every supplemental table or weather variable.** The `manifest.json` inside the archive documents source files and limitations.

On first boot, the API imports `deliveries.csv` into the Render persistent disk. Subsequent restarts compare the archive SHA-256 and skip re-import when unchanged; a changed archive is merged into the existing index.

### Uploading data not included

The Data Centre keeps an **Additional historical match data** upload. It accepts supported ball-by-ball CSVs, Cricsheet-style JSON files, and ZIP archives containing those files. New records are merged with existing records and deduplicated by match / innings / over / delivery. Keep the source competition and match identifiers in the upload where possible. A CSV without date, batter, bowler and match context cannot be used as a complete ball-by-ball feed; upload the source data with its metadata or use a compatible JSON/CSV export.

Other user uploads remain available:
- Current franchise squads: CSV in Data Centre.
- Season fixtures: CSV in Data Centre.
- Auction candidates: CSV in Auction Room.
- More historical T20 ball-by-ball records: CSV / JSON / ZIP in Data Centre.

The supplemental batting/bowling scorecards, weather data and identity tables are not automatically imported into the live analytics index. Uploading a file does not imply those fields are already used by a model; unsupported schemas are rejected rather than silently treated as canonical deliveries.

### Data provenance and redistribution

The canonical IPL history is retained from the previous normalized IPL archive, including 2026 IPL deliveries. The seven additional competitions are parsed from the supplied men's T20 JSON archives. The companion tables and source archives are copied from the files supplied for this build. Before publishing the repository publicly, verify the original dataset licences, source attribution and redistribution terms.

### Squad CSV

Required logical fields: `team`, `player`, `role`, `country`. `type` is optional. Common aliases include `franchise`, `team_name`, `player_name`, `playing_role`, `nationality` and `player_type`.

Role values should describe wicketkeeper, batter, all-rounder or bowler. Unknown role labels are mapped conservatively; check the uploaded data before selection.

### Fixtures CSV

Required: `date`, `team1`, `team2`. Optional: `venue`, `stage`.

## CSV templates

Header-only starter files are in `templates/`:

- `squad_template.csv`
- `fixtures_template.csv`
- `auction_players_template.csv`

Use the upload guide in Data Centre for accepted aliases and required fields. Historical deliveries can be extended through the Data Centre history upload.

## Auction CSV

Required logical fields: `player`, `role`, `country`. Optional: `type`. Add your own purse, reserve price, availability, overseas slot and auction category columns for your dashboard or a future valuation model. The first release does not forecast auction prices or contract values.

## Push to GitHub

From the project root:

```bash
git init
git add .
git commit -m "Build IPL Cricket Intelligence platform"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ipl-cricket-intelligence.git
git push -u origin main
```

Replace the remote URL with your own GitHub repository.

## Deploy to Render

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and connect the repository.
3. Render reads `render.yaml` and builds the Docker image. The Dockerfile builds React and installs the Python API in one deployable service.
4. The blueprint provisions a persistent disk at `/var/data/ipl`, so uploaded datasets survive normal redeploys.
5. Open the deployed URL. Historical IPL records are already indexed. Upload the latest franchise squad CSV in Data Centre and the auction candidate CSV in Auction Room. Upload a fixture CSV only when setting up a new tournament schedule.

The persistent disk is important: without it, files written to a Render instance's local filesystem can be lost on redeploy or instance replacement. The blueprint uses a 10 GB disk. Keep the repository archive backed up in GitHub; the persistent disk stores the indexed CSV and user workspace data.

## GitHub repository layout

```text
backend/
  app/
    main.py             FastAPI routes and static app hosting
    config.py           Paths and upload limits
    data_service.py     CSV/JSON/ZIP ingestion and persistence
    analytics.py        Profiles, XI, matchup, simulation and auction logic
    legacy_engine.py    Retained original IPL algorithm implementation
  requirements.txt
frontend/
  src/main.jsx          React application and modules
  src/style.css         Metallic / RCB-accent interface
  package.json
  index.html
render.yaml
README.md
```

## Important limitations before a real club deployment

- The bundled dataset needs periodic schema checks, duplicate reconciliation and player identity review. Names that differ across feeds may split a player's career.
- Current profile aggregates are calculated from all indexed records. For fair historical backtests, filter training data by date and hold later matches out.
- The current tournament endpoint assumes a league fixture list and awards two points per simulated win; it does not yet model the official IPL playoff bracket, NRR tiebreaks, rain/DLS or abandoned fixtures.
- The match simulator's win probabilities and score intervals are scenario heuristics, not calibrated probabilities. They should not be presented as betting odds or objective win forecasts.
- TL-GE-Entropy over-plan suggestions are support-gated. Where two or more bowlers lack the required historical support in a state, it uses the policy's empirical fallback. The UI should preserve this source label.
- Auction ranking is a squad-fit shortlist, not an auction valuation model. It does not yet account for purse constraints, availability, injury, player registration or reserve prices.
- Stadium/pitch input currently changes the scenario profile label and coarse score adjustment. Venue-specific pitch behaviour requires enough venue-tagged historical records and a separate validation pass.
- Player profiles expose the metrics supported by the uploaded ball-by-ball schema. Fielding, catches, run-outs, injury, fitness, tracking data and detailed phase/line-length data are not fabricated when absent.
- IPL team logos are not bundled. Use approved brand assets if you have rights to publish them; the interface currently uses neutral monograms.

## Security and production checklist

Before exposing this publicly, add authentication, per-user workspaces, upload scanning, strict CORS, request throttling, audit logs, backups and role-based access. The first release is a private analyst prototype. It is not hardened for public multi-tenant use.
