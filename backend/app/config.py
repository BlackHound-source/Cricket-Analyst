from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv('IPL_DATA_DIR', BASE_DIR / 'data'))
UPLOAD_DIR = DATA_DIR / 'uploads'
STATE_DIR = DATA_DIR / 'state'
for p in (DATA_DIR, UPLOAD_DIR, STATE_DIR):
    p.mkdir(parents=True, exist_ok=True)
DELIVERIES_FILE = STATE_DIR / 'deliveries.csv'
SQUADS_FILE = STATE_DIR / 'squads.csv'
FIXTURES_FILE = STATE_DIR / 'fixtures.csv'
META_FILE = STATE_DIR / 'metadata.json'
BUNDLED_HISTORY_ARCHIVE = BASE_DIR / 'data' / 'archive' / 'Cricket_History_MultiLeague_2008_2026.zip'
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', '300'))
