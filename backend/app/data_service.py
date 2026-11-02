from __future__ import annotations
import io, json, zipfile, tempfile, shutil, hashlib
import numpy as np
from pathlib import Path
import pandas as pd
from .config import *
from . import legacy_engine as engine

CANON = ['date','season','competition','match_key','innings','over','delivery_idx','venue','city','batting_team','bowling_team','batter','bowler','batter_id','bowler_id','batter_runs','total_runs','runs_conceded','legal','batter_ball','wides','noballs','wicket','bowler_wickets','is_four','is_six','over_number']
ALIASES = {
 'date':['date','start_date','match_date'], 'season':['season','year'], 'competition':['competition','event','tournament','league'], 'match_key':['match_key','matchid','match_id','id'],
 'innings':['innings','inning','inning_no'], 'over':['over','overs','over_number'], 'venue':['venue','stadium','ground'],
 'batting_team':['batting_team','bat_team','team_batting'], 'bowling_team':['bowling_team','bowl_team','team_bowling'],
 'batter':['batter','batsman','striker'], 'bowler':['bowler','bowling_player'],
 'batter_runs':['batter_runs','batsman_runs','runs_off_bat','runs_batter','batsmanrun'], 'total_runs':['total_runs','total_run','runs_total','totalrun'],
 'runs_conceded':['runs_conceded','bowler_runs','runs_bowler','total_runs','totalrun'], 'legal':['legal','is_legal_ball','is_legal'],
 'batter_ball':['batter_ball','is_batter_ball'], 'wicket':['wicket','is_wicket','iswicketdelivery','player_dismissed_flag','wicket_any'],
 'bowler_wickets':['bowler_wickets','credited_wickets','wicket_bowler_credited'], 'wides':['wides','iswide','is_wide'], 'noballs':['noballs','isnoball','is_noball']
}

def _pick(df, key, default=None):
    low = {str(c).strip().lower(): c for c in df.columns}
    for a in ALIASES.get(key, [key]):
        if a in low: return df[low[a]]
    return pd.Series(default, index=df.index)

def normalize_csv(raw: bytes, filename: str) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(raw), low_memory=False)
    if df.empty: raise ValueError('The CSV contains no rows.')
    d = pd.DataFrame(index=df.index)
    d['date'] = pd.to_datetime(_pick(df,'date'), errors='coerce')
    d['season'] = _pick(df,'season', '').astype(str)
    d.loc[d.season.str.strip().isin(['','nan','None']), 'season'] = d.loc[d.season.str.strip().isin(['','nan','None']), 'date'].dt.year.astype('Int64').astype(str)
    d['competition'] = _pick(df,'competition','').fillna('').astype(str)
    blank_comp = d.competition.str.strip().isin(['','nan','None'])
    inferred_comp = 'IPL' if 'ipl' in str(filename).lower() else 'UNKNOWN'
    d.loc[blank_comp, 'competition'] = inferred_comp
    raw_match_key = _pick(df,'match_key', '').fillna('').astype(str)
    d['match_key'] = np.where(d['competition'].str.strip().str.lower().eq('ipl'), raw_match_key, d['competition'].astype(str) + ':' + raw_match_key)
    missing_key = raw_match_key.str.strip().eq('')
    if missing_key.any():
        pair=pd.DataFrame({'a':_pick(df,'batting_team','').astype(str),'b':_pick(df,'bowling_team','').astype(str)}).apply(lambda x:'|'.join(sorted([x.a,x.b])),axis=1)
        d.loc[missing_key, 'match_key'] = 'upload:' + d.loc[missing_key, 'date'].astype(str) + ':' + pair.loc[missing_key]
    d['innings'] = pd.to_numeric(_pick(df,'innings',1),errors='coerce').fillna(1).astype(int)
    d['over'] = pd.to_numeric(_pick(df,'over',0),errors='coerce').fillna(0).astype(int)
    d['delivery_idx'] = df.groupby([d.match_key,d.innings,d.over],dropna=False).cumcount()
    for k,default in [('venue',''),('batting_team',''),('bowling_team',''),('batter',''),('bowler','')]: d[k]=_pick(df,k,default).fillna(default).astype(str)
    d['city']=''; d['batter_id']=''; d['bowler_id']=''
    for k,default in [('batter_runs',0),('total_runs',None),('runs_conceded',None),('legal',1),('batter_ball',1),('wides',0),('noballs',0),('wicket',0),('bowler_wickets',None)]:
        d[k]=pd.to_numeric(_pick(df,k,default),errors='coerce')
    extras_col = pd.to_numeric(_pick(df,'extras',0), errors='coerce').fillna(0)
    d['total_runs']=d.total_runs.fillna(d.batter_runs + extras_col)
    byes_col = pd.to_numeric(_pick(df,'byes',0), errors='coerce').fillna(0)
    legbyes_col = pd.to_numeric(_pick(df,'legbyes',0), errors='coerce').fillna(0)
    penalty_col = pd.to_numeric(_pick(df,'penalty',0), errors='coerce').fillna(0)
    d['runs_conceded']=d.runs_conceded.fillna(d.total_runs - byes_col - legbyes_col - penalty_col)
    d['legal']=d.legal.fillna(1).astype(int); d['batter_ball']=d.batter_ball.fillna(1).astype(int)
    d['wides']=d.wides.fillna(0); d['noballs']=d.noballs.fillna(0)
    # Derive wicket flags from Cricsheet-style dismissal columns when explicit flags are absent.
    lowcols={str(c).strip().lower():c for c in df.columns}
    if 'player_dismissed' in lowcols:
        dismissed=df[lowcols['player_dismissed']].fillna('').astype(str).str.strip().ne('')
        if not any(c in lowcols for c in ['wicket','is_wicket','player_dismissed_flag']): d['wicket']=dismissed.astype(int)
        kind_col = lowcols.get('dismissal_kind', lowcols.get('kind'))
        kind=df[kind_col].fillna('').astype(str).str.lower() if kind_col else pd.Series('',index=df.index)
        non_credit={'run out','retired hurt','obstructing the field','retired out'}
        if not any(c in lowcols for c in ['bowler_wickets','credited_wickets']): d['bowler_wickets']=(dismissed & ~kind.isin(non_credit)).astype(int)
    d['wicket']=d.wicket.fillna(0).astype(int); d['bowler_wickets']=d.bowler_wickets.fillna(d.wicket).astype(int)
    # Wides and no-balls are not legal deliveries; retain boolean and numeric source flags.
    d['wides']=d.wides.fillna(0); d['noballs']=d.noballs.fillna(0)
    wide_raw=_pick(df,'wides',0).fillna(0).astype(str).str.lower()
    nb_raw=_pick(df,'noballs',0).fillna(0).astype(str).str.lower()
    wide_flag=wide_raw.isin(['1','1.0','true','yes']) | (pd.to_numeric(wide_raw,errors='coerce').fillna(0)>0)
    nb_flag=nb_raw.isin(['1','1.0','true','yes']) | (pd.to_numeric(nb_raw,errors='coerce').fillna(0)>0)
    d['legal']=np.where(wide_flag|nb_flag,0,d.legal).astype(int)
    d['batter_ball']=np.where(wide_flag,0,d.batter_ball).astype(int)
    d['is_four']=(d.batter_runs==4).astype(int); d['is_six']=(d.batter_runs==6).astype(int); d['over_number']=d.over
    d=d[d.date.notna() & d.batter.ne('') & d.bowler.ne('')].copy()
    if d.empty: raise ValueError('Could not identify valid ball-by-ball rows. Check date, batter, bowler and run columns.')
    return d[CANON]

def parse_json_bytes(raw: bytes, source: str) -> pd.DataFrame:
    obj=json.loads(raw)
    engine.CUTOFF=pd.Timestamp.max
    rows=engine.parse_match(obj, source)
    if not rows:
        return pd.DataFrame()
    # Stable match identity must not depend on the uploaded ZIP filename;
    # this lets a user re-upload a source archive without duplicating matches.
    first=rows[0]
    teams=sorted({str(first.get('batting_team','')),str(first.get('bowling_team',''))})
    date=pd.to_datetime(first.get('date'),errors='coerce')
    comp=str(first.get('competition','UNKNOWN'))
    venue=str(first.get('venue','')).strip().lower()
    stable_key=f"{comp}:{date.date()}:{'|'.join(teams)}:{venue}"
    for row in rows:
        row['match_key']=stable_key
    return pd.DataFrame(rows)

def ingest_file(filename: str, raw: bytes) -> tuple[pd.DataFrame,str]:
    ext=Path(filename).suffix.lower()
    frames=[]
    if ext=='.csv':
        frames=[normalize_csv(raw,filename)]
    elif ext=='.json':
        frames=[parse_json_bytes(raw,filename)]
    elif ext=='.zip':
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names=[n for n in z.namelist() if not n.endswith('/') and not n.startswith('__MACOSX')]
            # Prefer a single canonical delivery table when a source ZIP contains
            # multiple overlapping exports. Otherwise the same match can be counted
            # more than once with different ball-number conventions.
            root_canonical=next((n for n in names if Path(n).name.lower()=='deliveries.csv'),None)
            preferred_ipl=next((n for n in names if Path(n).name.lower()=='deliveries_updated_ipl_upto_2025.csv'),None)
            if root_canonical:
                names=[root_canonical]
            elif preferred_ipl:
                names=[preferred_ipl]
            for name in names:
                b=z.read(name); low=name.lower()
                try:
                    if low.endswith('.csv'): frames.append(normalize_csv(b,name))
                    elif low.endswith('.json'): 
                        parsed=parse_json_bytes(b,name)
                        if parsed is not None and not parsed.empty: frames.append(parsed)
                    elif low.endswith('.zip'):
                        nested,_=ingest_file(name,b); frames.append(nested)
                except Exception:
                    continue
    else: raise ValueError('Upload CSV, JSON or ZIP.')
    frames=[x for x in frames if x is not None and not x.empty]
    if not frames: raise ValueError('No usable match or delivery records were found in this upload.')
    out=pd.concat(frames,ignore_index=True,sort=False)
    out['date']=pd.to_datetime(out.date,errors='coerce')
    out=out.dropna(subset=['date','batter','bowler']).drop_duplicates(['match_key','innings','over','delivery_idx'])
    return out, f'{len(out):,} delivery rows across {out.match_key.nunique():,} matches'

def append_deliveries(incoming: pd.DataFrame):
    if DELIVERIES_FILE.exists():
        old=pd.read_csv(DELIVERIES_FILE,low_memory=False,parse_dates=['date'])
        merged=pd.concat([old,incoming],ignore_index=True,sort=False)
    else: merged=incoming.copy()
    merged['date']=pd.to_datetime(merged.date,errors='coerce')
    merged=merged.drop_duplicates(['match_key','innings','over','delivery_idx'],keep='last')
    merged.to_csv(DELIVERIES_FILE,index=False)
    meta=metadata(); meta.update({'delivery_rows':int(len(merged)),'matches':int(merged.match_key.nunique()),'first_date':str(merged.date.min().date()),'last_date':str(merged.date.max().date()),'seasons':sorted(merged.season.dropna().astype(str).unique().tolist()),'competitions':sorted(merged.competition.dropna().astype(str).unique().tolist()),'competition_counts':{str(k):{'delivery_rows':int(v),'matches':int(merged.loc[merged.competition==k,'match_key'].nunique())} for k,v in merged.groupby('competition').size().items()},'updated_at':pd.Timestamp.utcnow().isoformat()})
    META_FILE.write_text(json.dumps(meta,indent=2))
    return merged


def seed_bundled_history():
    """Seed or refresh the persistent workspace from the versioned Git archive."""
    if not BUNDLED_HISTORY_ARCHIVE.exists():
        return {'seeded': False, 'reason': 'bundled_archive_missing'}
    archive_hash = hashlib.sha256(BUNDLED_HISTORY_ARCHIVE.read_bytes()).hexdigest()
    current_meta = metadata()
    if DELIVERIES_FILE.exists() and DELIVERIES_FILE.stat().st_size > 0 and current_meta.get('archive_sha256') == archive_hash:
        return {'seeded': False, 'reason': 'bundled_archive_unchanged', **current_meta}
    with zipfile.ZipFile(BUNDLED_HISTORY_ARCHIVE) as archive:
        with archive.open('deliveries.csv') as stream:
            incoming = pd.read_csv(stream, low_memory=False, parse_dates=['date'])
    missing = [c for c in CANON if c not in incoming.columns]
    if missing:
        raise ValueError(f'Bundled archive is missing canonical columns: {missing}')
    # Merge instead of replacing so any separately maintained records survive an archive refresh.
    merged = append_deliveries(incoming[CANON])
    meta = metadata()
    meta.update({
        'source': 'Git-bundled multi-competition T20 archive (2008–2026)',
        'archive_name': BUNDLED_HISTORY_ARCHIVE.name,
        'competitions': sorted(merged.competition.dropna().astype(str).unique().tolist()),
        'competition_counts': {str(k): {'delivery_rows': int(v), 'matches': int(merged.loc[merged.competition==k, 'match_key'].nunique())} for k,v in merged.groupby('competition').size().items()},
        'archive_sha256': archive_hash,
        'seeded_at': pd.Timestamp.utcnow().isoformat(),
    })
    META_FILE.write_text(json.dumps(meta, indent=2))
    return {'seeded': True, 'delivery_rows': int(len(merged)), 'matches': int(merged.match_key.nunique()), 'years': meta.get('seasons', []), 'archive_sha256': archive_hash}


def deliveries():
    if not DELIVERIES_FILE.exists(): return pd.DataFrame(columns=CANON)
    d=pd.read_csv(DELIVERIES_FILE,low_memory=False,parse_dates=['date'])
    for c in CANON:
        if c not in d: d[c]=None
    return d

def metadata():
    if META_FILE.exists():
        try:return json.loads(META_FILE.read_text())
        except Exception:pass
    return {'delivery_rows':0,'matches':0,'seasons':[],'note':'No data uploaded yet.'}

def save_squads(raw:bytes,filename:str):
    s=pd.read_csv(io.BytesIO(raw)); cols={c.lower().strip():c for c in s.columns}
    def get(c,alts=()):
        for a in (c,*alts):
            if a in cols:return s[cols[a]]
        return pd.Series('',index=s.index)
    out=pd.DataFrame({'team':get('team',('franchise','team_name')).astype(str).str.strip(),'player':get('player',('name','player_name')).astype(str).str.strip(),'role':get('role',('playing_role',)).astype(str).str.strip(),'country':get('country',('nationality',)).astype(str).str.strip(),'type':get('type',('player_type','status')).astype(str).str.strip()})
    out=out[(out.team!='')&(out.player!='')].drop_duplicates(['team','player'])
    if out.empty: raise ValueError('Squad file needs at least team and player columns.')
    counts=out.groupby('team').player.nunique()
    if (counts<11).any(): raise ValueError('Each uploaded team needs at least 11 unique players: '+str(counts[counts<11].to_dict()))
    out.to_csv(SQUADS_FILE,index=False)
    return out

def save_fixtures(raw:bytes):
    f=pd.read_csv(io.BytesIO(raw)); cols={c.lower().strip():c for c in f.columns}
    def get(c,alts=()):
        for a in (c,*alts):
            if a in cols:return f[cols[a]]
        return pd.Series('',index=f.index)
    out=pd.DataFrame({'date':pd.to_datetime(get('date'),errors='coerce'),'team1':get('team1',('home_team',)).astype(str),'team2':get('team2',('away_team',)).astype(str),'venue':get('venue',('stadium','ground')).astype(str),'stage':get('stage',('match_type',)).astype(str).replace('','LEAGUE')})
    out=out.dropna(subset=['date']); out=out[(out.team1!='')&(out.team2!='')]
    if out.empty:raise ValueError('Fixtures need date, team1 and team2 columns.')
    out.to_csv(FIXTURES_FILE,index=False); return out
