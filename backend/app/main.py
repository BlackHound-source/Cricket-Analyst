from __future__ import annotations
import json, traceback, io, uuid, threading, time
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from . import data_service as ds, analytics as an
from .config import BASE_DIR, MAX_UPLOAD_MB

JOBS={}; JOB_LOCK=threading.Lock()

def _run_tournament_job(job_id, iterations, batting_bias):
    with JOB_LOCK: JOBS[job_id]={'status':'running','created_at':time.time()}
    try:
        result=an.tournament_sim(iterations,batting_bias)
        with JOB_LOCK: JOBS[job_id]={'status':'completed','result':result,'created_at':JOBS[job_id]['created_at']}
    except Exception as exc:
        with JOB_LOCK: JOBS[job_id]={'status':'failed','error':str(exc),'created_at':JOBS[job_id].get('created_at',time.time())}

app=FastAPI(title='IPL Cricket Intelligence',version='1.2.0',description='Evidence-led IPL squad, player and match decision support.')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_credentials=True,allow_methods=['*'],allow_headers=['*'])

@app.on_event('startup')
def initialize_bundled_dataset():
    # Historical multi-competition T20 data ships with the repository. Additional supported match data, squads, fixtures and auction pools remain uploadable.
    try:
        result = ds.seed_bundled_history()
        print('Bundled cricket data initialization:', json.dumps(result, default=str))
    except Exception as exc:
        # Keep the API available so the failure remains visible in Render logs.
        print('Bundled cricket data initialization failed:', repr(exc))

class XIRequest(BaseModel): team:str; batting_bias:float=Field(.58,ge=0,le=1); conditions:str='balanced'; venue:str=''; opponent:str|None=None
class MatchRequest(BaseModel): team1:str; team2:str; batting_bias:float=Field(.58,ge=0,le=1); venue:str=''; pitch:str='balanced'; iterations:int=Field(1000,ge=100,le=10000)
class TournamentRequest(BaseModel): iterations:int=Field(25,ge=1,le=250); batting_bias:float=Field(.58,ge=0,le=1)
class CapRequest(BaseModel): season:str|None=None; team:str|None=None
class SingleAuctionPlayer(BaseModel): team:str; player:str; role:str; country:str; batting_bias:float=Field(.58,ge=0,le=1)

@app.get('/api/health')
def health(): return {'status':'ok','service':'ipl-cricket-intelligence'}

@app.get('/api/data/status')
def data_status():
    d=ds.deliveries(); m=ds.metadata(); s=an.squad_frame()
    return {**m,'squad_rows':len(s),'teams':sorted(s.team.unique().tolist()) if len(s) else [],'fixtures_uploaded':Path(ds.FIXTURES_FILE).exists(),'years':sorted(d.date.dt.year.dropna().astype(int).unique().tolist()) if len(d) else [],'competitions':sorted(d.competition.dropna().astype(str).unique().tolist()) if len(d) else [],'data_ready':len(d)>0}

async def read_upload(file:UploadFile):
    raw=await file.read()
    if len(raw)>MAX_UPLOAD_MB*1024*1024: raise HTTPException(413,f'File exceeds {MAX_UPLOAD_MB} MB upload limit.')
    return raw

@app.post('/api/data/squads/upload')
async def upload_squads(file:UploadFile=File(...)):
    try:
        out=ds.save_squads(await read_upload(file),file.filename or 'squads.csv')
        return {'message':'Squads uploaded.','rows':len(out),'teams':sorted(out.team.unique().tolist())}
    except Exception as e: raise HTTPException(400,str(e))


@app.post('/api/data/history/upload')
async def upload_history(file:UploadFile=File(...)):
    try:
        raw=await read_upload(file)
        incoming,summary=ds.ingest_file(file.filename or 'history.zip',raw)
        merged=ds.append_deliveries(incoming)
        return {'message':'Additional historical match data imported and merged.','upload_summary':summary,
                'total_delivery_rows':int(len(merged)),'total_matches':int(merged.match_key.nunique()),
                'competitions':sorted(merged.competition.dropna().astype(str).unique().tolist())}
    except Exception as e:
        raise HTTPException(400,str(e))

@app.post('/api/data/fixtures/upload')
async def upload_fixtures(file:UploadFile=File(...)):
    try:
        out=ds.save_fixtures(await read_upload(file)); return {'message':'Fixtures uploaded.','fixtures':len(out),'teams':sorted(set(out.team1)|set(out.team2))}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/api/venues')
def venues():
    d=ds.deliveries()
    if d.empty:return {'venues':[]}
    x=d[d.venue.fillna('').astype(str).str.strip().ne('')].groupby('venue').match_key.nunique().sort_values(ascending=False)
    return {'venues':[{'name':str(k),'matches':int(v)} for k,v in x.items()]}

@app.get('/api/teams')
def teams():
    s=an.squad_frame();
    if s.empty:return {'teams':[]}
    return {'teams':[{'name':t,'players':int((s.team==t).sum())} for t in sorted(s.team.unique())]}

@app.get('/api/players/search')
def search_players(q:str=Query('',min_length=1),limit:int=30):
    d=ds.deliveries()
    if d.empty:return {'results':[]}
    names=sorted(set(d.batter.dropna().astype(str))|set(d.bowler.dropna().astype(str)))
    ql=q.lower(); found=[x for x in names if ql in x.lower()][:limit]
    p=an.player_table(d); p=p[p.player.isin(found)]
    return {'results':p.fillna(0).to_dict('records')}

@app.get('/api/players/{name}/profile')
def player_profile(name:str):
    try:return an.player_profile(name)
    except Exception as e:raise HTTPException(404,str(e))

@app.get('/api/players/{name}/matchups')
def player_matchups(name:str):
    try:
        nm,rows=an.player_matchups(name);return {'player':nm,'matchups':rows}
    except Exception as e:raise HTTPException(404,str(e))

@app.post('/api/teams/xi')
def make_xi(req:XIRequest):
    try:
        xi=an.select_xi(req.team,req.batting_bias,req.conditions,req.venue,req.opponent)
        return {'team':xi.attrs.get('team',req.team),'selection_status':xi.attrs.get('status'),'batting_bias':req.batting_bias,'conditions':req.conditions,'venue':req.venue,'xi':xi.to_dict('records'),'ranked_squad':an._scores_for_players(an.get_team(req.team),an.history(),req.batting_bias,req.conditions,req.venue,req.opponent).sort_values('score',ascending=False).to_dict('records')}
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/matches/simulate')
def match_sim(req:MatchRequest):
    try:return an.simulate_fixture(req.team1,req.team2,req.batting_bias,req.venue,req.pitch,True,req.iterations)
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/tournament/simulate')
def tournament(req:TournamentRequest, background_tasks:BackgroundTasks):
    job_id=uuid.uuid4().hex[:12]
    with JOB_LOCK: JOBS[job_id]={'status':'queued','created_at':time.time()}
    background_tasks.add_task(_run_tournament_job,job_id,req.iterations,req.batting_bias)
    return {'job_id':job_id,'status':'queued','message':'Full tournament simulation queued.'}

@app.get('/api/tournament/jobs/{job_id}')
def tournament_job(job_id:str):
    with JOB_LOCK: job=JOBS.get(job_id)
    if not job: raise HTTPException(404,'Simulation job not found. It may have expired after a server restart.')
    return job

@app.get('/api/matchups/team')
def team_matchups(team:str,opponent:str):
    try:return {'team':team,'opponent':opponent,'matchups':an.matchup_report(team,opponent)}
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/auction/shortlist')
async def auction_shortlist(team:str,batting_bias:float=Query(.58,ge=0,le=1),file:UploadFile=File(...)):
    try:
        raw=await read_upload(file); candidates=pd.read_csv(io.BytesIO(raw)); cols={c.lower().strip():c for c in candidates.columns}
        def col(key,alts=()):
            for k in (key,*alts):
                if k in cols:return candidates[cols[k]]
            return pd.Series('',index=candidates.index)
        frame=pd.DataFrame({'player':col('player',('name','player_name')).astype(str),'team':team,'role':col('role',('playing_role',)).astype(str),'country':col('country',('nationality',)).astype(str),'type':col('type',('player_type',)).astype(str)})
        frame['role_code']=frame.role.map(an.norm_role); country=frame.country.fillna('').astype(str).str.lower(); frame['overseas']=country.ne('') & ~country.str.contains('india')
        result=an.auction_shortlist(frame,team,batting_bias=batting_bias)
        return {'team':team,'shortlist':result.to_dict('records'),'note':'Ranking reflects squad-role fit and available historical evidence. It is not a valuation or auction-price forecast.'}
    except Exception as e:raise HTTPException(400,str(e))

@app.post('/api/auction/shortlist/single')
def auction_single(req:SingleAuctionPlayer):
    try:
        frame=pd.DataFrame([{'player':req.player,'team':req.team,'role':req.role,'country':req.country,'type':''}])
        frame['role_code']=frame.role.map(an.norm_role); country=frame.country.fillna('').astype(str).str.lower(); frame['overseas']=country.ne('') & ~country.str.contains('india')
        result=an.auction_shortlist(frame,req.team,batting_bias=req.batting_bias)
        return {'team':req.team,'shortlist':result.to_dict('records'),'note':'Single-player fit assessment against the selected squad.'}
    except Exception as e: raise HTTPException(400,str(e))

@app.get('/api/caps/projections')
def cap_projections(season:str|None=None):
    d=an.history()
    if d.empty:raise HTTPException(400,'Upload historical delivery data first.')
    # Season-to-date leaders are observed stats; projected values scale observed rates to a full-season match volume.
    if season: d=d[d.season.astype(str)==str(season)]
    else:
        latest=str(d.date.dt.year.max()); d=d[d.date.dt.year.astype(str)==latest]; season=latest
    bat=d.groupby('batter').agg(runs=('batter_runs','sum'),balls=('batter_ball','sum'),matches=('match_key','nunique'),dismissals=('wicket','sum')).reset_index().rename(columns={'batter':'player'})
    bowl=d.groupby('bowler').agg(wickets=('bowler_wickets','sum'),balls=('legal','sum'),matches=('match_key','nunique'),runs_conceded=('runs_conceded','sum')).reset_index().rename(columns={'bowler':'player'})
    bat['strike_rate']=100*bat.runs/bat.balls.clip(lower=1); bat['projected_runs']=bat.runs*(14/bat.matches.clip(lower=1))
    bowl['economy']=6*bowl.runs_conceded/bowl.balls.clip(lower=1); bowl['projected_wickets']=bowl.wickets*(14/bowl.matches.clip(lower=1))
    return {'season':season or 'all uploaded seasons','orange_cap':bat.sort_values('projected_runs',ascending=False).head(20).round(2).to_dict('records'),'purple_cap':bowl.sort_values('projected_wickets',ascending=False).head(20).round(2).to_dict('records'),'interpretation':'Observed and rate-scaled projections, not a forward-looking calibrated season simulation.'}

# Serve the built React app when present. During local development use Vite on port 5173.
FRONTEND_DIST=BASE_DIR/'frontend'/'dist'
if FRONTEND_DIST.exists():
    app.mount('/assets',StaticFiles(directory=FRONTEND_DIST/'assets'),name='assets')
    @app.get('/{full_path:path}')
    def spa(full_path:str):
        candidate=FRONTEND_DIST/full_path
        if candidate.is_file():return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST/'index.html')
