from __future__ import annotations
import math, re
from collections import defaultdict
import numpy as np
import pandas as pd
from .data_service import deliveries
from .config import SQUADS_FILE, FIXTURES_FILE
from . import legacy_engine as core

ROLE_MAP={'WK':'WK','WICKETKEEPER':'WK','WICKET KEEPER':'WK','BAT':'BAT','BATTER':'BAT','BATSMAN':'BAT','TOP ORDER BATTER':'BAT','MIDDLE ORDER BATTER':'BAT','AR':'AR','ALL ROUNDER':'AR','ALL-ROUNDER':'AR','BOWLING ALLROUNDER':'AR','BOWLER':'BOWL','BOWL':'BOWL','FAST BOWLER':'BOWL','SPIN BOWLER':'BOWL'}

def clean(s): return re.sub(r'[^a-z0-9]+',' ',str(s).lower()).strip()
def norm_role(x):
    raw=str(x).lower().strip()
    if any(k in raw for k in ['wicket','keeper','wk']): return 'WK'
    if any(k in raw for k in ['allround','all-round','all round']): return 'AR'
    if any(k in raw for k in ['bowl','fast','seam','spin']): return 'BOWL'
    if any(k in raw for k in ['bat','batter','batsman']): return 'BAT'
    return ROLE_MAP.get(str(x).upper().strip(),'BAT')
def history():
    d=deliveries()
    if not d.empty:
        d['is_ipl']=d.competition.astype(str).str.contains('ipl',case=False)
        d['season']=d.season.fillna('').astype(str)
        d['date']=pd.to_datetime(d.date,errors='coerce')
    return d

def player_table(d=None):
    d=d if d is not None else history()
    if d.empty:return pd.DataFrame()
    bat=d.groupby('batter').agg(bat_matches=('match_key','nunique'),runs=('batter_runs','sum'),balls_faced=('batter_ball','sum'),fours=('is_four','sum'),sixes=('is_six','sum'),dismissals=('wicket','sum')).reset_index().rename(columns={'batter':'player'})
    bowl=d.groupby('bowler').agg(bowl_matches=('match_key','nunique'),balls_bowled=('legal','sum'),runs_conceded=('runs_conceded','sum'),wickets=('bowler_wickets','sum')).reset_index().rename(columns={'bowler':'player'})
    p=bat.merge(bowl,on='player',how='outer').fillna(0)
    p['matches']=p[['bat_matches','bowl_matches']].max(axis=1)
    p['strike_rate']=100*p.runs/p.balls_faced.clip(lower=1)
    p['batting_average']=p.runs/p.dismissals.clip(lower=1)
    p['economy']=6*p.runs_conceded/p.balls_bowled.clip(lower=1)
    p['bowling_average']=p.runs_conceded/p.wickets.clip(lower=1)
    p['bowling_strike_rate']=p.balls_bowled/p.wickets.clip(lower=1)
    p['dot_ball_pct']=100*(1-(d.assign(dot=(d.batter_runs==0).astype(int)).groupby('batter').dot.sum().reindex(p.player).fillna(0).to_numpy()/p.balls_faced.clip(lower=1).to_numpy()))
    p['boundary_pct']=100*(p.fours+p.sixes)/p.balls_faced.clip(lower=1)
    return p.sort_values('runs',ascending=False)

def player_profile(name):
    d=history()
    if d.empty: raise ValueError('Upload ball-by-ball data first.')
    q=clean(name); names=pd.concat([d.batter,d.bowler]).dropna().astype(str).unique().tolist()
    matches=[n for n in names if q in clean(n) or clean(n) in q]
    if not matches: raise ValueError(f'No player found matching {name!r}.')
    chosen=sorted(matches,key=lambda n:(clean(n)!=q,len(n)))[0]
    bat=d[d.batter==chosen].copy(); bowl=d[d.bowler==chosen].copy()
    p=player_table(d); row=p[p.player==chosen].head(1)
    season=[]
    for yr,g in bat.groupby('season'):
        season.append({'season':str(yr),'runs':int(g.batter_runs.sum()),'balls':int(g.batter_ball.sum()),'dismissals':int(g.wicket.sum()),'strike_rate':round(100*g.batter_runs.sum()/max(1,g.batter_ball.sum()),2),'average':round(g.batter_runs.sum()/max(1,g.wicket.sum()),2),'boundary_pct':round(100*(g.is_four.sum()+g.is_six.sum())/max(1,g.batter_ball.sum()),2),'fours':int(g.is_four.sum()),'sixes':int(g.is_six.sum()),'matches':int(g.match_key.nunique())})
    bowl_season=[]
    for yr,g in bowl.groupby('season'):
        balls=int(g.legal.sum()); runs=int(g.runs_conceded.sum()); wk=int(g.bowler_wickets.sum())
        bowl_season.append({'season':str(yr),'wickets':wk,'balls':balls,'runs_conceded':runs,'overs':round(balls/6,1),'economy':round(6*runs/max(1,balls),2),'average':round(runs/max(1,wk),2),'bowling_strike_rate':round(balls/max(1,wk),2),'matches':int(g.match_key.nunique())})
    recent=bat.groupby(['match_key','date','venue']).agg(runs=('batter_runs','sum'),balls=('batter_ball','sum'),fours=('is_four','sum'),sixes=('is_six','sum')).reset_index().sort_values('date',ascending=False).head(30)
    recent_bowl=bowl.groupby(['match_key','date','venue']).agg(wickets=('bowler_wickets','sum'),runs_conceded=('runs_conceded','sum'),balls=('legal','sum')).reset_index().sort_values('date',ascending=False).head(30)
    recent_bowl['overs']=recent_bowl.balls/6; recent_bowl['economy']=6*recent_bowl.runs_conceded/recent_bowl.balls.clip(lower=1)
    bat['phase']=np.where(bat.over<6,'Powerplay',np.where(bat.over<15,'Middle','Death'))
    bowl['phase']=np.where(bowl.over<6,'Powerplay',np.where(bowl.over<15,'Middle','Death'))
    bp=[]
    for phase,g in bat.groupby('phase'):
        bp.append({'phase':phase,'runs':int(g.batter_runs.sum()),'balls':int(g.batter_ball.sum()),'dismissals':int(g.wicket.sum()),'strike_rate':round(100*g.batter_runs.sum()/max(1,g.batter_ball.sum()),2),'matches':int(g.match_key.nunique())})
    wp=[]
    for phase,g in bowl.groupby('phase'):
        balls=int(g.legal.sum()); runs=int(g.runs_conceded.sum()); wickets=int(g.bowler_wickets.sum())
        wp.append({'phase':phase,'wickets':wickets,'balls':balls,'runs_conceded':runs,'economy':round(6*runs/max(1,balls),2),'average':round(runs/max(1,wk),2),'bowling_strike_rate':round(balls/max(1,wk),2),'matches':int(g.match_key.nunique())})
    venue_rows=bat.groupby('venue').agg(runs=('batter_runs','sum'),balls=('batter_ball','sum'),matches=('match_key','nunique')).reset_index()
    venue_rows['strike_rate']=100*venue_rows.runs/venue_rows.balls.clip(lower=1)
    opp_rows=bat.groupby('bowling_team').agg(runs=('batter_runs','sum'),balls=('batter_ball','sum'),dismissals=('wicket','sum'),matches=('match_key','nunique')).reset_index().rename(columns={'bowling_team':'opposition'})
    opp_rows['strike_rate']=100*opp_rows.runs/opp_rows.balls.clip(lower=1);opp_rows['average']=opp_rows.runs/opp_rows.dismissals.clip(lower=1)
    bat_comp=bat.groupby('competition').agg(matches=('match_key','nunique'),runs=('batter_runs','sum'),balls=('batter_ball','sum'),dismissals=('wicket','sum')).reset_index()
    bat_comp['average']=bat_comp.runs/bat_comp.dismissals.clip(lower=1); bat_comp['strike_rate']=100*bat_comp.runs/bat_comp.balls.clip(lower=1)
    bowl_comp=bowl.groupby('competition').agg(matches=('match_key','nunique'),wickets=('bowler_wickets','sum'),balls=('legal','sum'),runs_conceded=('runs_conceded','sum')).reset_index()
    bowl_comp['overs']=bowl_comp.balls/6; bowl_comp['economy']=6*bowl_comp.runs_conceded/bowl_comp.balls.clip(lower=1)
    bowl_comp['average']=bowl_comp.runs_conceded/bowl_comp.wickets.clip(lower=1)
    return {'name':chosen,'summary':row.iloc[0].to_dict() if len(row) else {},'batting_by_season':season,'bowling_by_season':bowl_season,'batting_by_competition':bat_comp.round(2).to_dict('records'),'bowling_by_competition':bowl_comp.round(2).to_dict('records'),'batting_by_phase':bp,'bowling_by_phase':wp,'venue_batting':venue_rows.sort_values('matches',ascending=False).head(20).round(2).to_dict('records'),'opposition_batting':opp_rows.sort_values('balls',ascending=False).head(30).round(2).to_dict('records'),'recent_batting':recent.fillna('').to_dict('records'),'recent_bowling':recent_bowl.round(2).fillna('').to_dict('records'),'seasons':sorted(set(bat.season.astype(str))|set(bowl.season.astype(str)))}

def player_matchups(name):
    d=history(); q=clean(name)
    candidates=pd.concat([d.batter,d.bowler]).dropna().astype(str).unique()
    nm=next((x for x in candidates if clean(x)==q),None)
    if not nm: nm=next((x for x in candidates if q in clean(x)),None)
    if not nm: raise ValueError('Player not found in uploaded delivery data.')
    parts=[]
    if (d.batter==nm).any():
        x=d[d.batter==nm].groupby('bowler').agg(balls=('batter_ball','sum'),runs=('batter_runs','sum'),dismissals=('wicket','sum'),matches=('match_key','nunique')).reset_index().rename(columns={'bowler':'opponent'})
        x['strike_rate']=100*x.runs/x.balls.clip(lower=1); x['average']=x.runs/x.dismissals.clip(lower=1); x['type']='Batter vs bowler'; parts.append(x)
    if (d.bowler==nm).any():
        x=d[d.bowler==nm].groupby('batter').agg(balls=('legal','sum'),runs=('runs_conceded','sum'),dismissals=('bowler_wickets','sum'),matches=('match_key','nunique')).reset_index().rename(columns={'batter':'opponent'})
        x['strike_rate']=100*x.runs/x.balls.clip(lower=1); x['average']=x.runs/x.dismissals.clip(lower=1); x['type']='Bowler vs batter'; parts.append(x)
    combined=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    return nm,combined.sort_values(['balls','dismissals'],ascending=False).head(100).round(2).to_dict('records')

def squad_frame():
    if not SQUADS_FILE.exists():return pd.DataFrame()
    s=pd.read_csv(SQUADS_FILE); s['role_code']=s.role.map(norm_role); country=s.country.fillna('').astype(str).str.lower(); kind=s.type.fillna('').astype(str).str.lower(); s['overseas']=kind.str.contains('overseas') | (country.ne('') & ~country.str.contains('india'))
    return s

def get_team(team):
    s=squad_frame()
    if s.empty: raise ValueError('Upload a squad CSV first.')
    q=clean(team); names=s.team.unique().tolist(); hit=next((x for x in names if clean(x)==q),None)
    if not hit: hit=next((x for x in names if q in clean(x)),None)
    if not hit: raise ValueError(f'Team {team!r} is not in the uploaded squad.')
    return s[s.team==hit].copy()

def _scores_for_players(squad, d, bat_weight=0.58, conditions='balanced', venue='', opponent=None):
    p=player_table(d)
    lookup={clean(r.player):r for r in p.itertuples(index=False)}
    rows=[]
    for r in squad.itertuples(index=False):
        x=lookup.get(clean(r.player)); runs=float(getattr(x,'runs',0)) if x else 0; balls=float(getattr(x,'balls_faced',0)) if x else 0
        sr=float(getattr(x,'strike_rate',125)) if x else 125; avg=float(getattr(x,'batting_average',22)) if x else 22
        econ=float(getattr(x,'economy',8.7)) if x else 8.7; wk=float(getattr(x,'wickets',0)) if x else 0; bb=float(getattr(x,'balls_bowled',0)) if x else 0
        sr_adj=(sr*balls+125*100)/(balls+100); avg_adj=(avg*balls+22*100)/(balls+100); econ_adj=(econ*bb+8.7*100)/(bb+100)
        bat=np.clip(.6*sr_adj/160*100+.4*avg_adj/40*100,0,100); bowl=np.clip((10.5-econ_adj)/4*65+(wk/max(1,bb)*1000)*35,0,100)
        role=r.role_code; rolebonus={'WK':5,'BAT':3,'AR':7,'BOWL':4}[role]
        style=clean(getattr(r,'role',''))+' '+clean(getattr(r,'type',''))
        condition_delta=0.0
        if conditions=='spin' and role in ('AR','BOWL') and any(k in style for k in ['spin','offbreak','legbreak','orthodox','slow']): condition_delta=7.0
        elif conditions=='pace' and role in ('AR','BOWL') and any(k in style for k in ['fast','seam','swing','pace']): condition_delta=7.0
        elif conditions=='batting': condition_delta=3.0*bat/100
        elif conditions=='bowling': condition_delta=3.0*bowl/100
        venue_delta=0.0
        if venue and not d.empty:
            vg=d[d.venue.astype(str).str.contains(re.escape(venue),case=False,na=False)]
            if not vg.empty:
                vbat=vg[vg.batter.map(clean)==clean(r.player)]
                if len(vbat)>=30:
                    vsr=100*vbat.batter_runs.sum()/max(1,vbat.batter_ball.sum()); venue_delta+=float(np.clip((vsr-sr)*.035,-4,4))*bat_weight
                vbowl=vg[vg.bowler.map(clean)==clean(r.player)]
                if vbowl.legal.sum()>=30:
                    vecon=6*vbowl.runs_conceded.sum()/max(1,vbowl.legal.sum()); venue_delta+=float(np.clip((econ-vecon)*1.1,-4,4))*(1-bat_weight)
        score=bat_weight*bat+(1-bat_weight)*bowl+rolebonus+min(8,math.log1p(balls+bb)/2)+condition_delta+venue_delta
        rows.append({'player':r.player,'team':r.team,'role':role,'overseas':bool(r.overseas),'bat_score':round(float(bat),1),'bowl_score':round(float(bowl),1),'condition_adjustment':round(condition_delta,2),'venue_adjustment':round(venue_delta,2),'score':round(float(score),2),'runs':int(runs),'balls_faced':int(balls),'strike_rate':round(sr,1),'average':round(avg,1),'wickets':int(wk),'economy':round(econ,2),'historical_bowling_balls':int(bb),'evidence':'Historical' if x else 'No IPL record'})
    return pd.DataFrame(rows)

def select_xi(team, batting_bias=0.58, conditions='balanced', venue='', opponent=None):
    d=history(); s=get_team(team); p=_scores_for_players(s,d,batting_bias,conditions,venue,opponent)
    if opponent and not d.empty:
        try:
            opp_squad=get_team(opponent); opp_bowlers=opp_squad.player.tolist(); opp_batters=opp_squad.player.tolist()
            fit=[]
            for player in p.player:
                vsb=d[(d.batter.map(clean)==clean(player)) & d.bowler.map(clean).isin([clean(x) for x in opp_bowlers])]
                vsa=d[(d.bowler.map(clean)==clean(player)) & d.batter.map(clean).isin([clean(x) for x in opp_batters])]
                bat_signal=0.0; bowl_signal=0.0
                if vsb.batter_ball.sum()>=12:
                    sr_vs=100*vsb.batter_runs.sum()/max(1,vsb.batter_ball.sum()); dismiss_rate=vsb.wicket.sum()/max(1,vsb.batter_ball.sum())
                    bat_signal=float(np.clip((sr_vs-125)*.08 - dismiss_rate*30,-5,5))*min(1,vsb.batter_ball.sum()/60)
                if vsa.legal.sum()>=12:
                    econ_vs=6*vsa.runs_conceded.sum()/max(1,vsa.legal.sum()); wk_rate=vsa.bowler_wickets.sum()/max(1,vsa.legal.sum())
                    bowl_signal=float(np.clip((8.7-econ_vs)*1.2 + (wk_rate-.018)*100,-5,5))*min(1,vsa.legal.sum()/60)
                fit.append((bat_signal*batting_bias+bowl_signal*(1-batting_bias)))
            p['opposition_fit']=np.round(fit,2); p['score']=p.score+p.opposition_fit
        except Exception:
            p['opposition_fit']=0.0
    else:p['opposition_fit']=0.0
    # Role and overseas constraints, with a transparent fallback if the squad cannot satisfy them.
    p=p.sort_values('score',ascending=False); best=None; bestscore=-1e9
    import itertools
    pool=p.head(min(18,len(p))).to_dict('records')
    for combo in itertools.combinations(pool,11):
        roles=[x['role'] for x in combo]; overseas=sum(x['overseas'] for x in combo)
        valid=(overseas<=4 and 1<=roles.count('WK')<=2 and sum(r in ('BAT','WK','AR') for r in roles)>=6 and sum(r in ('BOWL','AR') for r in roles)>=5 and roles.count('BAT')>=2 and roles.count('BOWL')>=2)
        if not valid:continue
        score=sum(x['score'] for x in combo)
        if score>bestscore:best,bestscore=combo,score
    status='CONSTRAINT_VALID'
    if best is None: best=pool[:11]; status='PROVISIONAL_CONSTRAINT_RELAXED'
    out=pd.DataFrame(best).sort_values('score',ascending=False)
    if opponent:
        out['opposition_focus']=f'Matchup review required vs {opponent}; direct evidence is displayed separately.'
    out.attrs['team']=s.team.iloc[0]; out.attrs['status']=status; out.attrs['batting_bias']=batting_bias
    return out

def matchup_report(team, opponent):
    d=history(); s=get_team(team); opp=get_team(opponent)
    rows=[]
    for b in s.player:
        for w in opp.player:
            g=d[(d.batter.map(clean)==clean(b))&(d.bowler.map(clean)==clean(w))]
            balls=int(g.batter_ball.sum()); runs=int(g.batter_runs.sum()); outs=int(g.wicket.sum())
            if balls:
                rows.append({'batter':b,'bowler':w,'balls':balls,'runs':runs,'dismissals':outs,'strike_rate':round(100*runs/balls,1),'evidence_matches':int(g.match_key.nunique()),'evidence':'Direct historical matchup'})
    return sorted(rows,key=lambda x:(x['dismissals'],x['balls']),reverse=True)[:60]

def simulate_fixture(team1,team2,batting_bias=.58,venue='',pitch='balanced',opponent_focus=True,iterations=1000):
    from types import SimpleNamespace
    from collections import Counter
    d=history()
    if d.empty: raise ValueError('Upload ball-by-ball history first.')
    a=select_xi(team1,batting_bias,pitch,venue,team2)
    b=select_xi(team2,batting_bias,pitch,venue,team1)
    ra=float(a.score.mean()); rb=float(b.score.mean())
    # Venue reference is used only when a meaningful sample is available.
    venue_context={'venue':venue or 'Not specified','matches':0,'median_innings_score':None,'source':'generic IPL baseline'}
    venue_base=166.0
    if venue:
        vg=d[d.venue.astype(str).str.contains(re.escape(venue),case=False,na=False)]
        if not vg.empty:
            innings=vg.groupby(['match_key','innings']).total_runs.sum().reset_index()
            venue_context['matches']=int(vg.match_key.nunique())
            if venue_context['matches']>=8:
                venue_base=float(innings.total_runs.median())
                venue_context.update({'median_innings_score':round(venue_base,1),'source':'venue-specific uploaded history'})

    # Stable name IDs provide exact-name identity linkage when a formal registry is absent.
    hist=d.copy(); hist['batter_id']=hist.batter_id.fillna('').astype(str); hist['bowler_id']=hist.bowler_id.fillna('').astype(str)
    hist.loc[hist.batter_id.eq(''),'batter_id']=hist.loc[hist.batter_id.eq(''),'batter'].map(core.norm)
    hist.loc[hist.bowler_id.eq(''),'bowler_id']=hist.loc[hist.bowler_id.eq(''),'bowler'].map(core.norm)
    resolved=squad_frame().copy(); resolved['TeamCanonical']=resolved.team; resolved['Player']=resolved.player
    resolved['PlayerNorm']=resolved.player.map(core.norm); resolved['RoleCode']=resolved.role_code
    resolved['Overseas']=resolved.overseas; resolved['player_id']=resolved.PlayerNorm
    resolved['identity_status']='NAME_MATCH'; resolved['identity_method']='normalized_name'; resolved['historical_names']=resolved.player
    xis={team1:a.rename(columns={'player':'Player'}).copy(),team2:b.rename(columns={'player':'Player'}).copy()}
    bdist,wskill,base_dist,extras_params=core.build_ball_distributions(hist,resolved,{}, {})
    matchup_mods=core.build_matchup_adjustments(hist,resolved)
    # Apply modest, explicit surface priors to the empirical ball distributions.
    venue_delta=float(np.clip((venue_base-166.0)/100.0,-.12,.12))
    for player,dist in list(bdist.items()):
        q=np.asarray(dist,dtype=float).copy()
        if pitch=='batting': q[0]*=.94; q[4]*=1.05; q[6]*=1.08
        elif pitch=='bowling': q[0]*=1.05; q[4]*=.94; q[6]*=.90
        elif pitch=='spin': q[0]*=1.01; q[4]*=.99
        elif pitch=='pace': q[0]*=1.01; q[4]*=.99
        if venue_delta>0: q[0]*=(1-venue_delta*.18); q[4]*=(1+venue_delta*.10); q[6]*=(1+venue_delta*.12)
        elif venue_delta<0: q[0]*=(1+abs(venue_delta)*.16); q[4]*=(1-abs(venue_delta)*.08); q[6]*=(1-abs(venue_delta)*.10)
        bdist[player]=q/q.sum()
    for player,skill in wskill.items():
        if pitch=='bowling': skill['wicket_per_ball']*=1.04; skill['econ_per_ball']*=.98
        elif pitch=='batting': skill['wicket_per_ball']*=.97; skill['econ_per_ball']*=1.02
        elif pitch in ('spin','pace'): skill['wicket_per_ball']*=1.02
        if venue_delta>0: skill['econ_per_ball']*=1-venue_delta*.08
        elif venue_delta<0: skill['econ_per_ball']*=1+abs(venue_delta)*.08
    core._TLGE_OVER_CACHE=None
    policies={team1:core.CricketTLGEPolicy(xis[team1],hist,resolved,min_overs=2),
              team2:core.CricketTLGEPolicy(xis[team2],hist,resolved,min_overs=2)}
    strength=core.make_team_strength(xis)
    rng=np.random.default_rng(42); score1=[]; score2=[]; wickets1=[]; wickets2=[]; wins=Counter()
    bat_totals=defaultdict(Counter); bowl_totals=defaultdict(Counter)
    row=SimpleNamespace(match_id='INTERACTIVE_MATCH',stage='SCENARIO',date=pd.Timestamp.utcnow(),
                        venue=venue,team1=team1,team2=team2)
    for _ in range(int(iterations)):
        result=core.simulate_match(row,xis,bdist,wskill,rng,bat_totals,bowl_totals,strength,
                                   policies,matchup_mods,extras_params)
        score1.append(result['team1_score']); score2.append(result['team2_score'])
        wickets1.append(result['team1_wickets']); wickets2.append(result['team2_wickets'])
        wins[result['winner']]+=1
    bowling_plan=[]; tlge_meta={}
    for defending_team,xi in [(team1,xis[team1]),(team2,xis[team2])]:
        policy=policies[defending_team]
        eligible=xi[xi.role.isin(['BOWL','AR'])].Player.tolist()
        if not eligible: eligible=xi.Player.tolist()
        recs=[]
        for phase,state in [('Powerplay',18),('Middle overs',48),('Death overs',84)]:
            chosen,source=policy.choose(state,eligible,wskill,rng)
            recs.append({'phase':phase,'recommended_bowler':chosen or 'No supported recommendation','policy_source':source,'state':state})
        bowling_plan.append({'defending_team':defending_team,'plan':recs})
        tlge_meta[defending_team]={'training_overs':policy.training_overs,'supported_states':policy.supported_states,
                                   'bowlers_with_history':sum(v>0 for v in policy.action_overs.values())}
    def band(values):return [round(float(np.percentile(values,10)),1),round(float(np.percentile(values,90)),1)]
    bat_watch=[]
    for player,v in bat_totals.items():
        bat_watch.append({'player':player,'mean_runs_per_simulation':round(float(v['runs']/iterations),2),
                          'mean_balls_per_simulation':round(float(v['balls']/iterations),2)})
    bowl_watch=[]
    for player,v in bowl_totals.items():
        bowl_watch.append({'player':player,'mean_wickets_per_simulation':round(float(v['wickets']/iterations),2),
                           'mean_runs_conceded_per_simulation':round(float(v['runs']/iterations),2)})
    return {'team1':team1,'team2':team2,'team1_win_probability':round(wins[team1]/iterations,4),
            'team2_win_probability':round(wins[team2]/iterations,4),
            'team1_score_median':round(float(np.median(score1)),1),'team2_score_median':round(float(np.median(score2)),1),
            'team1_score_p10_p90':band(score1),'team2_score_p10_p90':band(score2),
            'team1_wickets_median':round(float(np.median(wickets1)),1),'team2_wickets_median':round(float(np.median(wickets2)),1),
            'iterations':int(iterations),'venue_context':venue_context,'xi1':a.to_dict('records'),'xi2':b.to_dict('records'),
            'matchups':matchup_report(team1,team2) if opponent_focus else [],'tlge_bowling_plan':bowling_plan,
            'tlge_support':tlge_meta,'player_contributions':{'batting':sorted(bat_watch,key=lambda x:x['mean_runs_per_simulation'],reverse=True)[:15],
                                                               'bowling':sorted(bowl_watch,key=lambda x:x['mean_wickets_per_simulation'],reverse=True)[:15]},
            'caveat':'Delivery-level empirical simulation using the retained IPL engine, historical player distributions, matchup adjustments and TL-GE-Entropy bowling policy. Win probabilities remain model scenarios, not calibrated forecasts. Venue and pitch priors are coarse and should be validated against held-out seasons.'}

def auction_shortlist(raw_players:pd.DataFrame, team, budget_balance='balanced', batting_bias=.58):
    d=history(); s=get_team(team); p=_scores_for_players(raw_players.rename(columns={'player':'player'}),d,batting_bias)
    current=_scores_for_players(s,d,batting_bias)
    need={'batting':max(0,55-current.bat_score.mean()),'bowling':max(0,55-current.bowl_score.mean())}
    p['need_fit']=p.apply(lambda r: r.bat_score*need['batting']+r.bowl_score*need['bowling'],axis=1)
    p['recommendation_score']=(p.score*.45+p.need_fit*.55)
    return p.sort_values('recommendation_score',ascending=False).head(100)

def tournament_sim(iterations=50,batting_bias=.58):
    if not FIXTURES_FILE.exists(): raise ValueError('Upload a fixtures CSV first.')
    fixtures=pd.read_csv(FIXTURES_FILE,parse_dates=['date'])
    teams=sorted(set(fixtures.team1)|set(fixtures.team2))
    if len(teams)<2: raise ValueError('At least two teams are required in the fixture file.')
    s=squad_frame()
    missing=sorted(set(teams)-set(s.team.unique()))
    if missing: raise ValueError('Squads missing for fixture teams: '+', '.join(missing))
    hist=history()
    if hist.empty: raise ValueError('Upload ball-by-ball history first.')
    # Stable name IDs allow direct exact-name matching where a registry crosswalk is unavailable.
    hist=hist.copy(); hist['batter_id']=hist.batter_id.fillna('').astype(str); hist['bowler_id']=hist.bowler_id.fillna('').astype(str)
    hist.loc[hist.batter_id.eq(''),'batter_id']=hist.loc[hist.batter_id.eq(''),'batter'].map(core.norm)
    hist.loc[hist.bowler_id.eq(''),'bowler_id']=hist.loc[hist.bowler_id.eq(''),'bowler'].map(core.norm)
    resolved=s.copy(); resolved['TeamCanonical']=resolved.team; resolved['Player']=resolved.player
    resolved['PlayerNorm']=resolved.player.map(core.norm); resolved['RoleCode']=resolved.role_code
    resolved['Overseas']=resolved.overseas; resolved['player_id']=resolved.PlayerNorm
    resolved['identity_status']='NAME_MATCH'; resolved['identity_method']='normalized_name'
    resolved['historical_names']=resolved.player
    xis={}
    for team in teams:
        selected=select_xi(team,batting_bias)
        selected=selected.rename(columns={'player':'Player'}).copy()
        xis[team]=selected
    bdist,wskill,base_dist,extras_params=core.build_ball_distributions(hist,resolved,{}, {})
    matchup_mods=core.build_matchup_adjustments(hist,resolved)
    core._TLGE_OVER_CACHE=None
    policies={t:core.CricketTLGEPolicy(xis[t],hist,resolved,min_overs=2) for t in teams}
    league=fixtures[~fixtures.stage.astype(str).str.upper().eq('PLAYOFF')].copy()
    league['match_id']=[f'FIXTURE_{i+1:03d}' for i in range(len(league))]
    league['stage']='LEAGUE'
    playoff=fixtures[fixtures.stage.astype(str).str.upper().eq('PLAYOFF')][['date','venue']].copy().reset_index(drop=True)
    playoff_slots=playoff if len(playoff)>=4 else None
    table_rows=[]; bat_rows=[]; bowl_rows=[]
    for i in range(int(iterations)):
        _,tab,bat,bowl=core.season_run(league,xis,bdist,wskill,seed=4200+i,playoff_slots=playoff_slots,
                                      simulation_index=i,tlge_policies=policies,matchup_mods=matchup_mods,extras_params=extras_params)
        tab['simulation']=i; table_rows.append(tab)
        if not bat.empty: bat['simulation']=i; bat_rows.append(bat)
        if not bowl.empty: bowl['simulation']=i; bowl_rows.append(bowl)
    alltab=pd.concat(table_rows,ignore_index=True)
    rows=[]
    for team,g in alltab.groupby('team'):
        rows.append({'team':team,'mean_points':round(float(g.points.mean()),2),'mean_wins':round(float(g.won.mean()),2),
                     'top4_probability':round(float(g.qualified_top4.mean()),3),'mean_nrr':round(float(g.nrr.mean()),3),
                     'p10_points':round(float(g.points.quantile(.10)),1),'p90_points':round(float(g.points.quantile(.90)),1),
                     'mean_position':round(float(g.position.mean()),2)})
    rows.sort(key=lambda x:(x['top4_probability'],x['mean_points'],x['mean_nrr']),reverse=True)
    bat_all=pd.concat(bat_rows,ignore_index=True) if bat_rows else pd.DataFrame()
    bowl_all=pd.concat(bowl_rows,ignore_index=True) if bowl_rows else pd.DataFrame()
    orange=[]; purple=[]
    if not bat_all.empty:
        for player,g in bat_all.groupby('player'):
            vals=g.groupby('simulation').runs.sum()
            orange.append({'player':player,'projected_runs':round(float(vals.mean()),1),'runs_p10':round(float(vals.quantile(.10)),1),'runs_p90':round(float(vals.quantile(.90)),1)})
    if not bowl_all.empty:
        for player,g in bowl_all.groupby('player'):
            vals=g.groupby('simulation').wickets.sum()
            purple.append({'player':player,'projected_wickets':round(float(vals.mean()),1),'wickets_p10':round(float(vals.quantile(.10)),1),'wickets_p90':round(float(vals.quantile(.90)),1)})
    return {'table':rows,'orange_cap':sorted(orange,key=lambda x:x['projected_runs'],reverse=True)[:20],
            'purple_cap':sorted(purple,key=lambda x:x['projected_wickets'],reverse=True)[:20],
            'simulations':int(iterations),'fixtures':int(len(league)),'playoff_slots_simulated':int(4 if playoff_slots is not None else 0),
            'algorithm':'Legacy empirical match simulator + TL-GE-Entropy support-gated bowling policy + historical batter/bowler matchup adjustments.',
            'note':'Each iteration simulates the uploaded league fixtures; the playoff bracket is simulated only when four playoff-slot rows are supplied. Output remains a scenario distribution, not a guaranteed forecast.'}

