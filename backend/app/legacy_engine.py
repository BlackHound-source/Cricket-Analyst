#!/usr/bin/env python3
"""
IPL CRICKET INTELLIGENCE — DECISION-SUPPORT ALGORITHM ONLY
====================================================================
Training boundary: 2025-12-31 (strict).

Inputs:
  * Historical Cricsheet JSON/ZIPs and/or Total_data.zip under --input-dir
  * 10-team squad CSV (Team, Player, Role, Country, Type)
  * Optional v13 results ZIP containing 2026_HOLDOUT_MATCHES.csv. Only fixture
    metadata (date, teams, venue) is read for the pre-season simulation.
    Actual toss, outcome, innings, deliveries and player statistics are not
    exposed to the prediction engine. They are read only by the evaluator
    after predictions have been frozen.

This is a moderate-criticality decision-support simulation. It is NOT a
validated betting, win-probability or autonomous selection system.

Notebook use: import this file and call notebook_main(...). The script does not
call IPython exit helpers, sys.exit(), or quit().
"""
from __future__ import annotations
import argparse, difflib, hashlib, io, json, math, sys, time, unicodedata, zipfile
import re
from collections import defaultdict, Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

SEED = 42
CUTOFF = pd.Timestamp("2025-12-31")

# Canonical TL-GE-Entropy class copied unchanged from the v13 in-house implementation.
N_STATES = 108
N_ACTIONS = 2


class TLGEConfig:
    def __init__(self, lr=0.10, gamma=0.95, beta=0.80, alpha_max=1.00):
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.alpha_max = alpha_max


class TLGEEntropy:
    """
    Canonical TL-GE-Entropy implementation.

    Active lane:    V_A(s,a) = Q(s,a) - D_A(s,a)
    Inactive lane:  V_I(s,a) = Q(s,a) - D_I(s,a) + U(s,a)

    Action-specific Laplace-smoothed entropy:
        p_hat(s,a) = (N_1(s,a) + 1) / (N(s,a) + 2)
        H(s,a)     = -p log2(p) - (1-p) log2(1-p)

    Entropy-adaptive uncertainty:
        alpha_U(s,a) = alpha_max * H(s,a)
        U(s,a)       = alpha_U(s,a) / sqrt(N(s,a) + 1)

    TL challenger:  a_TL = argmax_{a != a_A} V_I(s,a)
    TL disagreement: Delta_TL = V_I(s,a_TL) - V_A(s,a_A)
    Trigger:        Delta_TL > 0

    GE: G_hat(s,a) = min observed ACTUAL rewards for (s,a)

    Critical causal rule:
        The lane role is captured BEFORE action execution.
        The TD deviation is then written only to the memory
        corresponding to that role. D_A and D_I are never
        simultaneously updated.

    Critical entropy rule:
        Entropy is action-specific H(s,a), never pooled over actions.
    """

    def __init__(self, config=None, n_states=N_STATES, n_actions=N_ACTIONS):
        self.cfg = config or TLGEConfig()
        self.n_states = n_states
        self.n_actions = n_actions

        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.d_active = np.zeros((n_states, n_actions), dtype=np.float64)
        self.d_inactive = np.zeros((n_states, n_actions), dtype=np.float64)
        self.active_action = np.zeros(n_states, dtype=np.int64)

        self.reward_count = np.zeros((n_states, n_actions), dtype=np.int64)
        self.reward_mean = np.zeros((n_states, n_actions), dtype=np.float64)
        self.guaranteed = np.full((n_states, n_actions), np.inf, dtype=np.float64)

        self.outcome_ones = np.zeros((n_states, n_actions), dtype=np.int64)
        self.outcome_total = np.zeros((n_states, n_actions), dtype=np.int64)

        self.last_tl_delta = 0.0
        self.last_tl_trigger = 0
        self.last_tl_action = -1
        self.last_active_action = -1
        self.last_selected_action = -1
        self.last_source = "active"
        self.last_entropy = 0.0
        self.last_alpha = 0.0
        self.last_uncertainty = 0.0
        self.last_active_value = 0.0
        self.last_candidate_value = 0.0
        self.last_guaranteed = np.nan
        self.last_decision_lane = "active"

    # ---------- ACTION-SPECIFIC ENTROPY ----------
    def entropy(self, s, a):
        n1 = int(self.outcome_ones[s, a])
        n = int(self.outcome_total[s, a])
        p = (n1 + 1.0) / (n + 2.0)
        if p <= 0.0 or p >= 1.0:
            return 0.0
        h = -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)
        return float(np.clip(h, 0.0, 1.0))

    # ---------- ENTROPY-ADAPTIVE UNCERTAINTY ----------
    def uncertainty(self, s, a):
        h = self.entropy(s, a)
        alpha_u = self.cfg.alpha_max * h
        count = int(self.reward_count[s, a])
        return float(alpha_u / math.sqrt(count + 1.0))

    # ---------- LANE VALUES ----------
    def active_value(self, s, a):
        return float(self.q[s, a] - self.d_active[s, a])

    def inactive_value(self, s, a):
        return float(self.q[s, a] - self.d_inactive[s, a] + self.uncertainty(s, a))

    # ---------- GUARANTEED EXPLORATION ----------
    def select_ge_action(self, s, candidates):
        """
        Unseen candidates are chosen by inactive-lane value. If every candidate
        has been observed, the primary criterion is the empirical minimum of
        ACTUAL observed rewards; V_I is only a deterministic tie-breaker.
        """
        candidates = list(candidates)
        unseen = [a for a in candidates if self.reward_count[s, a] == 0]
        if unseen:
            return int(max(unseen, key=lambda a: (self.inactive_value(s, a), -a)))
        return int(max(candidates, key=lambda a: (
            self.guaranteed[s, a], self.inactive_value(s, a), -a)))

    # ---------- ACTION SELECTION ----------
    def act(self, s):
        active = int(self.active_action[s])
        inactive = [a for a in range(self.n_actions) if a != active]

        self.last_tl_trigger = 0
        self.last_source = "active"
        self.last_selected_action = active
        self.last_active_action = active
        self.last_tl_action = -1
        self.last_guaranteed = np.nan
        self.last_decision_lane = "active"

        if not inactive:
            self.last_tl_delta = 0.0
            self.last_entropy = 0.0
            self.last_alpha = 0.0
            self.last_uncertainty = 0.0
            self.last_active_value = self.active_value(s, active)
            self.last_candidate_value = self.last_active_value
            return active

        candidate = int(max(inactive, key=lambda a: (self.inactive_value(s, a), -a)))

        active_value = self.active_value(s, active)
        candidate_value = self.inactive_value(s, candidate)
        delta = candidate_value - active_value

        candidate_entropy = self.entropy(s, candidate)
        candidate_alpha = self.cfg.alpha_max * candidate_entropy
        candidate_uncertainty = self.uncertainty(s, candidate)

        self.last_tl_delta = float(delta)
        self.last_tl_action = candidate
        self.last_active_action = active
        self.last_active_value = float(active_value)
        self.last_candidate_value = float(candidate_value)
        self.last_entropy = float(candidate_entropy)
        self.last_alpha = float(candidate_alpha)
        self.last_uncertainty = float(candidate_uncertainty)

        trigger = int(delta > 0.0)
        self.last_tl_trigger = trigger

        if trigger:
            selected = self.select_ge_action(s, inactive)
            self.active_action[s] = selected
            self.last_selected_action = selected
            self.last_source = "GE"
            self.last_decision_lane = "inactive"
            if self.reward_count[s, selected] > 0:
                self.last_guaranteed = float(self.guaranteed[s, selected])
            return selected

        self.last_selected_action = active
        self.last_source = "active"
        self.last_decision_lane = "active"
        return active

    # ---------- UPDATE ----------
    def update(self, s, a, r, ns, outcome, decision_lane):
        """
        Update after observing (r, ns, outcome). decision_lane is the lane
        occupied by (s,a) at decision time; only that lane's deviation memory
        is updated.
        """
        if decision_lane not in ("active", "inactive"):
            raise ValueError("decision_lane must be 'active' or 'inactive'")

        target = r + self.cfg.gamma * np.max(self.q[ns])
        td_error = target - self.q[s, a]
        self.q[s, a] += self.cfg.lr * td_error

        deviation = abs(td_error)
        if decision_lane == "active":
            self.d_active[s, a] = (self.cfg.beta * self.d_active[s, a]
                                   + (1.0 - self.cfg.beta) * deviation)
        else:
            self.d_inactive[s, a] = (self.cfg.beta * self.d_inactive[s, a]
                                     + (1.0 - self.cfg.beta) * deviation)

        count = int(self.reward_count[s, a])
        self.reward_count[s, a] = count + 1
        self.reward_mean[s, a] += (r - self.reward_mean[s, a]) / (count + 1)

        # Empirical guaranteed reward uses ACTUAL reward, not Q-value.
        if count == 0:
            self.guaranteed[s, a] = float(r)
        else:
            self.guaranteed[s, a] = min(self.guaranteed[s, a], float(r))

        self.outcome_total[s, a] += 1
        self.outcome_ones[s, a] += int(outcome)

    # ---------- DIAGNOSTICS ----------
    def diagnostics(self):
        return {
            "Entropy": self.last_entropy,
            "AlphaU": self.last_alpha,
            "TL_Delta": self.last_tl_delta,
            "TL_Trigger": self.last_tl_trigger,
            "TL_Candidate": self.last_tl_action,
            "ActiveAction": self.last_active_action,
            "SelectedAction": self.last_selected_action,
            "Uncertainty": self.last_uncertainty,
            "ActiveValue": self.last_active_value,
            "CandidateValue": self.last_candidate_value,
            "GuaranteedReward": self.last_guaranteed,
            "Source": self.last_source,
            "DecisionLane": self.last_decision_lane,
        }


TEAM_ALIASES = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "rcb": "Royal Challengers Bengaluru",
    "gujarat titans": "Gujarat Titans", "gt": "Gujarat Titans",
    "chennai super kings": "Chennai Super Kings", "csk": "Chennai Super Kings",
    "delhi capitals": "Delhi Capitals", "dc": "Delhi Capitals",
    "kolkata knight riders": "Kolkata Knight Riders", "kkr": "Kolkata Knight Riders",
    "lucknow super giants": "Lucknow Super Giants", "lsg": "Lucknow Super Giants",
    "mumbai indians": "Mumbai Indians", "mi": "Mumbai Indians",
    "punjab kings": "Punjab Kings", "pbks": "Punjab Kings", "kxip": "Punjab Kings",
    "rajasthan royals": "Rajasthan Royals", "rr": "Rajasthan Royals",
    "sunrisers hyderabad": "Sunrisers Hyderabad", "srh": "Sunrisers Hyderabad",
}

ROLE_MAP = {
    "batter": "BAT", "batsman": "BAT", "bat": "BAT",
    "bowler": "BOWL", "bowling": "BOWL",
    "all-rounder": "AR", "all rounder": "AR", "allrounder": "AR",
    "wicketkeeper-batter": "WK", "wicketkeeper batter": "WK",
    "wicketkeeper": "WK", "wk": "WK",
}


def norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\b(jr|sr|ii|iii)\b", " ", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def team_name(s):
    n = norm(s)
    return TEAM_ALIASES.get(n, str(s).strip())


def safe_json(v):
    try:
        return json.loads(v) if isinstance(v, str) and v.strip() else {}
    except Exception:
        return {}


def discover_files(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".zip", ".json", ".csv", ".xlsx"}:
            files.append(p)
    return files


def walk_zip_json(zf: zipfile.ZipFile, prefix=""):
    """Yield JSON bytes from nested ZIPs; skip corrupt members and cap recursion."""
    for name in zf.namelist():
        low = name.lower()
        if low.endswith(".json"):
            try:
                yield name, zf.read(name)
            except Exception:
                continue
        elif low.endswith(".zip"):
            try:
                raw = zf.read(name)
                with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                    yield from walk_zip_json(nested, prefix + name + "!")
            except Exception:
                continue


def iter_json_files(root: Path):
    seen = set()
    for p in root.rglob("*.json"):
        try:
            b = p.read_bytes()
            key = hashlib.sha256(b).hexdigest()
            if key not in seen:
                seen.add(key)
                yield str(p), b
        except Exception:
            pass
    for p in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(p) as zf:
                for n, b in walk_zip_json(zf):
                    key = hashlib.sha256(b).hexdigest()
                    if key not in seen:
                        seen.add(key)
                        yield f"{p.name}!{n}", b
        except Exception:
            continue


def load_identity_crosswalk(root: Path):
    alias_to_id = defaultdict(set)
    id_to_names = defaultdict(set)
    for p in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(p) as zf:
                for n in zf.namelist():
                    if n.lower().endswith(("people.csv", "names.csv")):
                        try:
                            d = pd.read_csv(io.BytesIO(zf.read(n)))
                        except Exception:
                            continue
                        cols = {c.lower(): c for c in d.columns}
                        idc = cols.get("identifier")
                        if not idc:
                            continue
                        namecols = [cols[c] for c in ("name", "unique_name") if c in cols]
                        for _, r in d.iterrows():
                            pid = str(r[idc]).strip()
                            if not pid or pid.lower() == "nan":
                                continue
                            for c in namecols:
                                nm = str(r[c]).strip()
                                if nm and nm.lower() != "nan":
                                    alias_to_id[norm(nm)].add(pid)
                                    id_to_names[pid].add(nm)
        except Exception:
            pass
    # Also read standalone crosswalk CSVs.
    for p in root.rglob("*.csv"):
        if p.name.lower() not in {"people.csv", "names.csv"}:
            continue
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in d.columns}
        idc = cols.get("identifier")
        if not idc:
            continue
        for _, r in d.iterrows():
            pid = str(r[idc]).strip()
            for c in [cols[x] for x in ("name", "unique_name") if x in cols]:
                nm = str(r[c]).strip()
                if pid and pid.lower() != "nan" and nm and nm.lower() != "nan":
                    alias_to_id[norm(nm)].add(pid)
                    id_to_names[pid].add(nm)
    return alias_to_id, id_to_names


def parse_match(obj, source):
    info = obj.get("info", {})
    if str(info.get("gender", "male")).lower() != "male":
        return None
    match_type = str(info.get("match_type", "")).upper()
    if match_type not in {"T20", "IT20"}:
        return None
    if int(info.get("overs", 20) or 20) > 20:
        return None
    teams = info.get("teams", [])
    if len(teams) != 2:
        return None
    dates = info.get("dates", [])
    try:
        date = pd.to_datetime(dates[0], errors="coerce")
    except Exception:
        date = pd.NaT
    if pd.isna(date) or date > CUTOFF:
        return None
    comp = info.get("event", {}).get("name", "") if isinstance(info.get("event"), dict) else ""
    src = source.lower()
    if "ipl" in src:
        comp = "IPL"
    if not comp:
        comp = "UNKNOWN"
    registry = info.get("registry", {}).get("people", {}) or {}
    player_ids = {norm(k): v for k, v in registry.items()}
    innings = obj.get("innings", [])
    rows = []
    for inn_no, inn in enumerate(innings, 1):
        batting = inn.get("team")
        if batting not in teams:
            continue
        bowling = next((t for t in teams if t != batting), "")
        for over in inn.get("overs", []):
            over_no = int(over.get("over", 0))
            for delivery_idx, d in enumerate(over.get("deliveries", [])):
                batter = d.get("batter", "")
                bowler = d.get("bowler", "")
                runs = d.get("runs", {}) or {}
                extras = d.get("extras", {}) or {}
                total = int(runs.get("total", 0))
                br = int(runs.get("batter", 0))
                wides = int(extras.get("wides", 0) or 0)
                noballs = int(extras.get("noballs", 0) or 0)
                legal = (wides == 0 and noballs == 0)
                wickets = d.get("wickets", []) or []
                credited = sum(1 for w in wickets if w.get("kind") not in
                               {"run out", "retired hurt", "obstructing the field", "retired out"})
                batter_out = sum(1 for w in wickets if w.get("player_out") == batter)
                rows.append({
                    "date": date, "competition": comp, "season": str(info.get("season", "")),
                    "venue": info.get("venue", ""), "city": info.get("city", ""),
                    "match_key": f"{comp}:{date.date()}:{'|'.join(sorted(teams))}:{source.split('!')[-1]}",
                    "innings": inn_no, "over": over_no, "delivery_idx": delivery_idx,
                    "batting_team": batting, "bowling_team": bowling,
                    "batter": batter, "bowler": bowler,
                    "batter_id": player_ids.get(norm(batter), ""),
                    "bowler_id": player_ids.get(norm(bowler), ""),
                    "batter_runs": br, "total_runs": total,
                    "runs_conceded": total - int(extras.get("byes", 0) or 0)
                                     - int(extras.get("legbyes", 0) or 0)
                                     - int(extras.get("penalty", 0) or 0),
                    "legal": int(legal), "batter_ball": int(wides == 0),
                    "wides": wides, "noballs": noballs, "wicket": int(batter_out > 0),
                    "bowler_wickets": credited, "is_four": int(br == 4), "is_six": int(br == 6),
                    "over_number": over_no,
                })
    if not rows:
        return None
    return rows


def load_ipl_csv_from_archives(root: Path):
    """Load the supplied canonical IPL ball-by-ball CSV through 2025, including nested ZIPs."""
    payload = None

    def search_zip(zf, depth=0):
        nonlocal payload
        if depth > 4 or payload is not None:
            return
        for name in zf.namelist():
            low = name.lower()
            try:
                raw = zf.read(name)
                if low.endswith("deliveries_updated_ipl_upto_2025.csv"):
                    payload = raw
                    return
                if low.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                        search_zip(nested, depth + 1)
                if payload is not None:
                    return
            except Exception:
                continue

    for zp in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                search_zip(zf)
        except Exception:
            continue
        if payload is not None:
            break
    if payload is None:
        return None

    d = pd.read_csv(io.BytesIO(payload))
    cols = {c.lower(): c for c in d.columns}

    def col(name, default=None):
        return d[cols[name]] if name in cols else pd.Series(default, index=d.index)

    date = pd.to_datetime(col("date"), errors="coerce")
    date2 = pd.to_datetime(col("start_date"), errors="coerce")
    date = date.fillna(date2)
    d = d.loc[date.notna() & (date <= CUTOFF)].copy()
    date = date.loc[d.index]
    batsman = col("batsman").loc[d.index].fillna("").astype(str)
    bowler = col("bowler").loc[d.index].fillna("").astype(str)
    runs = pd.to_numeric(col("batsman_runs", 0).loc[d.index], errors="coerce").fillna(0).astype(int)
    extras = pd.to_numeric(col("extras", 0).loc[d.index], errors="coerce").fillna(0).astype(int)
    wides = pd.to_numeric(col("iswide", col("wides", 0)).loc[d.index], errors="coerce").fillna(0)
    noballs = pd.to_numeric(col("isnoball", col("noballs", 0)).loc[d.index], errors="coerce").fillna(0)
    # Some feeds use boolean flags; others use numeric extra counts.
    wide_flag = wides.astype(str).str.lower().isin(["1", "1.0", "true", "yes"]) | (wides > 0)
    nb_flag = noballs.astype(str).str.lower().isin(["1", "1.0", "true", "yes"]) | (noballs > 0)
    wicket_kind = col("dismissal_kind", col("wicket_type", "")).loc[d.index].fillna("").astype(str).str.lower()
    player_out = col("player_dismissed", "").loc[d.index].fillna("").astype(str)
    non_credit = {"run out", "retired hurt", "obstructing the field", "retired out"}
    wicket = player_out.ne("") & (player_out.str.lower() != "nan")
    credited = wicket & ~wicket_kind.isin(non_credit)
    matchid = col("matchid", col("match_id", "")).loc[d.index].astype(str)
    inn = pd.to_numeric(col("inning", col("innings", 1)).loc[d.index], errors="coerce").fillna(1).astype(int)
    over = pd.to_numeric(col("over", 0).loc[d.index], errors="coerce").fillna(0).astype(int)
    venue = col("venue", "").loc[d.index].fillna("").astype(str)
    batting = col("batting_team", "").loc[d.index].fillna("").astype(str)
    bowling = col("bowling_team", "").loc[d.index].fillna("").astype(str)
    byes = pd.to_numeric(col("byes", 0).loc[d.index], errors="coerce").fillna(0)
    legbyes = pd.to_numeric(col("legbyes", 0).loc[d.index], errors="coerce").fillna(0)
    penalty = pd.to_numeric(col("penalty", 0).loc[d.index], errors="coerce").fillna(0)

    out = pd.DataFrame({
        "date": date.values, "competition": "IPL",
        "season": col("season", "").loc[d.index].astype(str).values,
        "venue": venue.values, "city": "",
        "match_key": ["IPL:" + x for x in matchid],
        "innings": inn.values, "over": over.values, "delivery_idx": np.arange(len(d)),
        "batting_team": batting.values, "bowling_team": bowling.values,
        "batter": batsman.values, "bowler": bowler.values,
        "batter_id": "", "bowler_id": "",
        "batter_runs": runs.values,
        "total_runs": (runs + extras).values,
        "runs_conceded": (runs + extras - byes - legbyes - penalty).values,
        "legal": (~(wide_flag | nb_flag)).astype(int).values,
        "batter_ball": (~wide_flag).astype(int).values,
        "wides": wide_flag.astype(int).values,
        "noballs": nb_flag.astype(int).values,
        "wicket": (wicket & player_out.eq(batsman)).astype(int).values,
        "bowler_wickets": credited.astype(int).values,
        "is_four": (runs == 4).astype(int).values,
        "is_six": (runs == 6).astype(int).values,
        "over_number": over.values,
    })
    out = out.drop_duplicates(["match_key", "innings", "over", "delivery_idx"])
    return out


def load_historical(root: Path, max_json=None):
    allrows = []
    parsed = 0
    bad = 0
    for source, raw in iter_json_files(root):
        try:
            obj = json.loads(raw)
            rows = parse_match(obj, source)
            if rows:
                allrows.extend(rows)
                parsed += 1
        except Exception:
            bad += 1
        if max_json and parsed >= max_json:
            break
    json_df = pd.DataFrame(allrows) if allrows else pd.DataFrame()
    ipl_csv = load_ipl_csv_from_archives(root)
    if ipl_csv is not None and not ipl_csv.empty:
        # The canonical IPL CSV supersedes IPL JSON coverage and extends through 2025.
        if not json_df.empty:
            json_df = json_df[~json_df.competition.astype(str).str.contains("IPL", case=False)]
        d = pd.concat([json_df, ipl_csv], ignore_index=True, sort=False)
    else:
        d = json_df
    if d.empty:
        raise RuntimeError("No eligible pre-2026 male T20 delivery records found.")
    d = d.drop_duplicates(["match_key", "innings", "over", "delivery_idx"], keep="first")
    d["is_ipl"] = d.competition.astype(str).str.contains("ipl", case=False)
    return d, {
        "parsed_matches": int(d.match_key.nunique()),
        "delivery_rows": len(d),
        "bad_or_skipped_json": bad,
        "canonical_ipl_csv_rows": int(d.is_ipl.sum()),
        "first_date": str(d.date.min().date()),
        "last_date": str(d.date.max().date()),
        "ipl_rows": int(d.is_ipl.sum()),
        "competitions": sorted(d.competition.unique().tolist()),
    }




def find_squad_file(root: Path, explicit=None):
    """Find the canonical all-team 2026 squad CSV, preferring the most complete copy."""
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(explicit)
        return p
    hits = []
    for base in [Path(root)]:
        if base.exists():
            hits.extend(p for p in base.rglob("*.csv")
                        if "squad" in p.name.lower() and p not in hits)
    if not hits:
        raise FileNotFoundError(
            "Could not find a squad CSV under " + str(root) + ". Add the ipl_2026_all_teams_squad dataset to the notebook."
        )
    scored = []
    for p in hits:
        try:
            d = pd.read_csv(p, usecols=["Team", "Player"])
            valid = d.Team.nunique() == 10 and d.Player.nunique() >= 100
            scored.append((int(valid), len(d), str(p), p))
        except Exception:
            scored.append((0, 0, str(p), p))
    return max(scored, key=lambda x: (x[0], x[1], x[2]))[3]


def load_squads(path: Path):
    s = pd.read_csv(path)
    required = {"Team", "Player", "Role", "Country", "Type"}
    if not required.issubset(s.columns):
        raise ValueError(f"Squad CSV needs columns {sorted(required)}; found {list(s.columns)}")
    s = s.dropna(subset=["Team", "Player"]).copy()
    s["TeamCanonical"] = s.Team.map(team_name)
    s["PlayerNorm"] = s.Player.map(norm)
    s["RoleCode"] = s.Role.map(lambda x: ROLE_MAP.get(norm(x).replace(" ", "-"), ROLE_MAP.get(norm(x), "UNKNOWN")))
    if (s.RoleCode == "UNKNOWN").any():
        raise ValueError("Unknown squad role(s): " + str(s.loc[s.RoleCode == "UNKNOWN", "Role"].unique().tolist()))
    s["Overseas"] = s.Type.astype(str).str.contains("overseas", case=False) | \
                    (~s.Country.astype(str).str.lower().isin(["india", "nan", ""]))
    s = s.drop_duplicates(["TeamCanonical", "PlayerNorm"])
    if s.TeamCanonical.nunique() != 10:
        raise ValueError(f"Expected 10 teams, found {s.TeamCanonical.nunique()}: {s.TeamCanonical.unique().tolist()}")
    expected_teams = {
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    }
    if set(s.TeamCanonical) != expected_teams:
        raise ValueError("Squad CSV must contain the canonical ten IPL franchises.")
    if s.duplicated(["TeamCanonical", "PlayerNorm"]).any():
        raise ValueError("Duplicate player within a team in squad CSV.")
    counts = s.groupby("TeamCanonical").PlayerNorm.nunique()
    if (counts < 11).any():
        raise ValueError("Every franchise must have at least 11 distinct squad players: " + str(counts.to_dict()))
    if s.Country.fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every squad row must include a player country.")
    return s


def resolve_identities(squads, alias_to_id, id_to_names):
    # Exact alias only if unique. Fuzzy matches are proposals, accepted only with
    # a high score and a clear margin over the runner-up.
    all_alias = list(alias_to_id.keys())
    rows = []
    for _, r in squads.iterrows():
        q = r.PlayerNorm
        exact = alias_to_id.get(q, set())
        pid = ""
        status = "NO_HISTORY"
        method = "none"
        candidate = ""
        score = 0.0
        if len(exact) == 1:
            pid = next(iter(exact))
            status = "MATCHED"
            method = "exact_crosswalk"
            score = 1.0
        elif len(exact) > 1:
            status = "AMBIGUOUS"
            method = "non_unique_exact"
        else:
            qt = q.split()
            alias_ids = set()
            if len(qt) >= 2:
                for a in all_alias:
                    at = a.split()
                    if len(at) >= 2 and at[-1] == qt[-1]:
                        if at[-len(qt):] == qt or (at[0][:1] == qt[0][:1] and at[-1] == qt[-1]):
                            alias_ids.update(alias_to_id.get(a, set()))
            if len(alias_ids) == 1:
                pid = next(iter(alias_ids))
                status = "ALIAS_MATCHED"
                method = "initial_surname_or_suffix_alias"
                score = .96
            elif len(alias_ids) > 1:
                status = "AMBIGUOUS"
                method = "initial_surname_collision"
            scored = difflib.get_close_matches(q, all_alias, n=5, cutoff=0.90)
            candidates = []
            for a in scored:
                ids = alias_to_id.get(a, set())
                if len(ids) == 1:
                    ratio = difflib.SequenceMatcher(None, q, a).ratio()
                    candidates.append((ratio, a, next(iter(ids))))
            candidates.sort(reverse=True)
            if status == "NO_HISTORY" and candidates:
                score, candidate, cid = candidates[0]
                margin = score - (candidates[1][0] if len(candidates) > 1 else 0)
                if score >= 0.94 and margin >= 0.035:
                    pid = cid
                    status = "ALIAS_MATCHED"
                    method = "high_confidence_fuzzy_alias"
                else:
                    status = "AMBIGUOUS"
                    method = "fuzzy_candidate_not_accepted"
        rows.append({**r.to_dict(), "player_id": pid, "identity_status": status,
                     "identity_method": method, "identity_candidate": candidate,
                     "identity_score": score,
                     "historical_names": " | ".join(sorted(id_to_names.get(pid, set()))) if pid else ""})
    return pd.DataFrame(rows)


def make_profiles(d):
    # IPL evidence is primary; cross-league fallback is discounted and flagged.
    def aggregate(x):
        g = x.groupby("batter").agg(
            matches=("match_key", "nunique"), balls=("batter_ball", "sum"),
            runs=("batter_runs", "sum"), outs=("wicket", "sum"),
            fours=("is_four", "sum"), sixes=("is_six", "sum"),
        ).reset_index()
        g["sr"] = 100 * g.runs / g.balls.clip(lower=1)
        g["avg"] = g.runs / g.outs.clip(lower=1)
        g["boundary_rate"] = (g.fours + g.sixes) / g.balls.clip(lower=1)
        return g

    bat = aggregate(d[d.is_ipl])
    bowl = d[d.is_ipl].groupby("bowler").agg(
        matches=("match_key", "nunique"), balls=("legal", "sum"),
        runs=("runs_conceded", "sum"), wickets=("bowler_wickets", "sum"),
    ).reset_index()
    bowl["overs"] = bowl.balls / 6
    bowl["econ"] = bowl.runs / bowl.overs.clip(lower=.1)
    bowl["wicket_rate"] = bowl.wickets / bowl.balls.clip(lower=1)

    xb = aggregate(d[~d.is_ipl])
    xw = d[~d.is_ipl].groupby("bowler").agg(
        matches=("match_key", "nunique"), balls=("legal", "sum"),
        runs=("runs_conceded", "sum"), wickets=("bowler_wickets", "sum"),
    ).reset_index()
    xw["overs"] = xw.balls / 6
    xw["econ"] = xw.runs / xw.overs.clip(lower=.1)
    xw["wicket_rate"] = xw.wickets / xw.balls.clip(lower=1)

    batmap = {}
    bowlmap = {}
    for _, r in bat.iterrows():
        batmap[norm(r.batter)] = {**r.to_dict(), "evidence": "IPL", "discount": 1.0}
    for _, r in bowl.iterrows():
        bowlmap[norm(r.bowler)] = {**r.to_dict(), "evidence": "IPL", "discount": 1.0}
    for _, r in xb.iterrows():
        k = norm(r.batter)
        if k not in batmap:
            batmap[k] = {**r.to_dict(), "evidence": "CROSS_LEAGUE", "discount": .65}
    for _, r in xw.iterrows():
        k = norm(r.bowler)
        if k not in bowlmap:
            bowlmap[k] = {**r.to_dict(), "evidence": "CROSS_LEAGUE", "discount": .65}
    return batmap, bowlmap, bat, bowl


def attach_profile_aliases(resolved, batmap, bowlmap, alias_to_id, d):
    # Match historical appearance names to player ids from each delivery's registry.
    id_bat = defaultdict(list)
    id_bowl = defaultdict(list)
    for _, r in d.drop_duplicates(["batter", "batter_id"]).iterrows():
        if r.batter_id:
            id_bat[str(r.batter_id)].append(norm(r.batter))
    for _, r in d.drop_duplicates(["bowler", "bowler_id"]).iterrows():
        if r.bowler_id:
            id_bowl[str(r.bowler_id)].append(norm(r.bowler))
    bprof = {}
    wprof = {}
    for _, r in resolved.iterrows():
        pid = r.player_id
        bkeys = set(id_bat.get(str(pid), [])) if pid else {r.PlayerNorm}
        wkeys = set(id_bowl.get(str(pid), [])) if pid else {r.PlayerNorm}
        bp = next((batmap[k] for k in bkeys if k in batmap), None)
        wp = next((bowlmap[k] for k in wkeys if k in bowlmap), None)
        bprof[r.PlayerNorm] = bp
        wprof[r.PlayerNorm] = wp
    return bprof, wprof


def choose_xi(team_df, bprof, wprof):
    # Transparent heuristic with conservative missing-history priors.
    rows = []
    for _, r in team_df.iterrows():
        bp = bprof.get(r.PlayerNorm)
        wp = wprof.get(r.PlayerNorm)
        bsr = float(bp.get("sr", 125)) if bp else 125.0
        bavg = float(bp.get("avg", 22)) if bp else 22.0
        bexp = float(bp.get("balls", 0)) if bp else 0
        econ = float(wp.get("econ", 8.7)) if wp else 8.7
        wr = float(wp.get("wicket_rate", .018)) if wp else .018
        wexp = float(wp.get("balls", 0)) if wp else 0
        # Smooth rates toward neutral priors to reduce small-sample extremes.
        bsr_adj = (bsr * bexp + 125 * 120) / (bexp + 120)
        bavg_adj = (bavg * bexp + 22 * 120) / (bexp + 120)
        econ_adj = (econ * wexp + 8.7 * 120) / (wexp + 120)
        wr_adj = (wr * wexp + .018 * 120) / (wexp + 120)
        batting = np.clip(.48 * (bsr_adj / 150 * 100) + .32 * (bavg_adj / 40 * 100)
                          + .20 * min(100, ((bp or {}).get("boundary_rate", .15)) * 350), 0, 100)
        bowling = np.clip(.58 * (10.5 - econ_adj) / 4 * 100 + .42 * (wr_adj / .03 * 100), 0, 100)
        role = r.RoleCode
        rolebonus = {"BAT": 9, "WK": 8, "AR": 10, "BOWL": 8}[role]
        expbonus = min(10, math.log1p(max(bexp, wexp)) / 2)
        score = .55 * batting + .35 * bowling + rolebonus + expbonus
        rows.append({
            "Player": r.Player, "role": role, "overseas": bool(r.Overseas), "score": score,
            "bat_score": batting, "bowl_score": bowling,
            "bat_evidence": (bp or {}).get("evidence", "NO_HISTORY"),
            "bowl_evidence": (wp or {}).get("evidence", "NO_HISTORY"),
            "historical_balls_faced": bexp, "historical_bowling_balls": wexp,
            "identity_status": r.identity_status, "player_id": r.player_id,
        })
    p = pd.DataFrame(rows).sort_values("score", ascending=False)

    # Enumerate feasible XIs from a reduced top pool; constraints checked exactly.
    pool = p.head(min(18, len(p))).to_dict("records")
    best = None
    bestscore = -1e9
    chosen = []
    for role in ["WK", "BAT", "AR", "BOWL"]:
        need = {"WK": 1, "BAT": 3, "AR": 1, "BOWL": 3}[role]
        for x in [q for q in pool if q["role"] == role]:
            if len([a for a in chosen if a["Player"] == x["Player"]]) == 0 and \
               sum(a["role"] == role for a in chosen) < need:
                chosen.append(x)
    for x in pool:
        if len(chosen) >= 11:
            break
        if x not in chosen:
            chosen.append(x)
    if len(chosen) < 11:
        chosen = p.head(11).to_dict("records")

    def valid_base(xs):
        return (len(xs) == 11 and sum(x["overseas"] for x in xs) <= 4 and
                1 <= sum(x["role"] == "WK" for x in xs) <= 2 and
                sum(x["role"] in ("BAT", "WK", "AR") for x in xs) >= 6 and
                sum(x["role"] in ("BOWL", "AR") for x in xs) >= 5 and
                sum(x["role"] == "BAT" for x in xs) >= 3 and
                sum(x["role"] == "BOWL" for x in xs) >= 3)

    def evidence_overs(xs):
        return sum(min(4, float(x.get("historical_bowling_balls", 0)) / 6)
                   for x in xs if x["role"] in ("BOWL", "AR"))

    def fitness(xs):
        return sum(x["score"] for x in xs) + min(6, sum(x["bowl_score"] for x in xs) / 11 * .06)

    # Search limited combinations among top 18. First enforce >=12 historically
    # supported bowling overs; if impossible, retain role constraints and flag it.
    import itertools
    feasible_base = []
    feasible_strict = []
    for combo in itertools.combinations(pool, 11):
        if not valid_base(combo):
            continue
        feasible_base.append(combo)
        if evidence_overs(combo) >= 12:
            feasible_strict.append(combo)
    candidates = feasible_strict if feasible_strict else feasible_base
    for combo in candidates:
        sc = fitness(combo)
        if sc > bestscore:
            bestscore = sc
            best = combo
    if best is None:
        best = tuple(p.head(11).to_dict("records"))
        selection_status = "PROVISIONAL_CONSTRAINT_RELAXED"
    elif feasible_strict:
        selection_status = "VALID_ALL_CONSTRAINTS"
    else:
        selection_status = "LOW_BOWLING_EVIDENCE"

    xi = pd.DataFrame(best).sort_values("score", ascending=False)
    xi.insert(0, "team", team_df.TeamCanonical.iloc[0])
    xi["xi_score"] = bestscore / 11 if bestscore > -1e8 else np.nan
    xi["evidence_backed_bowling_overs"] = evidence_overs(best)
    xi["selection_status"] = selection_status
    return xi, p


def build_ball_distributions(d, resolved, batmap, bowlmap):
    # Run-value distribution is player-specific when support exists, shrunk to IPL baseline.
    bat_runs = defaultdict(Counter)
    bat_balls = Counter()
    bowl_runs = defaultdict(list)
    bowl_wk = Counter()
    bowl_balls = Counter()
    batter_ids = defaultdict(set)
    bowler_ids = defaultdict(set)
    for r in d.itertuples():
        if r.is_ipl:
            k = norm(r.batter)
            if int(r.batter_ball):
                bat_runs[k][int(r.batter_runs)] += 1
                bat_balls[k] += 1
            if r.batter_id:
                batter_ids[str(r.batter_id)].add(k)
            k2 = norm(r.bowler)
            bowl_runs[k2].append(int(r.runs_conceded))
            bowl_balls[k2] += int(r.legal)
            bowl_wk[k2] += int(r.bowler_wickets)
            if r.bowler_id:
                bowler_ids[str(r.bowler_id)].add(k2)

    base = Counter()
    for c in bat_runs.values():
        base.update(c)
    base_dist = np.array([base.get(i, 0) + 1 for i in range(7)], float)
    base_dist /= base_dist.sum()

    id_bat = {pid: list(keys) for pid, keys in batter_ids.items()}
    id_bowl = {pid: list(keys) for pid, keys in bowler_ids.items()}

    bdist = {}
    wskill = {}
    for _, r in resolved.iterrows():
        pid = str(r.player_id)
        bkeys = id_bat.get(pid, []) if pid and pid != "nan" else []
        wkeys = id_bowl.get(pid, []) if pid and pid != "nan" else []
        if bkeys:
            c = Counter()
            for k in bkeys:
                c.update(bat_runs[k])
            n = sum(c.values())
            a = np.array([c.get(i, 0) for i in range(7)], float)
            a += base_dist * min(n, 160) * .65 + 1.0
            a /= a.sum()
            bdist[r.PlayerNorm] = a
        else:
            bdist[r.PlayerNorm] = base_dist.copy()
        if wkeys:
            rr = [x for k in wkeys for x in bowl_runs[k]]
            balls = sum(bowl_balls[k] for k in wkeys)
            wk = sum(bowl_wk[k] for k in wkeys)
            total_conceded = float(sum(rr)) if rr else 0.0
            wr = (wk + 0.018 * 180) / (balls + 180)
            econ = (total_conceded + 1.42 * 180) / (balls + 180)
            wskill[r.PlayerNorm] = {"econ_per_ball": econ, "wicket_per_ball": wr, "balls": balls}
        else:
            wskill[r.PlayerNorm] = {"econ_per_ball": 1.42, "wicket_per_ball": .018, "balls": 0}

    extras_total = (pd.to_numeric(d.total_runs, errors="coerce").fillna(0)
                    - pd.to_numeric(d.batter_runs, errors="coerce").fillna(0)).clip(lower=0)
    extras_bowler = (pd.to_numeric(d.runs_conceded, errors="coerce").fillna(0)
                     - pd.to_numeric(d.batter_runs, errors="coerce").fillna(0)).clip(lower=0)
    extras_params = {
        "team_extras_per_ball": float(extras_total.mean()) if len(extras_total) else .12,
        "bowler_extras_per_ball": float(extras_bowler.mean()) if len(extras_bowler) else .08,
    }
    return bdist, wskill, base_dist, extras_params


_TLGE_OVER_CACHE = None


def build_matchup_adjustments(hist, resolved):
    """Shrink direct IPL batter-v-bowler evidence toward neutral when sample is small."""
    x = hist[hist.is_ipl].copy()
    x = x[(x.batter_id.fillna("").astype(str) != "") & (x.bowler_id.fillna("").astype(str) != "")]
    if x.empty:
        return {}
    agg = x.groupby(["batter_id", "bowler_id"]).agg(
        balls=("batter_ball", "sum"), runs=("batter_runs", "sum"), outs=("bowler_wickets", "sum"),
    ).reset_index()
    id_to_name = {str(r.player_id): r.PlayerNorm for r in resolved.itertuples()
                  if str(r.player_id).strip() and str(r.player_id).lower() != "nan"}
    out = {}
    for r in agg.itertuples(index=False):
        bk = id_to_name.get(str(r.batter_id))
        wk = id_to_name.get(str(r.bowler_id))
        if not bk or not wk or int(r.balls) < 20:
            continue
        out[(bk, wk)] = {"balls": int(r.balls),
                         "sr": 100 * float(r.runs) / max(1, int(r.balls)),
                         "dismissal_rate": float(r.outs) / max(1, int(r.balls))}
    return out


def historical_over_state_table(hist):
    """Map logged IPL completed overs to the same 108-state cricket context."""
    global _TLGE_OVER_CACHE
    if _TLGE_OVER_CACHE is not None:
        return _TLGE_OVER_CACHE
    x = hist[hist.is_ipl].copy()
    if x.empty:
        return pd.DataFrame()
    keys = ["match_key", "innings", "over"]
    inn = x.groupby(keys, dropna=False).agg(
        runs_before_source=("total_runs", "sum"),
        wickets_in_over=("wicket", "sum"),
        date=("date", "min"),
    ).reset_index().sort_values(["match_key", "innings", "over"])
    inn["runs_before"] = inn.groupby(["match_key", "innings"]).runs_before_source.cumsum() - inn.runs_before_source
    inn["wickets_before"] = inn.groupby(["match_key", "innings"]).wickets_in_over.cumsum() - inn.wickets_in_over
    inn["phase"] = np.where(inn.over < 6, 0, np.where(inn.over < 15, 1, 2))
    inn["crr_before"] = inn.runs_before / inn.over.clip(lower=1)
    rr_band = np.minimum(5, np.floor(inn.crr_before / 2.5).astype(int))
    wk_band = np.minimum(5, inn.wickets_before.astype(int))
    inn["state"] = (inn.phase * 36 + rr_band * 6 + wk_band).astype(int)
    bowl = x.groupby(keys + ["bowler", "bowler_id"], dropna=False).agg(
        runs_conceded=("runs_conceded", "sum"),
        wickets=("bowler_wickets", "sum"),
        legal_balls=("legal", "sum"),
    ).reset_index()
    out = bowl.merge(inn[keys + ["date", "state"]], on=keys, how="inner")
    out = out[out.legal_balls > 0].sort_values(["date", "match_key", "innings", "over"])
    _TLGE_OVER_CACHE = out
    return out


class CricketTLGEPolicy:
    """Support-gated cricket adapter around the unchanged canonical TL-GE-Entropy agent."""

    def __init__(self, xi, hist, resolved, min_overs=2):
        self.bowlers = xi[xi.role.isin(["BOWL", "AR"])].Player.tolist()
        if not self.bowlers:
            self.bowlers = xi.Player.tolist()
        self.agent = TLGEEntropy(TLGEConfig(), n_states=108, n_actions=len(self.bowlers))
        self.min_overs = int(min_overs)
        self.name_to_action = {norm(p): i for i, p in enumerate(self.bowlers)}
        self.id_to_action = {}
        for _, r in resolved[resolved.Player.isin(self.bowlers)].iterrows():
            if str(r.player_id).strip() and str(r.player_id).lower() != "nan":
                self.id_to_action[str(r.player_id)] = self.name_to_action[norm(r.Player)]

        over_table = historical_over_state_table(hist)
        observations = []
        for r in over_table.itertuples(index=False):
            a = self.id_to_action.get(str(r.bowler_id), self.name_to_action.get(norm(r.bowler), None))
            if a is None:
                continue
            observations.append((int(r.state), int(a), 6.0 * int(r.wickets) - float(r.runs_conceded),
                                 int(int(r.wickets) > 0), int(r.legal_balls)))

        # Initialise active lane as the most frequently logged action per state.
        counts = Counter((s, a) for s, a, _, _, _ in observations)
        for state in range(108):
            by_action = [(counts.get((state, a), 0), a) for a in range(len(self.bowlers))]
            self.agent.active_action[state] = max(by_action)[1]

        # Chronological logged updates. Lane is captured before every update.
        for state, action, reward, outcome, _balls in observations:
            lane = "active" if action == int(self.agent.active_action[state]) else "inactive"
            self.agent.update(state, action, reward, state, outcome, lane)

        self.support = counts
        self.training_overs = len(observations)
        self.supported_states = sum(
            sum(counts.get((s, a), 0) >= self.min_overs for a in range(len(self.bowlers))) >= 2
            for s in range(108)
        )
        self.action_overs = {self.bowlers[a]: sum(v for (s, aa), v in counts.items() if aa == a)
                             for a in range(len(self.bowlers))}

    def choose(self, state, eligible, wskill, rng):
        eligible = [p for p in eligible if norm(p) in self.name_to_action]
        if not eligible:
            return None, "fallback_no_action"
        supported = [self.name_to_action[norm(p)] for p in eligible
                     if self.support.get((int(state), self.name_to_action[norm(p)]), 0) >= self.min_overs]
        if len(supported) >= 2:
            action = self.agent.act(int(state))
            if action not in supported:
                action = max(supported, key=lambda a: (
                    self.agent.inactive_value(int(state), a),
                    self.agent.active_value(int(state), a), -a))
                source = "support_gated_tlge"
            else:
                source = "tlge"
            return self.bowlers[action], source
        # Evidence fallback: best historical economy, with controlled stochasticity.
        weights = np.array([1 / (.35 + wskill.get(norm(p), {}).get("econ_per_ball", 1.42))
                            for p in eligible], float)
        weights /= weights.sum()
        return str(rng.choice(eligible, p=weights)), "empirical_fallback"


_MATCHUP_INDEX_CACHE = None



def matchups_for_xis(xi_a, xi_b, d, resolved):
    """Return descriptive batter-v-bowler evidence using a cached vectorized index."""
    global _MATCHUP_INDEX_CACHE
    if _MATCHUP_INDEX_CACHE is None:
        x = d[d.is_ipl].copy()
        if x.empty:
            pairs = {}
            id_names = {}
        else:
            x["batter_norm"] = x.batter.fillna("").map(norm)
            x["bowler_norm"] = x.bowler.fillna("").map(norm)
            pair_agg = x.groupby(["batter_norm", "bowler_norm"], dropna=False).agg(
                balls=("batter_ball", "sum"),
                runs=("batter_runs", "sum"),
                outs=("wicket", "sum"),
                matches=("match_key", "nunique"),
            ).reset_index()
            pairs = {
                (r.batter_norm, r.bowler_norm): {
                    "balls": int(r.balls), "runs": int(r.runs),
                    "outs": int(r.outs), "matches": int(r.matches),
                }
                for r in pair_agg.itertuples(index=False)
            }

            id_names = defaultdict(set)
            id_rows = pd.concat([
                x.loc[x.batter_id.fillna("").astype(str).ne(""),
                      ["batter_id", "batter_norm"]].rename(
                          columns={"batter_id": "player_id", "batter_norm": "player_norm"}),
                x.loc[x.bowler_id.fillna("").astype(str).ne(""),
                      ["bowler_id", "bowler_norm"]].rename(
                          columns={"bowler_id": "player_id", "bowler_norm": "player_norm"}),
            ], ignore_index=True).drop_duplicates()
            for pid, group in id_rows.groupby("player_id"):
                id_names[str(pid)] = set(group.player_norm.astype(str))
        pid_by_norm = {
            str(r.PlayerNorm): ("" if pd.isna(r.player_id) else str(r.player_id))
            for r in resolved.itertuples(index=False)
        }
        _MATCHUP_INDEX_CACHE = (pairs, dict(id_names), pid_by_norm)

    pairs, id_names, pid_by_norm = _MATCHUP_INDEX_CACHE
    rows = []

    def keys(player):
        player_norm = norm(player)
        pid = pid_by_norm.get(player_norm, "")
        if pid and pid.lower() not in {"nan", "none"}:
            return id_names.get(pid, {player_norm})
        return {player_norm}

    def player_id(player):
        pid = pid_by_norm.get(norm(player), "")
        return "" if pid.lower() in {"nan", "none"} else pid

    for ba in xi_a.itertuples(index=False):
        batter_keys = keys(ba.Player)
        for bo in xi_b.itertuples(index=False):
            bowler_keys = keys(bo.Player)
            candidates = [pairs.get((bk, wk)) for bk in batter_keys for wk in bowler_keys]
            candidates = [v for v in candidates if v is not None]
            n = sum(v["balls"] for v in candidates)
            runs = sum(v["runs"] for v in candidates)
            dismissals = sum(v["outs"] for v in candidates)
            # Match counts are not additive across aliases; use the largest available count.
            nmatches = max((v["matches"] for v in candidates), default=0)
            tier = ("HIGH" if n >= 60 and nmatches >= 5 else
                    ("MODERATE" if n >= 30 and nmatches >= 3 else
                     ("LIMITED" if n > 0 else "NO_DIRECT_RECORD")))
            rows.append({
                "batting_team": xi_a.team.iloc[0], "batter": ba.Player,
                "batter_id": player_id(ba.Player), "batter_norm": norm(ba.Player),
                "bowling_team": xi_b.team.iloc[0], "bowler": bo.Player,
                "bowler_id": player_id(bo.Player), "bowler_norm": norm(bo.Player),
                "balls": n, "matches": nmatches, "runs": runs,
                "dismissals": dismissals,
                "strike_rate": (100 * runs / n if n else np.nan),
                "evidence_tier": tier,
            })
    return rows



def _schedule_rows_from_csv_bytes(raw, source_label):
    """Parse fixture metadata only. Result, toss and delivery columns are ignored."""
    try:
        d = pd.read_csv(io.BytesIO(raw))
    except Exception:
        return None
    cols = {str(c).strip().lower(): c for c in d.columns}

    def pick(*names):
        for name in names:
            if name in cols:
                return cols[name]
        return None

    comp_col = pick("competition", "tournament", "series")
    date_col = pick("date", "match_date", "start_date")
    venue_col = pick("venue", "ground", "stadium")
    city_col = pick("city", "location")
    teams_col = pick("teams", "team_names")
    team1_col = pick("team1", "team_1", "home_team", "home")
    team2_col = pick("team2", "team_2", "away_team", "away")
    id_col = pick("match_id", "fixture_id", "id")
    no_col = pick("match_no", "match_number", "match")
    stage_col = pick("stage", "match_stage", "round")
    time_col = pick("start_time_ist", "start_time", "time")

    if comp_col:
        d = d[d[comp_col].astype(str).str.upper().str.contains("IPL", na=False)].copy()
    if d.empty or date_col is None:
        return None

    out = []
    for idx, row in d.iterrows():
        if team1_col and team2_col:
            t1 = "" if pd.isna(row[team1_col]) else str(row[team1_col]).strip()
            t2 = "" if pd.isna(row[team2_col]) else str(row[team2_col]).strip()
            ts = [t1, t2]
        elif teams_col:
            raw_teams = str(row[teams_col])
            ts = [x.strip() for x in re.split(r"\s*\|\s*|\s+vs\.?\s+|\s+v\s+", raw_teams, flags=re.I) if x.strip()]
            if len(ts) != 2 and "," in raw_teams:
                ts = [x.strip() for x in raw_teams.split(",") if x.strip()]
        else:
            continue
        if len(ts) != 2:
            continue
        date_value = pd.to_datetime(row[date_col], errors="coerce")
        if pd.isna(date_value):
            continue
        stage = str(row[stage_col]).strip().upper() if stage_col and pd.notna(row[stage_col]) else ""
        match_no = pd.to_numeric(row[no_col], errors="coerce") if no_col else np.nan
        out.append({
            "match_no": int(match_no) if pd.notna(match_no) else idx + 1,
            "match_id": str(row[id_col]).strip() if id_col and pd.notna(row[id_col]) else f"IPL2026_FIXTURE_{idx + 1:03d}",
            "date": date_value,
            "stage": stage,
            "team1": team_name(ts[0]) if ts[0] else "TBC",
            "team2": team_name(ts[1]) if ts[1] else "TBC",
            "venue": str(row[venue_col]).strip() if venue_col and pd.notna(row[venue_col]) else "Unknown",
            "city": str(row[city_col]).strip() if city_col and pd.notna(row[city_col]) else "Unknown",
            "start_time_IST": str(row[time_col]).strip() if time_col and pd.notna(row[time_col]) else "",
        })
    if not out:
        return None
    result = pd.DataFrame(out)
    result = result.drop_duplicates(subset=["match_id"], keep="first")
    return result.sort_values(["match_no", "date", "match_id"]).reset_index(drop=True)


def _validate_2026_schedule(schedule, source_label):
    """Fail closed unless this is a complete, ten-team 2026 IPL schedule."""
    if schedule is None or schedule.empty:
        raise ValueError(f"Empty IPL 2026 schedule from {source_label}")
    s = schedule.copy()
    s["date"] = pd.to_datetime(s["date"], errors="coerce")
    if s["date"].isna().any() or not s["date"].dt.year.eq(2026).all():
        raise ValueError("Schedule contains invalid dates or dates outside calendar year 2026.")
    if "stage" not in s.columns:
        s["stage"] = ""
    s["stage"] = s["stage"].fillna("").astype(str).str.upper()
    if not s["stage"].isin(["LEAGUE", "PLAYOFF"]).all():
        s = s.sort_values(["date", "match_no", "match_id"]).reset_index(drop=True)
        if len(s) != 74:
            raise ValueError(f"Expected 74 IPL fixtures (70 league + 4 playoffs), found {len(s)}.")
        s.loc[:69, "stage"] = "LEAGUE"
        s.loc[70:, "stage"] = "PLAYOFF"
    s = s.sort_values(["match_no", "date", "match_id"]).reset_index(drop=True)
    league = s[s.stage.eq("LEAGUE")].copy()
    playoffs = s[s.stage.eq("PLAYOFF")].copy()
    if len(s) != 74 or len(league) != 70 or len(playoffs) != 4:
        raise ValueError(
            f"Invalid schedule structure: total={len(s)}, league={len(league)}, playoffs={len(playoffs)}; "
            "required 74/70/4."
        )
    expected = {
        "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
        "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
        "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bengaluru",
        "Sunrisers Hyderabad",
    }
    league["team1"] = league["team1"].map(team_name)
    league["team2"] = league["team2"].map(team_name)
    if league[["team1", "team2"]].isna().any().any():
        raise ValueError("League fixture has a missing team.")
    if (league.team1 == league.team2).any():
        raise ValueError("League fixture has the same team in both slots.")
    appearances = pd.concat([league.team1, league.team2]).value_counts()
    if set(appearances.index) != expected:
        raise ValueError(
            f"League schedule does not contain exactly the 10 IPL teams. "
            f"Found={sorted(appearances.index.tolist())}"
        )
    bad = appearances[appearances.ne(14)]
    if not bad.empty:
        raise ValueError(
            "Each team must play exactly 14 league matches. Invalid appearance counts: "
            + str(bad.to_dict())
        )
    # Remove any actual playoff participants from schedule data. Only dates and venues are used.
    s.loc[s.stage.eq("PLAYOFF"), ["team1", "team2"]] = "TBC"
    s.loc[s.stage.eq("LEAGUE"), "team1"] = s.loc[s.stage.eq("LEAGUE"), "team1"].map(team_name)
    s.loc[s.stage.eq("LEAGUE"), "team2"] = s.loc[s.stage.eq("LEAGUE"), "team2"].map(team_name)
    return s


def load_schedule_from_v13(root: Path, explicit=None):
    """
    Load only the exact 2026 schedule CSV or the exact 2026 holdout metadata file.
    Historical CSVs with generic names such as 'matches_updated_ipl.csv' are never
    eligible schedule sources.
    """
    candidates = []
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(f"Specified schedule file/archive does not exist: {p}")
        candidates.append(p)
    else:
        exact_names = {"ipl_2026_schedule.csv", "2026_holdout_matches.csv"}
        for base in [Path(root)]:
            if base.exists():
                candidates.extend(
                    p for p in base.rglob("*")
                    if p.is_file() and (
                        (p.suffix.lower() == ".csv" and p.name.lower() in exact_names)
                        or (p.suffix.lower() == ".zip")
                    )
                )

    seen = set()
    for path in candidates:
        try:
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            if path.suffix.lower() == ".csv":
                if path.name.lower() != "ipl_2026_schedule.csv" and explicit is None:
                    continue
                parsed = _schedule_rows_from_csv_bytes(path.read_bytes(), str(path))
                if parsed is not None:
                    return _validate_2026_schedule(parsed, str(path)), str(path)
                continue

            # Only the specifically named 2026 holdout metadata CSV is accepted from ZIPs.
            with zipfile.ZipFile(path) as zf:
                names = [n for n in zf.namelist() if Path(n).name.lower() == "2026_holdout_matches.csv"]
                for name in names:
                    try:
                        parsed = _schedule_rows_from_csv_bytes(zf.read(name), f"{path}:{name}")
                        if parsed is not None:
                            return _validate_2026_schedule(parsed, f"{path}:{name}"), f"{path}:{name}"
                    except Exception:
                        continue
        except zipfile.BadZipFile:
            continue
        except Exception:
            if explicit:
                raise

    return None, None



def load_holdout_actuals(root: Path):
    """
    Locate the 2026 holdout actual files under root, either as plain files or
    nested inside ZIP archives (to any depth). Returns (matches, innings,
    deliveries) DataFrames, or None if any of the three files cannot be found.

    Only called after predictions have been frozen.
    """
    import gzip
    targets = {"2026_HOLDOUT_MATCHES.csv", "2026_HOLDOUT_INNINGS.csv",
               "2026_HOLDOUT_DELIVERIES.csv.gz"}
    found = {}

    for p in root.rglob("*"):
        if p.is_file() and p.name in targets:
            found.setdefault(p.name, p.read_bytes())

    def scan_zip(blob, depth=0):
        if depth > 3:
            return
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile:
            return
        with zf:
            for n in zf.namelist():
                if n.endswith("/"):
                    continue
                base = Path(n).name
                if base in targets:
                    found.setdefault(base, zf.read(n))
                elif n.lower().endswith(".zip"):
                    try:
                        scan_zip(zf.read(n), depth + 1)
                    except Exception:
                        continue

    for zp in root.rglob("*.zip"):
        try:
            scan_zip(zp.read_bytes())
        except Exception:
            continue

    if not all(t in found for t in targets):
        return None

    am = pd.read_csv(io.BytesIO(found["2026_HOLDOUT_MATCHES.csv"]))
    ai = pd.read_csv(io.BytesIO(found["2026_HOLDOUT_INNINGS.csv"]))
    ad = pd.read_csv(io.BytesIO(gzip.decompress(found["2026_HOLDOUT_DELIVERIES.csv.gz"])),
                     low_memory=False)
    return am, ai, ad



def simulate_innings(batting_xi, bowling_xi, bdist, wskill, rng, bat_totals, bowl_totals,
                     tlge_policy=None, matchup_mods=None, extras_params=None):
    role_order = {"BAT": 0, "WK": 1, "AR": 2, "BOWL": 3}
    batting_order = batting_xi.copy()
    batting_order["_role_order"] = batting_order.role.map(role_order).fillna(4)
    batting_order = batting_order.sort_values(["_role_order", "bat_score"], ascending=[True, False])
    batters = batting_order.Player.tolist()
    bowlers = bowling_xi[bowling_xi.role.isin(["BOWL", "AR"])].Player.tolist()
    if not bowlers:
        bowlers = bowling_xi.Player.tolist()
    overs_alloc = {p: 0 for p in bowlers}
    legal_balls = 0
    extras_total = 0
    extras_params = extras_params or {"team_extras_per_ball": .12, "bowler_extras_per_ball": .08}
    for ov in range(20):
        if len(batters) <= 1:
            break
        eligible = [p for p in bowlers if overs_alloc[p] < 4]
        if not eligible:
            break
        phase = 0 if ov < 6 else (2 if ov >= 15 else 1)
        current_runs = sum(v["runs"] for v in bat_totals.values())
        current_wickets = sum(v["outs"] for v in bat_totals.values())
        crr = current_runs / max(1, ov)
        state = int(phase * 36 + min(5, int(crr / 2.5)) * 6 + min(5, current_wickets))
        bowler = None
        if tlge_policy is not None:
            bowler, _source = tlge_policy.choose(state, eligible, wskill, rng)
        if bowler is None or bowler not in eligible:
            weights = np.array([1 / (.35 + wskill.get(norm(p), {}).get("econ_per_ball", 1.42))
                                for p in eligible], float)
            if phase == 2:
                weights = np.array([weights[i] * (1.15 if wskill.get(norm(p), {}).get("wicket_per_ball", .018) > .022 else 1)
                                    for i, p in enumerate(eligible)])
            probs = weights / weights.sum()
            bowler = str(rng.choice(eligible, p=probs))
        overs_alloc[bowler] += 1
        for _ in range(6):
            if len(batters) <= 1:
                break
            striker = batters[0]
            bd = bdist.get(norm(striker), None)
            if bd is None:
                bd = np.array([.34, .36, .08, .01, .13, .00, .08])
            ws = wskill.get(norm(bowler), {"econ_per_ball": 1.42, "wicket_per_ball": .018})
            strength = np.clip(ws["econ_per_ball"] / 1.42, .72, 1.30)
            probs_b = bd.copy()
            probs_b[0] = max(.01, probs_b[0] + (strength - 1) * .12)
            probs_b[4] = max(.005, probs_b[4] - (strength - 1) * .035)
            probs_b[6] = max(.003, probs_b[6] - (strength - 1) * .02)
            pair = (matchup_mods or {}).get((norm(striker), norm(bowler)))
            wicket_factor = 1.0
            if pair:
                batter_sr = max(35.0, 100 * float(np.dot(np.arange(7), bd)))
                shrink = pair["balls"] / (pair["balls"] + 60.0)
                sr_factor = float(np.clip(1 + shrink * (pair["sr"] / batter_sr - 1), .85, 1.15))
                probs_b[0] /= sr_factor
                probs_b[4] *= sr_factor
                probs_b[6] *= sr_factor
                baseline_wicket = max(.006, ws["wicket_per_ball"])
                wicket_factor = float(np.clip(1 + shrink * (pair["dismissal_rate"] / baseline_wicket - 1), .75, 1.30))
            probs_b /= probs_b.sum()
            runs = int(rng.choice(np.arange(7), p=probs_b))
            legal_balls += 1
            team_extra = int(rng.poisson(max(0.0, extras_params.get("team_extras_per_ball", .12))))
            bowler_extra = int(rng.poisson(max(0.0, extras_params.get("bowler_extras_per_ball", .08))))
            extras_total += team_extra
            bowl_totals[bowler]["runs"] += runs + bowler_extra
            wicket_p = np.clip(ws["wicket_per_ball"] * (1.12 if runs == 0 else .82) * wicket_factor, .003, .10)
            out = bool(rng.random() < wicket_p)
            bat_totals[striker]["runs"] += runs
            bat_totals[striker]["balls"] += 1
            if runs == 4:
                bat_totals[striker]["fours"] += 1
            if runs == 6:
                bat_totals[striker]["sixes"] += 1
            bowl_totals[bowler]["balls"] += 1
            if out:
                bat_totals[striker]["outs"] += 1
                bowl_totals[bowler]["wickets"] += 1
                batters.pop(0)
                if len(batters) <= 1:
                    break
            # Strike rotation on odd runs.
            if runs % 2 and len(batters) > 1:
                batters[0], batters[1] = batters[1], batters[0]
    score = sum(v["runs"] for v in bat_totals.values()) + extras_total
    wickets = sum(v["outs"] for v in bat_totals.values())
    return int(score), min(10, int(wickets)), int(legal_balls)


def simulate_match(row, xis, bdist, wskill, rng, season_bat, season_bowl, team_strength,
                   tlge_policies=None, matchup_mods=None, extras_params=None):
    t1, t2 = row.team1, row.team2
    # Neutral toss model: no actual 2026 toss information is used.
    first, second = (t1, t2) if rng.random() < .5 else (t2, t1)
    b1 = defaultdict(Counter)
    w1 = defaultdict(Counter)
    score1, wk1, balls1 = simulate_innings(xis[first], xis[second], bdist, wskill, rng, b1, w1,
                                           (tlge_policies or {}).get(second), matchup_mods, extras_params)
    b2 = defaultdict(Counter)
    w2 = defaultdict(Counter)
    score2, wk2, balls2 = simulate_innings(xis[second], xis[first], bdist, wskill, rng, b2, w2,
                                           (tlge_policies or {}).get(first), matchup_mods, extras_params)
    for p, v in list(b1.items()) + list(b2.items()):
        season_bat[p]["runs"] += v["runs"]
        season_bat[p]["balls"] += v["balls"]
        season_bat[p]["outs"] += v["outs"]
        season_bat[p]["fours"] += v["fours"]
        season_bat[p]["sixes"] += v["sixes"]
    for p, v in list(w1.items()) + list(w2.items()):
        season_bowl[p]["wickets"] += v["wickets"]
        season_bowl[p]["runs"] += v["runs"]
        season_bowl[p]["balls"] += v["balls"]
    winner = first if score1 > score2 else (second if score2 > score1 else (first if rng.random() < .5 else second))
    return {
        "match_id": row.match_id, "stage": getattr(row, "stage", "LEAGUE"), "date": row.date,
        "team1": t1, "team2": t2, "first_batting": first,
        "team1_score": score1 if first == t1 else score2,
        "team2_score": score2 if second == t2 else score1,
        "team1_wickets": wk1 if first == t1 else wk2,
        "team2_wickets": wk2 if second == t2 else wk1,
        "winner": winner,
        "team1_balls": balls1 if first == t1 else balls2,
        "team2_balls": balls2 if second == t2 else balls1,
    }


def make_team_strength(xis):
    return {t: float(xi.score.mean()) for t, xi in xis.items()}


def season_run(schedule, xis, bdist, wskill, seed, playoff_slots=None, simulation_index=0,
               tlge_policies=None, matchup_mods=None, extras_params=None):
    rng = np.random.default_rng(seed)
    bat = defaultdict(Counter)
    bowl = defaultdict(Counter)
    strength = make_team_strength(xis)
    matches = []
    for row in schedule.sort_values(["date", "match_id"]).itertuples(index=False):
        result = simulate_match(row, xis, bdist, wskill, rng, bat, bowl, strength,
                                tlge_policies, matchup_mods, extras_params)
        matches.append(result)

    standings = {t: {"played": 0, "won": 0, "lost": 0, "points": 0, "runs_for": 0,
                     "balls_for": 0, "runs_against": 0, "balls_against": 0} for t in xis}
    for m in matches:
        a, b = m["team1"], m["team2"]
        w = m["winner"]
        l = b if w == a else a
        standings[w]["played"] += 1
        standings[w]["won"] += 1
        standings[w]["points"] += 2
        standings[l]["played"] += 1
        standings[l]["lost"] += 1
        for t, sc, wk in [(a, m["team1_score"], m["team1_wickets"]), (b, m["team2_score"], m["team2_wickets"])]:
            opp = b if t == a else a
            st = standings[t]
            so = standings[opp]
            balls = m["team1_balls"] if t == a else m["team2_balls"]
            if wk >= 10:
                balls = 120
            st["runs_for"] += sc
            st["balls_for"] += balls
            so["runs_against"] += sc
            so["balls_against"] += balls

    tab = []
    for t, v in standings.items():
        nrr = (v["runs_for"] / max(v["balls_for"], 1) * 6) - (v["runs_against"] / max(v["balls_against"], 1) * 6)
        tab.append({"team": t, **v, "nrr": nrr})
    tab = pd.DataFrame(tab).sort_values(["points", "nrr", "won"], ascending=[False, False, False]).reset_index(drop=True)
    tab["position"] = np.arange(1, len(tab) + 1)
    tab["qualified_top4"] = (tab.position <= 4)

    # Playoffs are simulated from the simulated league table, never from actual 2026 playoff teams.
    if playoff_slots is not None and len(playoff_slots) >= 4 and len(tab) >= 4:
        top = tab.team.tolist()
        q1 = simulate_match(SimpleNamespace(match_id="PLAYOFF_Q1", stage="QUALIFIER_1",
                            date=playoff_slots.iloc[0].date, venue=playoff_slots.iloc[0].venue,
                            team1=top[0], team2=top[1]),
                            xis, bdist, wskill, rng, bat, bowl, strength, tlge_policies, matchup_mods, extras_params)
        q1w = q1["winner"]
        q1l = top[1] if q1w == top[0] else top[0]
        el = simulate_match(SimpleNamespace(match_id="PLAYOFF_ELIMINATOR", stage="ELIMINATOR",
                            date=playoff_slots.iloc[1].date, venue=playoff_slots.iloc[1].venue,
                            team1=top[2], team2=top[3]),
                            xis, bdist, wskill, rng, bat, bowl, strength, tlge_policies, matchup_mods, extras_params)
        elw = el["winner"]
        q2 = simulate_match(SimpleNamespace(match_id="PLAYOFF_Q2", stage="QUALIFIER_2",
                            date=playoff_slots.iloc[2].date, venue=playoff_slots.iloc[2].venue,
                            team1=q1l, team2=elw),
                            xis, bdist, wskill, rng, bat, bowl, strength, tlge_policies, matchup_mods, extras_params)
        q2w = q2["winner"]
        final = simulate_match(SimpleNamespace(match_id="PLAYOFF_FINAL", stage="FINAL",
                               date=playoff_slots.iloc[3].date, venue=playoff_slots.iloc[3].venue,
                               team1=q1w, team2=q2w),
                               xis, bdist, wskill, rng, bat, bowl, strength, tlge_policies, matchup_mods, extras_params)
        matches.extend([q1, el, q2, final])

    return (matches, tab,
            pd.DataFrame([{"player": p, **v} for p, v in bat.items()]),
            pd.DataFrame([{"player": p, **v} for p, v in bowl.items()]))


def do_or_die_proxy(schedule, table_samples, match_sims):
    """Estimate pre-season qualification pivotality from simulated counterfactuals."""
    q = table_samples.groupby("team").qualified_top4.mean().to_dict()
    sim_ids = sorted(table_samples.simulation.unique().tolist())
    rows = []
    for r in schedule.itertuples(index=False):
        p1 = q.get(r.team1, .5)
        p2 = q.get(r.team2, .5)
        pivot_count = 0
        checked = 0
        for sim in sim_ids:
            draw = match_sims[(match_sims.simulation == sim) & (match_sims.match_id == r.match_id) &
                              (match_sims.stage == "LEAGUE")]
            tab = table_samples[table_samples.simulation == sim].copy()
            if draw.empty or tab.empty:
                continue
            winner = str(draw.iloc[0].winner)
            if winner not in {r.team1, r.team2}:
                continue
            loser = r.team2 if winner == r.team1 else r.team1
            base_top = set(tab.sort_values(["points", "nrr", "won"], ascending=False).head(4).team)
            alt = tab.copy()
            alt.loc[alt.team == winner, "points"] -= 2
            alt.loc[alt.team == loser, "points"] += 2
            alt.loc[alt.team == winner, "won"] -= 1
            alt.loc[alt.team == loser, "won"] += 1
            alt_top = set(alt.sort_values(["points", "nrr", "won"], ascending=False).head(4).team)
            pivot_count += int(alt_top != base_top)
            checked += 1
        pivotal = (pivot_count / checked) if checked else 0.0
        pressure = (p1 * (1 - p1) + p2 * (1 - p2))
        score = max(pivotal, pressure * .25)
        label = ("HIGH LEVERAGE" if pivotal >= .50 else
                 ("POTENTIALLY IMPORTANT" if pivotal >= .20 else "LOW PIVOTALITY IN SIMULATIONS"))
        rows.append({
            "match_id": r.match_id, "date": r.date, "team1": r.team1, "team2": r.team2,
            "team1_playoff_probability": p1, "team2_playoff_probability": p2,
            "qualification_pivotal_probability": pivotal, "leverage_index": score,
            "classification": label, "counterfactual_draws": checked,
            "warning": "Pre-season estimate: flip one simulated match's points and wins, hold NRR fixed, "
                       "then compare final top four. Not an in-season mathematical elimination test.",
        })
    return pd.DataFrame(rows)


def write_csv(df, path):
    df.to_csv(path, index=False)


def build_phase_scoring_reference(hist):
    """
    Descriptive IPL phase reference built only from pre-cutoff historical data.
    This is a coaching reference, not a target-setting policy or a live forecast.
    """
    x = hist[hist.is_ipl].copy()
    if x.empty:
        return pd.DataFrame()

    x["phase"] = np.where(x["over"].astype(int) < 6, "POWERPLAY",
                   np.where(x["over"].astype(int) < 15, "MIDDLE", "DEATH"))
    keys = ["match_key", "innings", "venue", "phase"]
    innings_phase = x.groupby(keys, dropna=False).agg(
        runs=("total_runs", "sum"),
        wickets=("wicket", "sum"),
        legal_balls=("legal", "sum"),
    ).reset_index()
    innings_phase["overs"] = innings_phase.legal_balls / 6.0
    innings_phase = innings_phase[innings_phase.overs > 0].copy()
    innings_phase["runs_per_over"] = innings_phase.runs / innings_phase.overs
    innings_phase["wickets_per_over"] = innings_phase.wickets / innings_phase.overs

    rows = []
    for phase, g in innings_phase.groupby("phase"):
        rows.append({
            "scope": "ALL_VENUES",
            "venue": "ALL",
            "phase": phase,
            "innings_sample": int(len(g)),
            "median_runs_per_over": float(g.runs_per_over.median()),
            "p25_runs_per_over": float(g.runs_per_over.quantile(.25)),
            "p75_runs_per_over": float(g.runs_per_over.quantile(.75)),
            "median_wickets_per_over": float(g.wickets_per_over.median()),
        })
    for (venue, phase), g in innings_phase.groupby(["venue", "phase"], dropna=False):
        if len(g) >= 12 and str(venue).strip():
            rows.append({
                "scope": "VENUE",
                "venue": str(venue),
                "phase": phase,
                "innings_sample": int(len(g)),
                "median_runs_per_over": float(g.runs_per_over.median()),
                "p25_runs_per_over": float(g.runs_per_over.quantile(.25)),
                "p75_runs_per_over": float(g.runs_per_over.quantile(.75)),
                "median_wickets_per_over": float(g.wickets_per_over.median()),
            })
    return pd.DataFrame(rows)



def build_target_chase_reference(hist):
    """
    Venue-level innings and chase reference from historical IPL score totals.
    Results are descriptive only; no fixed par score is prescribed.
    """
    x = hist[hist.is_ipl].copy()
    if x.empty:
        return pd.DataFrame()

    x["innings_num"] = pd.to_numeric(x["innings"], errors="coerce")
    x = x[x.innings_num.isin([1, 2])].copy()
    if x.empty:
        return pd.DataFrame()

    inn = x.groupby(["match_key", "venue", "innings_num"], dropna=False).agg(
        score=("total_runs", "sum"),
        wickets=("wicket", "sum"),
        legal_balls=("legal", "sum"),
    ).reset_index()
    first = inn[inn.innings_num == 1].rename(columns={
        "score": "first_innings_score",
        "wickets": "first_innings_wickets",
        "legal_balls": "first_innings_legal_balls",
    }).drop(columns=["innings_num"])
    second = inn[inn.innings_num == 2].rename(columns={
        "score": "second_innings_score",
        "wickets": "second_innings_wickets",
        "legal_balls": "second_innings_legal_balls",
    }).drop(columns=["innings_num"])
    games = first.merge(second, on=["match_key", "venue"], how="inner")
    if games.empty:
        return pd.DataFrame()

    games["chase_success"] = (games.second_innings_score > games.first_innings_score).astype(int)
    games["successful_chase_balls"] = np.where(
        games.chase_success.eq(1), games.second_innings_legal_balls, np.nan
    )

    rows = []
    for scope, key, group_col in [
        ("ALL_VENUES", "ALL", None),
    ]:
        g = games
        rows.append({
            "scope": scope,
            "venue": key,
            "matches": int(len(g)),
            "median_first_innings_score": float(g.first_innings_score.median()),
            "p25_first_innings_score": float(g.first_innings_score.quantile(.25)),
            "p75_first_innings_score": float(g.first_innings_score.quantile(.75)),
            "chase_success_rate": float(g.chase_success.mean()),
            "successful_chases": int(g.chase_success.sum()),
            "median_successful_chase_balls": (
                float(g.successful_chase_balls.median())
                if g.successful_chase_balls.notna().any() else np.nan
            ),
        })
    for venue, g in games.groupby("venue", dropna=False):
        if len(g) >= 12 and str(venue).strip():
            rows.append({
                "scope": "VENUE",
                "venue": str(venue),
                "matches": int(len(g)),
                "median_first_innings_score": float(g.first_innings_score.median()),
                "p25_first_innings_score": float(g.first_innings_score.quantile(.25)),
                "p75_first_innings_score": float(g.first_innings_score.quantile(.75)),
                "chase_success_rate": float(g.chase_success.mean()),
                "successful_chases": int(g.chase_success.sum()),
                "median_successful_chase_balls": (
                    float(g.successful_chase_balls.median())
                    if g.successful_chase_balls.notna().any() else np.nan
                ),
            })
    return pd.DataFrame(rows)


def build_decision_support_outputs(
    out, xis, suitability_df, xi_all, tlge_status, fixture_matchups,
    match_predictions, predicted_table, orange, purple, do_die,
    hist, batmap, bowlmap
):
    """
    Coach-facing recommendation layer.

    It does not change the simulator, selection scores, TL-GE policy, match
    probabilities, or player projections. It translates existing outputs into
    evidence-labelled options for human review.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    actions = []
    team_cards = []
    player_rows = []
    matchup_rows = []
    fixture_rows = []

    def add_action(priority, category, team, opponent, trigger, suggestion,
                   evidence, confidence, source):
        actions.append({
            "priority": priority,
            "category": category,
            "team": team,
            "opponent": opponent,
            "trigger": trigger,
            "suggestion": suggestion,
            "supporting_evidence": evidence,
            "evidence_confidence": confidence,
            "source_segment": source,
            "coach_review_required": "YES",
        })

    # 1) Team strengths and role-balanced player shortlists.
    for team in sorted(xis):
        xi = xis[team].copy()
        ranks = suitability_df[suitability_df.team == team].copy()
        ranks = ranks.sort_values("score", ascending=False)
        selected_names = set(xi.Player.astype(str))
        bat_options = ranks[ranks.role.isin(["BAT", "WK", "AR"])].sort_values(
            ["bat_score", "score"], ascending=False
        )
        bowl_options = ranks[ranks.role.isin(["BOWL", "AR"])].sort_values(
            ["bowl_score", "score"], ascending=False
        )
        top_bat = bat_options.iloc[0] if len(bat_options) else None
        top_bowl = bowl_options.iloc[0] if len(bowl_options) else None
        tlge = tlge_status[tlge_status.team == team]
        tlge_row = tlge.iloc[0] if len(tlge) else None
        st_row = predicted_table[predicted_table.team == team]
        st_row = st_row.iloc[0] if len(st_row) else None

        overseas_count = int(xi.overseas.astype(bool).sum())
        batting_count = int(xi.role.isin(["BAT", "WK", "AR"]).sum())
        bowling_count = int(xi.role.isin(["BOWL", "AR"]).sum())
        keeper_count = int((xi.role == "WK").sum())
        top_bat_name = str(top_bat.Player) if top_bat is not None else "Not available"
        top_bowl_name = str(top_bowl.Player) if top_bowl is not None else "Not available"
        supported_states = int(tlge_row.supported_states_with_2plus_bowlers) if tlge_row is not None else 0
        training_overs = int(tlge_row.training_overs) if tlge_row is not None else 0

        strengths = [
            f"{batting_count} batting options in the selected XI",
            f"{bowling_count} bowling options in the selected XI",
            f"{keeper_count} wicketkeeper option(s)",
            f"{overseas_count}/4 overseas places used",
        ]
        if top_bat is not None:
            strengths.append(f"top batting suitability signal: {top_bat_name}")
        if top_bowl is not None:
            strengths.append(f"top bowling suitability signal: {top_bowl_name}")
        if supported_states:
            strengths.append(f"TL-GE-Entropy has multi-bowler historical support in {supported_states}/108 states")

        team_cards.append({
            "team": team,
            "xi_score": float(xi.xi_score.iloc[0]) if "xi_score" in xi else np.nan,
            "selected_players": " | ".join(xi.Player.astype(str).tolist()),
            "selected_xi_size": int(len(xi)),
            "overseas_used": overseas_count,
            "wicketkeepers": keeper_count,
            "batting_options": batting_count,
            "bowling_options": bowling_count,
            "top_batting_suitability_candidate": top_bat_name,
            "top_batting_suitability_score": float(top_bat.bat_score) if top_bat is not None else np.nan,
            "top_bowling_suitability_candidate": top_bowl_name,
            "top_bowling_suitability_score": float(top_bowl.bowl_score) if top_bowl is not None else np.nan,
            "tlge_training_overs": training_overs,
            "tlge_supported_states": supported_states,
            "projected_mean_points": float(st_row.mean_points) if st_row is not None else np.nan,
            "projected_playoff_probability": float(st_row.playoff_probability) if st_row is not None else np.nan,
            "strength_summary": "; ".join(strengths),
            "interpretation": "Shortlist and simulated strength indicators; not an automatic selection instruction.",
        })

        # Keep top three available options in each role group, whether selected or not.
        for role in ["WK", "BAT", "AR", "BOWL"]:
            group = ranks[ranks.role == role].sort_values("score", ascending=False).head(5)
            for rank_no, (_, r) in enumerate(group.iterrows(), 1):
                is_selected = str(r.Player) in selected_names
                band = "SELECTED_XI" if is_selected else ("NEXT_OPTIONS" if rank_no <= 3 else "SQUAD_DEPTH")
                player_rows.append({
                    "team": team,
                    "role": role,
                    "role_rank": rank_no,
                    "player": r.Player,
                    "selected_in_model_xi": int(is_selected),
                    "shortlist_band": band,
                    "overall_suitability_score": float(r.score),
                    "batting_suitability_score": float(r.bat_score),
                    "bowling_suitability_score": float(r.bowl_score),
                    "batting_evidence": r.get("bat_evidence", "UNKNOWN"),
                    "bowling_evidence": r.get("bowl_evidence", "UNKNOWN"),
                    "historical_balls_faced": float(r.get("historical_balls_faced", 0)),
                    "historical_bowling_balls": float(r.get("historical_bowling_balls", 0)),
                    "identity_status": r.get("identity_status", "UNKNOWN"),
                    "coach_prompt": (
                        "Review role fit, current availability and match conditions before selection."
                        if is_selected else
                        "Consider as a role-specific alternative; compare current form, availability and conditions."
                    ),
                })
            alternatives = group[~group.Player.astype(str).isin(selected_names)].head(1)
            if len(alternatives):
                alt = alternatives.iloc[0]
                add_action(
                    "SQUAD_DEPTH", "ROLE_ALTERNATIVE", team, "",
                    f"{role} role cover",
                    f"Keep {alt.Player} as a role-specific alternative for the {role} group; compare the option against the selected XI if conditions or availability change.",
                    f"Suitability score {float(alt.score):.1f}; batting score {float(alt.bat_score):.1f}; bowling score {float(alt.bowl_score):.1f}; identity status {alt.get('identity_status', 'UNKNOWN')}.",
                    "MODEL_SHORTLIST",
                    "PLAYER_SUITABILITY + ROLE_BALANCE",
                )

        add_action(
            "STANDARD", "XI_BALANCE", team, "",
            "Role balance and squad composition",
            f"Retain {team}'s role-balanced XI as the baseline discussion group; compare {top_bat_name} for batting impact and {top_bowl_name} for bowling contribution.",
            f"{len(xi)} selected; {batting_count} batting options; {bowling_count} bowling options; {overseas_count} overseas players.",
            "MODEL_DERIVED",
            "PLAYER_SUITABILITY + XI_SELECTION",
        )
        if tlge_row is not None:
            add_action(
                "STANDARD", "BOWLING_PREPARATION", team, "",
                "Historical TL-GE-Entropy policy coverage",
                (
                    f"Use the TL-GE-supported state map as a pre-match discussion aid across {supported_states} supported states; confirm the actual match-up and bowler workload with the bowling coach."
                    if supported_states else
                    "Use the empirical bowler profile and coach-led matchup planning; the current TL-GE evidence does not cover enough states for a multi-bowler recommendation."
                ),
                f"{training_overs} historical training overs; {supported_states}/108 states with at least two supported bowling actions.",
                "HISTORICAL_SUPPORT_GATED",
                "TL-GE-ENTROPY",
            )

    # 2) Historical batter-v-bowler opportunities. Only promote sufficiently supported signals.
    if fixture_matchups is not None and not fixture_matchups.empty:
        for r in fixture_matchups.itertuples(index=False):
            balls = int(getattr(r, "balls", 0) or 0)
            matches = int(getattr(r, "matches", 0) or 0)
            tier = str(getattr(r, "evidence_tier", "NO_DIRECT_RECORD"))
            if balls < 30 or matches < 3:
                continue
            bname = str(r.batter)
            wname = str(r.bowler)
            pair_sr = float(r.strike_rate) if pd.notna(r.strike_rate) else np.nan
            base = batmap.get(norm(bname), {})
            base_sr = float(base.get("sr", np.nan)) if base else np.nan
            pair_wicket_rate = float(r.dismissals) / max(1, balls)
            bowler_profile = bowlmap.get(norm(wname), {})
            bowler_base_wicket_rate = float(bowler_profile.get("wicket_rate", np.nan)) if bowler_profile else np.nan

            if pd.notna(pair_sr) and pd.notna(base_sr) and pair_sr >= base_sr + 12:
                matchup_rows.append({
                    "match_id": r.match_id,
                    "date": r.date,
                    "venue": r.venue,
                    "batting_team": r.batting_team,
                    "bowling_team": r.bowling_team,
                    "batter": bname,
                    "bowler": wname,
                    "signal": "BATTER_SCORING_MATCHUP",
                    "historical_balls": balls,
                    "historical_matches": matches,
                    "pair_strike_rate": pair_sr,
                    "batter_historical_strike_rate": base_sr,
                    "strike_rate_delta": pair_sr - base_sr,
                    "pair_dismissal_rate": pair_wicket_rate,
                    "evidence_tier": tier,
                    "suggestion": f"Consider {bname} as a possible scoring option against {wname}; this pairing has historically scored faster than the batter's overall IPL baseline. Check current form and the likely bowling phase.",
                    "caveat": "Descriptive head-to-head evidence, not a causal or guaranteed matchup edge.",
                })
            if (pd.notna(bowler_base_wicket_rate) and
                pair_wicket_rate >= bowler_base_wicket_rate + .012):
                matchup_rows.append({
                    "match_id": r.match_id,
                    "date": r.date,
                    "venue": r.venue,
                    "batting_team": r.batting_team,
                    "bowling_team": r.bowling_team,
                    "batter": bname,
                    "bowler": wname,
                    "signal": "BOWLER_DISMISSAL_MATCHUP",
                    "historical_balls": balls,
                    "historical_matches": matches,
                    "pair_strike_rate": pair_sr,
                    "batter_historical_strike_rate": base_sr,
                    "strike_rate_delta": pair_sr - base_sr if pd.notna(pair_sr) and pd.notna(base_sr) else np.nan,
                    "pair_dismissal_rate": pair_wicket_rate,
                    "evidence_tier": tier,
                    "suggestion": f"Consider {wname} as a matchup option against {bname}; the historical dismissal rate in this pairing is above the bowler's overall IPL wicket rate. Check sample size, batter role and phase.",
                    "caveat": "Descriptive head-to-head evidence, not a causal or guaranteed matchup edge.",
                })

    matchup_df = pd.DataFrame(matchup_rows)
    if not matchup_df.empty:
        matchup_df = matchup_df.sort_values(
            ["historical_matches", "historical_balls"], ascending=False
        ).reset_index(drop=True)
        for r in matchup_df.head(250).itertuples(index=False):
            add_action(
                "MATCH_PREP", r.signal, r.batting_team, r.bowling_team,
                f"{r.batter} vs {r.bowler}",
                r.suggestion,
                f"{r.historical_balls} balls across {r.historical_matches} matches; evidence tier {r.evidence_tier}; pair SR {r.pair_strike_rate:.1f}; batter baseline SR {r.batter_historical_strike_rate:.1f}; dismissal rate {r.pair_dismissal_rate:.3f}.",
                str(r.evidence_tier),
                "HISTORICAL_MATCHUP_EVIDENCE",
            )

    # 3) Fixture cards combine the season simulation and leverage proxy.
    leverage_map = {}
    if do_die is not None and not do_die.empty:
        leverage_map = {str(r.match_id): r for r in do_die.itertuples(index=False)}
    matchup_group = {}
    if not matchup_df.empty:
        for match_id, g in matchup_df.groupby("match_id"):
            matchup_group[str(match_id)] = g

    for r in match_predictions[match_predictions.stage == "LEAGUE"].itertuples(index=False):
        lev = leverage_map.get(str(r.match_id))
        leverage = str(getattr(lev, "classification", "NOT_RANKED")) if lev is not None else "NOT_RANKED"
        p1 = float(r.team1_win_probability)
        prob = max(p1, 1 - p1)
        g = matchup_group.get(str(r.match_id), pd.DataFrame())
        angles = []
        if not g.empty:
            angles = g.head(3).apply(
                lambda x: f"{x['batter']} vs {x['bowler']} ({x['signal']}, {int(x['historical_balls'])} balls)",
                axis=1
            ).tolist()
        fixture_rows.append({
            "match_id": r.match_id,
            "date": r.date,
            "team1": r.team1,
            "team2": r.team2,
            "model_favoured_team": r.predicted_winner,
            "model_win_probability": prob,
            "simulation_confidence_band": r.confidence,
            "team1_projected_score": float(r.team1_score_mean),
            "team1_score_p10": float(r.team1_score_p10),
            "team1_score_p90": float(r.team1_score_p90),
            "team2_projected_score": float(r.team2_score_mean),
            "team2_score_p10": float(r.team2_score_p10),
            "team2_score_p90": float(r.team2_score_p90),
            "qualification_leverage_proxy": leverage,
            "historical_matchup_angles": " | ".join(angles) if angles else "No promoted matchup signal; use role and phase preparation.",
            "coach_prompt": "Use the forecast as a scenario range; decide after toss, final XI, pitch and availability checks.",
        })
        add_action(
            "FIXTURE_PREP", "MATCH_SCENARIO", str(r.team1), str(r.team2),
            f"{r.match_id}: {r.team1} vs {r.team2}",
            f"Prepare for a scenario in which {r.predicted_winner} is marginally favoured by the season simulation; plan for both innings and both toss outcomes.",
            f"Model win probability {prob:.1%}; score means {r.team1_score_mean:.1f}-{r.team2_score_mean:.1f}; qualification leverage proxy {leverage}.",
            str(r.confidence),
            "MONTE_CARLO_MATCH_SIMULATION + DO_OR_DIE_PROXY",
        )

    # 4) Player season-projection spotlights.
    for label, frame, metric, topn in [
        ("BATTING_PROJECTION", orange, "projected_runs", 10),
        ("BOWLING_PROJECTION", purple, "projected_wickets", 10),
    ]:
        if frame is None or frame.empty:
            continue
        for r in frame.head(topn).itertuples(index=False):
            value = float(getattr(r, metric))
            lo = float(getattr(r, "runs_p10" if metric == "projected_runs" else "wickets_p10"))
            hi = float(getattr(r, "runs_p90" if metric == "projected_runs" else "wickets_p90"))
            add_action(
                "MONITOR", label, "", "",
                f"{r.player} appears among the leading simulated season contributors",
                f"Keep {r.player} on the season contribution watchlist; use the projection range to frame role and workload discussions.",
                f"Mean projection {value:.1f}; simulated P10-P90 range {lo:.1f}-{hi:.1f}.",
                "SIMULATION_DISTRIBUTION",
                "PLAYER_SEASON_PROJECTIONS",
            )

    # 5) Historical phase scoring reference, not a fixed target recommendation.
    phase_ref = build_phase_scoring_reference(hist)
    global_ref = phase_ref[phase_ref.scope == "ALL_VENUES"] if not phase_ref.empty else pd.DataFrame()
    target_chase_ref = build_target_chase_reference(hist)
    if not phase_ref.empty:
        for r in global_ref.itertuples(index=False):
            add_action(
                "MATCH_PREP", "PHASE_REFERENCE", "", "",
                f"{r.phase} innings planning",
                f"Use the historical {r.phase.lower()} scoring band as a reference when discussing innings tempo; adjust for venue, conditions, wickets and the available batting resources.",
                f"Median {r.median_runs_per_over:.2f} runs/over; P25-P75 {r.p25_runs_per_over:.2f}-{r.p75_runs_per_over:.2f}; {r.innings_sample} historical innings-phase samples.",
                "HISTORICAL_REFERENCE",
                "IPL_PHASE_SCORING_PROFILE",
            )

    if not target_chase_ref.empty:
        for r in target_chase_ref.itertuples(index=False):
            location_label = "across IPL venues" if r.scope == "ALL_VENUES" else f"at {r.venue}"
            add_action(
                "MATCH_PREP", "TARGET_CHASE_REFERENCE", "", "",
                f"Innings planning reference: {location_label}",
                f"Use the historical first-innings score distribution and chase record {location_label} as a planning reference. Reassess after the toss using pitch, dew, boundary size, current XI and available bowling resources.",
                f"{r.matches} matches; median first-innings score {r.median_first_innings_score:.1f}; P25-P75 {r.p25_first_innings_score:.1f}-{r.p75_first_innings_score:.1f}; historical chase success {r.chase_success_rate:.1%} ({r.successful_chases} successful chases).",
                "HISTORICAL_REFERENCE",
                "IPL_INNINGS_SCORECARD_HISTORY",
            )

    # Persist all DSS outputs without replacing any original simulator artifacts.
    team_cards_df = pd.DataFrame(team_cards)
    player_df = pd.DataFrame(player_rows)
    fixture_df = pd.DataFrame(fixture_rows)
    actions_df = pd.DataFrame(actions)

    write_csv(team_cards_df, out / "DSS_TEAM_STRENGTH_CARDS.csv")
    write_csv(player_df, out / "DSS_PLAYER_SHORTLISTS.csv")
    write_csv(matchup_df, out / "DSS_MATCHUP_SUGGESTIONS.csv")
    write_csv(fixture_df, out / "DSS_FIXTURE_BRIEFS.csv")
    write_csv(actions_df, out / "DSS_ACTION_FEED.csv")
    write_csv(phase_ref, out / "DSS_PHASE_SCORING_REFERENCE.csv")
    write_csv(target_chase_ref, out / "DSS_TARGET_CHASE_REFERENCE.csv")

    def markdown_table(df, cols=None, n=20):
        if df is None or df.empty:
            return "_No supported records generated._"
        view = df if cols is None else df[[c for c in cols if c in df.columns]]
        return view.head(n).to_string(index=False)

    brief = [
        "# IPL Cricket Intelligence — Coach Decision Support Brief",
        "",
        "## Purpose",
        "A suggestion engine that organizes the simulator's existing evidence into options for coaching and analyst review. It does not make autonomous selections or guarantee match outcomes.",
        "",
        "## Existing analytical segments retained",
        "- Historical T20/IPL data ingestion and identity resolution",
        "- Player suitability scoring and constrained playing-XI construction",
        "- TL-GE-Entropy bowling decision policy and support gating",
        "- Historical batter-v-bowler matchup evidence",
        "- Monte Carlo match and season simulation",
        "- Predicted standings, Orange Cap and Purple Cap projections",
        "- Do-or-die / qualification leverage proxy",
        "- Frozen-prediction retrospective evaluation against available 2026 holdout data",
        "",
        "## Team strength cards",
        markdown_table(team_cards_df, ["team", "batting_options", "bowling_options", "overseas_used",
                                       "top_batting_suitability_candidate", "top_bowling_suitability_candidate",
                                       "tlge_supported_states", "strength_summary"], 10),
        "",
        "## Match preparation suggestions",
        markdown_table(matchup_df, ["match_id", "batting_team", "bowling_team", "batter", "bowler",
                                    "signal", "historical_balls", "historical_matches", "suggestion"], 25),
        "",
        "## Fixture scenario briefs",
        markdown_table(fixture_df, ["date", "team1", "team2", "model_favoured_team",
                                    "model_win_probability", "simulation_confidence_band",
                                    "qualification_leverage_proxy", "historical_matchup_angles"], 30),
        "",
        "## Player contribution watchlist",
        "### Batting",
        markdown_table(orange, ["rank", "player", "projected_runs", "runs_p10", "runs_p90",
                                "projected_average", "projected_sr"], 10),
        "",
        "### Bowling",
        markdown_table(purple, ["rank", "player", "projected_wickets", "wickets_p10", "wickets_p90",
                                "projected_economy", "projected_overs"], 10),
        "",
        "## Historical phase reference",
        markdown_table(global_ref, ["phase", "innings_sample", "median_runs_per_over",
                                    "p25_runs_per_over", "p75_runs_per_over",
                                    "median_wickets_per_over"], 10),
        "",
        "## Target and chase reference",
        markdown_table(target_chase_ref, ["scope", "venue", "matches",
                                          "median_first_innings_score",
                                          "p25_first_innings_score", "p75_first_innings_score",
                                          "chase_success_rate", "successful_chases"], 30),
        "",
        "## How coaches should use the suggestions",
        "- Treat each item as a prompt for discussion, not an instruction.",
        "- Check the final XI, player availability, workload, pitch, weather and toss before acting.",
        "- Prefer HIGH/MODERATE direct matchup evidence; treat limited samples as exploratory.",
        "- Compare both teams' scenarios rather than relying on one model-favoured outcome.",
        "- The phase table is a historical scoring reference, not a prescribed par score or chase target.",
        "",
        "## Interpretation",
        "The decision-support layer does not alter the underlying simulation or its outputs. It adds readable summaries, evidence-linked suggestions and coach review prompts.",
    ]
    (out / "DSS_COACH_BRIEF.md").write_text("\n".join(brief), encoding="utf-8")

    print("Decision-support algorithm outputs created:")
    for name in [
        "DSS_COACH_BRIEF.md", "DSS_ACTION_FEED.csv",
        "DSS_TEAM_STRENGTH_CARDS.csv", "DSS_PLAYER_SHORTLISTS.csv",
        "DSS_MATCHUP_SUGGESTIONS.csv", "DSS_FIXTURE_BRIEFS.csv",
        "DSS_PHASE_SCORING_REFERENCE.csv", "DSS_TARGET_CHASE_REFERENCE.csv",
    ]:
        print(" -", name)

    return {
        "team_cards": team_cards_df,
        "player_shortlists": player_df,
        "matchup_suggestions": matchup_df,
        "fixture_briefs": fixture_df,
        "action_feed": actions_df,
        "phase_reference": phase_ref,
    }



def evaluate_against_actual_2026(root, league_schedule, match_predictions,
                                 predicted_table, orange, purple, resolved, xi_all,
                                 matchup_evidence, do_die, out, simulations):
    """Open actual 2026 outcomes only after prediction files have been written."""
    actuals = load_holdout_actuals(Path(root))
    if actuals is None:
        return {"status": "actual_holdout_files_not_found"}
    am, ai, ad = actuals

    am = am[am.competition.astype(str).str.upper().eq("IPL")].copy()
    ai_all = ai[ai.competition.astype(str).str.upper().eq("IPL")].copy()
    ad_all = ad[ad.competition.astype(str).str.upper().eq("IPL")].copy()
    league_ids = set(league_schedule.match_id.astype(str))
    am = am[am.match_id.astype(str).isin(league_ids)].copy()
    ai = ai_all[ai_all.match_id.astype(str).isin(league_ids)].copy()
    ad = ad_all[ad_all.match_id.astype(str).isin(league_ids)].copy()

    am["actual_winner"] = am.outcome.map(lambda x: safe_json(x).get("winner", ""))
    am["actual_winner"] = am.actual_winner.map(team_name)

    # Match winner evaluation.
    pred = match_predictions[match_predictions.stage == "LEAGUE"].copy()
    ev = pred.merge(am[["match_id", "actual_winner"]], on="match_id", how="left")
    ev["actual_team1_win"] = (ev.actual_winner == ev.team1).astype(float)
    ev["correct"] = ev.predicted_winner == ev.actual_winner
    ev["brier_score"] = (ev.team1_win_probability - ev.actual_team1_win) ** 2
    ev["log_loss"] = -(
        ev.actual_team1_win * np.log(ev.team1_win_probability.clip(1e-6, 1 - 1e-6)) +
        (1 - ev.actual_team1_win) * np.log((1 - ev.team1_win_probability).clip(1e-6, 1 - 1e-6))
    )
    actual_scores = ai.groupby(["match_id", "team_batting"]).runs.sum().to_dict()
    ev["actual_team1_score"] = [actual_scores.get((str(r.match_id), r.team1), np.nan) for r in ev.itertuples()]
    ev["actual_team2_score"] = [actual_scores.get((str(r.match_id), r.team2), np.nan) for r in ev.itertuples()]
    ev["team1_score_abs_error"] = (ev.team1_score_mean - ev.actual_team1_score).abs()
    ev["team2_score_abs_error"] = (ev.team2_score_mean - ev.actual_team2_score).abs()
    ev["score_interval_team1_covered"] = (ev.actual_team1_score >= ev.team1_score_p10) & (ev.actual_team1_score <= ev.team1_score_p90)
    ev["score_interval_team2_covered"] = (ev.actual_team2_score >= ev.team2_score_p10) & (ev.actual_team2_score <= ev.team2_score_p90)
    ev.to_csv(out / "MATCH_PREDICTIONS_VS_ACTUAL.csv", index=False)
    valid = ev[ev.actual_winner.astype(str).str.len() > 0]

    # Points and NRR from innings, using full quota when all out.
    team_tot = defaultdict(Counter)
    for r in ai.itertuples(index=False):
        bat = team_name(r.team_batting)
        bowl = team_name(r.team_bowling)
        quota = int(getattr(r, "max_overs", 20) or 20) * 6
        legal = int(getattr(r, "legal_balls", 0) or 0)
        balls = quota if int(r.wickets) >= 10 else legal
        team_tot[bat]["runs_for"] += int(r.runs)
        team_tot[bat]["balls_for"] += balls
        team_tot[bowl]["runs_against"] += int(r.runs)
        team_tot[bowl]["balls_against"] += balls

    points = defaultdict(Counter)
    for r in am.itertuples(index=False):
        ts = [team_name(x) for x in str(r.teams).split("|")]
        if len(ts) != 2:
            continue
        winner = team_name(r.actual_winner)
        points[ts[0]]["played"] += 1
        points[ts[1]]["played"] += 1
        if winner in ts:
            loser = ts[1] if winner == ts[0] else ts[0]
            points[winner]["won"] += 1
            points[winner]["points"] += 2
            points[loser]["lost"] += 1
        else:
            # No-result / abandoned match: one point each.
            points[ts[0]]["points"] += 1
            points[ts[1]]["points"] += 1

    tab = []
    for team in sorted(set(list(points.keys()) + list(team_tot.keys()))):
        q = points[team]
        st = team_tot[team]
        nrr = ((st["runs_for"] / (st["balls_for"] / 6) if st["balls_for"] else 0)
               - (st["runs_against"] / (st["balls_against"] / 6) if st["balls_against"] else 0))
        tab.append({"team": team, "actual_played": q["played"], "actual_wins": q["won"],
                    "actual_losses": q["lost"], "actual_points": q["points"], "actual_nrr": nrr})
    at = pd.DataFrame(tab).sort_values(["actual_points", "actual_nrr", "actual_wins"], ascending=False).reset_index(drop=True)
    at["actual_position"] = np.arange(1, len(at) + 1)

    comp = predicted_table.merge(at, on="team", how="left")
    comp["absolute_points_error"] = (comp.mean_points - comp.actual_points).abs()
    comp["absolute_position_error"] = (comp.mean_position - comp.actual_position).abs()
    comp["predicted_top4"] = False
    predicted_top4_idx = comp.sort_values(["playoff_probability", "mean_points", "mean_nrr"],
                                          ascending=False).head(4).index
    comp.loc[predicted_top4_idx, "predicted_top4"] = True
    comp["actual_top4"] = (comp.actual_position <= 4)
    comp.to_csv(out / "POINTS_TABLE_PREDICTED_VS_ACTUAL.csv", index=False)

    # Actual cap leaderboards, with player IDs as the primary identity key.
    ad_all["batter_norm"] = ad_all.batter.fillna("").map(norm)
    ad_all["bowler_norm"] = ad_all.bowler.fillna("").map(norm)
    wide = pd.to_numeric(ad_all.is_wide, errors="coerce").fillna(0).astype(int)
    bb = ad_all[wide.eq(0)].copy()
    bb["runs_batter"] = pd.to_numeric(bb.runs_batter, errors="coerce").fillna(0)
    bb["wicket_batter"] = (bb.player_out.fillna("").astype(str) == bb.batter.fillna("").astype(str)).astype(int)
    actual_bat = bb.groupby(["batter_id", "batter"], dropna=False).agg(
        actual_runs=("runs_batter", "sum"), actual_balls=("runs_batter", "size"),
        actual_dismissals=("wicket_batter", "sum")).reset_index()
    actual_bat = actual_bat.sort_values(["actual_runs", "actual_balls"], ascending=False).reset_index(drop=True)
    actual_bat["actual_rank"] = np.arange(1, len(actual_bat) + 1)

    ad_all["runs_bowler"] = pd.to_numeric(ad_all.runs_bowler, errors="coerce").fillna(0)
    ad_all["wicket_bowler_credited"] = pd.to_numeric(ad_all.wicket_bowler_credited, errors="coerce").fillna(0)
    actual_bowl = ad_all.groupby(["bowler_id", "bowler"], dropna=False).agg(
        actual_wickets=("wicket_bowler_credited", "sum"),
        actual_runs_conceded=("runs_bowler", "sum"),
        actual_legal_balls=("is_legal", "sum")).reset_index()
    actual_bowl = actual_bowl.sort_values(["actual_wickets", "actual_runs_conceded"],
                                          ascending=[False, True]).reset_index(drop=True)
    actual_bowl["actual_rank"] = np.arange(1, len(actual_bowl) + 1)

    id_map = {str(r.player_id): r.Player for r in resolved.itertuples()
              if str(r.player_id).strip() and str(r.player_id).lower() != "nan"}
    actual_bat["canonical_squad_player"] = actual_bat.batter_id.astype(str).map(id_map)
    actual_bowl["canonical_squad_player"] = actual_bowl.bowler_id.astype(str).map(id_map)
    actual_bat.to_csv(out / "ACTUAL_2026_ORANGE_CAP.csv", index=False)
    actual_bowl.to_csv(out / "ACTUAL_2026_PURPLE_CAP.csv", index=False)

    pred_orange = orange.copy()
    pred_purple = purple.copy()
    player_id_map = {r.PlayerNorm: str(r.player_id) for r in resolved.itertuples()}
    pred_orange["player_id"] = pred_orange.player.map(lambda x: player_id_map.get(norm(x), ""))
    pred_purple["player_id"] = pred_purple.player.map(lambda x: player_id_map.get(norm(x), ""))
    pred_orange.to_csv(out / "PREDICTED_ORANGE_CAP_WITH_IDS.csv", index=False)
    pred_purple.to_csv(out / "PREDICTED_PURPLE_CAP_WITH_IDS.csv", index=False)

    def cap_overlap(pred_df, actual_df, score_col, actual_id_col):
        ptop = pred_df.sort_values(score_col, ascending=False).head(10)
        atop = actual_df.sort_values("actual_rank").head(10)
        pids = set(ptop.player_id.astype(str)) - {"", "nan"}
        aids = set(atop[actual_id_col].astype(str)) - {"", "nan"}
        return {"predicted_top10": len(pids), "actual_top10": len(aids),
                "top10_identity_overlap": len(pids & aids),
                "overlap_players": [id_map.get(x, x) for x in sorted(pids & aids)]}

    cap_eval = {
        "orange_cap": cap_overlap(pred_orange, actual_bat, "projected_runs", "batter_id"),
        "purple_cap": cap_overlap(pred_purple, actual_bowl, "projected_wickets", "bowler_id"),
    }
    (out / "CAP_RACE_EVALUATION.json").write_text(json.dumps(cap_eval, indent=2))

    # Team selection: observed contributor audit, not an official XI reconstruction.
    actual_people = defaultdict(set)
    actual_matches = defaultdict(set)
    for r in ad_all.itertuples(index=False):
        if str(r.batter_id) not in {"", "nan", "None"}:
            tm = team_name(r.team_batting)
            actual_people[tm].add(str(r.batter_id))
            actual_matches[(tm, str(r.batter_id))].add(str(r.match_id))
        if str(r.bowler_id) not in {"", "nan", "None"}:
            tm = team_name(r.team_bowling)
            actual_people[tm].add(str(r.bowler_id))
            actual_matches[(tm, str(r.bowler_id))].add(str(r.match_id))
    sel = []
    for r in xi_all.itertuples(index=False):
        pid = str(r.player_id)
        used = len(actual_matches.get((r.team, pid), set())) if pid not in {"", "nan"} else 0
        sel.append({
            "team": r.team, "player": r.Player, "player_id": pid,
            "predicted_xi": 1, "appeared_as_batter_or_bowler": int(used > 0),
            "matches_contributed": used,
            "actual_contributor_for_team": int(pid in actual_people.get(r.team, set())) if pid not in {"", "nan"} else 0,
            "identity_status": r.identity_status,
            "warning": "Contribution evidence only; field-only players and official playing XI lists are not recoverable from deliveries.",
        })
    pd.DataFrame(sel).to_csv(out / "PREDICTED_XI_PARTICIPATION_AUDIT.csv", index=False)

    # Actual head-to-head matchup results against the pre-season matchup evidence.
    ad["match_id"] = ad.match_id.astype(str)
    ad["batter_norm"] = ad.batter.fillna("").map(norm)
    ad["bowler_norm"] = ad.bowler.fillna("").map(norm)
    ad["is_wide_num"] = pd.to_numeric(ad.is_wide, errors="coerce").fillna(0)
    ad["runs_batter_num"] = pd.to_numeric(ad.runs_batter, errors="coerce").fillna(0)
    ad["wicket_for_batter"] = (ad.player_out.fillna("").astype(str) == ad.batter.fillna("").astype(str)).astype(int)
    ap = ad[ad.is_wide_num.eq(0)].groupby(["match_id", "batter_norm", "bowler_norm"]).agg(
        actual_balls=("runs_batter_num", "size"),
        actual_runs=("runs_batter_num", "sum"),
        actual_dismissals=("wicket_for_batter", "sum")).reset_index()
    ap["actual_strike_rate"] = 100 * ap.actual_runs / ap.actual_balls.clip(lower=1)
    me = matchup_evidence.merge(ap, on=["match_id", "batter_norm", "bowler_norm"], how="inner")
    me.to_csv(out / "MATCHUP_PREDICTIONS_VS_ACTUAL.csv", index=False)

    # Do-or-die retrospective proxy: flip each league match's two points while
    # retaining all other actual outcomes; see whether final top-four membership changes.
    actual_result = {str(r.match_id): team_name(r.actual_winner) for r in am.itertuples(index=False)}
    base_top = set(at.head(4).team)
    piv = []
    for fx in league_schedule.itertuples(index=False):
        winner = actual_result.get(str(fx.match_id), "")
        if winner not in {fx.team1, fx.team2}:
            continue
        loser = fx.team2 if winner == fx.team1 else fx.team1
        alt = {tm: dict(points[tm]) for tm in points}
        alt[winner]["points"] = alt[winner].get("points", 0) - 2
        alt[loser]["points"] = alt[loser].get("points", 0) + 2
        alt_rows = []
        for tm in alt:
            nrr = next((x["actual_nrr"] for x in tab if x["team"] == tm), 0)
            alt_rows.append((tm, alt[tm].get("points", 0), nrr, alt[tm].get("won", 0)))
        alt_table = sorted(alt_rows, key=lambda x: (x[1], x[2], x[3]), reverse=True)
        alt_top = {x[0] for x in alt_table[:4]}
        piv.append({
            "match_id": fx.match_id, "actual_winner": winner, "actual_loser": loser,
            "actual_top4_changed_if_result_flipped": int(alt_top != base_top),
            "evaluation_definition": "Final-table points-only counterfactual; not a match-day mathematical elimination test.",
        })
    pivot = pd.DataFrame(piv)
    dd = do_die.merge(pivot, on="match_id", how="left")
    dd.to_csv(out / "DO_OR_DIE_PREDICTIONS_VS_ACTUAL.csv", index=False)

    metrics = {
        "matches_evaluated": int(len(valid)),
        "winner_accuracy": float(valid.correct.mean()) if len(valid) else None,
        "winner_brier_score": float(valid.brier_score.mean()) if len(valid) else None,
        "winner_log_loss": float(valid.log_loss.mean()) if len(valid) else None,
        "team_score_mae": float(pd.concat([valid.team1_score_abs_error, valid.team2_score_abs_error]).mean()) if len(valid) else None,
        "team_score_80pct_interval_coverage": float(pd.concat([valid.score_interval_team1_covered, valid.score_interval_team2_covered]).mean()) if len(valid) else None,
        "points_table_mae": float(comp.absolute_points_error.mean()) if len(comp) else None,
        "mean_absolute_position_error": float(comp.absolute_position_error.mean()) if len(comp) else None,
        "predicted_top4_actual_top4_overlap": int((comp.predicted_top4 & comp.actual_top4).sum()),
        "matchups_with_direct_historical_evidence": int((me.balls > 0).sum()) if len(me) else 0,
        "realized_matchups_compared": int(len(me)),
        "orange_cap": cap_eval["orange_cap"],
        "purple_cap": cap_eval["purple_cap"],
        "actual_league_matches": int(len(am)),
        "actual_league_dates": [str(pd.to_datetime(am.date).min().date()), str(pd.to_datetime(am.date).max().date())] if len(am) else [],
        "warning": "All actual 2026 results are used only in this evaluator after the prediction outputs have been frozen. Do not tune on this same holdout.",
    }
    (out / "RETROSPECTIVE_EVALUATION.json").write_text(json.dumps(metrics, indent=2, default=str))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="IPL Cricket Intelligence: algorithm-only simulation and evidence-backed decision-support outputs; no dashboard.",
        allow_abbrev=False)
    ap.add_argument("--input-dir", default="/kaggle/input")
    ap.add_argument("--output-dir", default="/kaggle/working/IPL_2026_Simulation")
    ap.add_argument("--squad-file", default=None)
    ap.add_argument("--schedule-zip", default=None, help="Optional ZIP containing the IPL 2026 fixture list; schedule metadata only.")
    ap.add_argument("--schedule-file", default=None, help="Validated standalone ipl_2026_schedule.csv; preferred schedule source.")
    ap.add_argument("--simulations", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-json", type=int, default=None)
    ap.add_argument("--no-zip", action="store_true")
    # In Jupyter/IPython, sys.argv belongs to the kernel (e.g. -f <connection-file>).
    # Use documented defaults in that case; callers should use notebook_main().
    if argv is None and ("ipykernel" in sys.modules or "IPython" in sys.modules):
        argv = []
    args = ap.parse_args(argv)

    start = time.time()
    global _TLGE_OVER_CACHE, _MATCHUP_INDEX_CACHE
    _TLGE_OVER_CACHE = None
    _MATCHUP_INDEX_CACHE = None
    root = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not root.exists():
        raise FileNotFoundError(root)

    print("Loading historical data through 2025-12-31...")
    hist, ingest = load_historical(root, args.max_json)
    print("Historical ingestion:", json.dumps(ingest, indent=2))

    squad_path = find_squad_file(root, args.squad_file)
    squads = load_squads(squad_path)
    alias_to_id, id_to_names = load_identity_crosswalk(root)

    # Resolve canonical IPL CSV names to registry identifiers where the crosswalk is unique.
    for name_col, id_col in [("batter", "batter_id"), ("bowler", "bowler_id")]:
        missing = hist[id_col].fillna("").astype(str).eq("")
        hist.loc[missing, id_col] = hist.loc[missing, name_col].map(
            lambda nm: next(iter(alias_to_id.get(norm(nm), set())))
            if len(alias_to_id.get(norm(nm), set())) == 1 else "")

    # Add registry name-to-ID observations from the parsed pre-2026 deliveries.
    for col_name, col_id in [("batter", "batter_id"), ("bowler", "bowler_id")]:
        for nm, pid in hist[[col_name, col_id]].dropna().drop_duplicates().itertuples(index=False, name=None):
            if str(pid).strip() and str(pid).lower() != "nan":
                alias_to_id[norm(nm)].add(str(pid))
                id_to_names[str(pid)].add(str(nm))

    resolved = resolve_identities(squads, alias_to_id, id_to_names)
    print("Identity resolution:", resolved.identity_status.value_counts().to_dict())

    batmap, bowlmap, bat_profiles, bowl_profiles = make_profiles(hist)
    bprof, wprof = attach_profile_aliases(resolved, batmap, bowlmap, alias_to_id, hist)

    xis = {}
    suitability = []
    for team, g in resolved.groupby("TeamCanonical"):
        xi, rank = choose_xi(g, bprof, wprof)
        xis[team] = xi
        rank.insert(0, "team", team)
        suitability.append(rank)

    schedule, schedule_source = load_schedule_from_v13(root, args.schedule_file or args.schedule_zip)
    if schedule is None:
        raise FileNotFoundError(
            "No validated IPL 2026 schedule found. Upload ipl_2026_schedule.csv to /kaggle/input "
            "or provide it with --schedule-file. Historical matches CSVs are deliberately rejected.")

    schedule = schedule.sort_values(["match_no", "date", "match_id"]).reset_index(drop=True)
    league_schedule = schedule[schedule.stage.eq("LEAGUE")].copy().reset_index(drop=True)
    playoff_slots = schedule[schedule.stage.eq("PLAYOFF")][["date", "venue", "city"]].copy().reset_index(drop=True)
    if len(league_schedule) != 70 or len(playoff_slots) != 4:
        raise RuntimeError(
            f"Schedule validation failed after loading: {len(league_schedule)} league fixtures, "
            f"{len(playoff_slots)} playoff slots.")
    if set(pd.concat([league_schedule.team1, league_schedule.team2]).unique()) != set(xis):
        raise RuntimeError("Schedule teams do not exactly match the ten loaded squad teams.")
    league_appearances = pd.concat([league_schedule.team1, league_schedule.team2]).value_counts()
    if not league_appearances.eq(14).all():
        raise RuntimeError("Each team must have exactly 14 league appearances: " + str(league_appearances.to_dict()))
    print(f"Validated 2026 schedule: 70 league fixtures, 4 playoff slots, "
          f"10 teams x 14 matches; source: {schedule_source}")

    bdist, wskill, base_dist, extras_params = build_ball_distributions(hist, resolved, batmap, bowlmap)
    matchup_mods = build_matchup_adjustments(hist, resolved)
    tlge_policies = {team: CricketTLGEPolicy(xi, hist, resolved, min_overs=2) for team, xi in xis.items()}
    tlge_status = pd.DataFrame([{
        "team": team, "training_overs": pol.training_overs,
        "supported_states_with_2plus_bowlers": pol.supported_states,
        "bowling_actions": len(pol.bowlers),
        "action_support": json.dumps(pol.action_overs, sort_keys=True),
    } for team, pol in tlge_policies.items()])

    # Freeze predictions before any actual 2026 outcome is opened.
    all_tables = []
    bat_runs = []
    bowl_runs = []
    match_sim_rows = []
    for sim in range(args.simulations):
        matches, table, bat, bowl = season_run(
            league_schedule, xis, bdist, wskill, args.seed + sim, playoff_slots, sim,
            tlge_policies, matchup_mods, extras_params)
        for m in matches:
            match_sim_rows.append({**m, "simulation": sim})
        table["simulation"] = sim
        all_tables.append(table)
        if not bat.empty:
            bat["simulation"] = sim
            bat_runs.append(bat)
        if not bowl.empty:
            bowl["simulation"] = sim
            bowl_runs.append(bowl)
        if (sim + 1) % max(1, args.simulations // 10) == 0:
            print(f"Season simulations: {sim + 1}/{args.simulations}")

    tables = pd.concat(all_tables, ignore_index=True)
    match_sims = pd.DataFrame(match_sim_rows)

    playoff_draws = match_sims[match_sims.stage != "LEAGUE"].copy()
    if not playoff_draws.empty:
        write_csv(playoff_draws, out / "PLAYOFF_SIMULATION_DRAWS.csv")
        champion = playoff_draws[playoff_draws.stage == "FINAL"].groupby("winner").size().rename("titles").reset_index()
        champion["title_probability"] = champion.titles / max(1, args.simulations)
        champion = champion.sort_values("title_probability", ascending=False)
        write_csv(champion, out / "PREDICTED_TITLE_PROBABILITIES.csv")

    # Match outcome probabilities and score projections.
    mp = match_sims.groupby(["match_id", "stage", "date", "team1", "team2"]).agg(
        team1_score_mean=("team1_score", "mean"),
        team2_score_mean=("team2_score", "mean"),
        team1_score_p10=("team1_score", lambda x: x.quantile(.10)),
        team1_score_p90=("team1_score", lambda x: x.quantile(.90)),
        team2_score_p10=("team2_score", lambda x: x.quantile(.10)),
        team2_score_p90=("team2_score", lambda x: x.quantile(.90)),
    ).reset_index()
    win_stats = match_sims.assign(team1_win=(match_sims.winner == match_sims.team1).astype(int)) \
        .groupby(["match_id", "stage", "team1", "team2"]).team1_win.agg(["sum", "count"])
    # Laplace smoothing avoids zero/one probabilities from finite Monte Carlo draws.
    winp = ((win_stats["sum"] + 1) / (win_stats["count"] + 2)).to_dict()
    mp["team1_win_probability"] = [winp.get((r.match_id, r.stage, r.team1, r.team2), np.nan) for r in mp.itertuples()]
    mp["predicted_winner"] = np.where(mp.team1_win_probability >= .5, mp.team1, mp.team2)
    mp["win_probability"] = np.maximum(mp.team1_win_probability, 1 - mp.team1_win_probability)
    mp["confidence"] = pd.cut(mp.win_probability, [0, .55, .65, .75, 1.001],
                              labels=["LOW", "MODERATE", "GOOD", "STRONG"], include_lowest=True).astype(str)

    # Standings forecast.
    st = tables.groupby("team").agg(
        mean_points=("points", "mean"),
        p10_points=("points", lambda x: x.quantile(.1)),
        p90_points=("points", lambda x: x.quantile(.9)),
        mean_wins=("won", "mean"),
        mean_nrr=("nrr", "mean"),
        playoff_probability=("qualified_top4", "mean"),
        mean_position=("position", "mean"),
    ).reset_index().sort_values(["playoff_probability", "mean_points"], ascending=False)

    # Cap races from simulated individual totals.
    batdf = pd.concat(bat_runs, ignore_index=True) if bat_runs else pd.DataFrame()
    bowldf = pd.concat(bowl_runs, ignore_index=True) if bowl_runs else pd.DataFrame()
    if not batdf.empty:
        orange = batdf.groupby("player").agg(
            projected_runs=("runs", "mean"),
            runs_p10=("runs", lambda x: x.quantile(.1)),
            runs_p90=("runs", lambda x: x.quantile(.9)),
            mean_balls=("balls", "mean"),
            mean_outs=("outs", "mean"),
            mean_fours=("fours", "mean"),
            mean_sixes=("sixes", "mean"),
        ).reset_index().sort_values("projected_runs", ascending=False)
        orange["projected_average"] = orange.projected_runs / orange.mean_outs.clip(lower=1)
        orange["projected_sr"] = 100 * orange.projected_runs / orange.mean_balls.clip(lower=1)
        orange.insert(0, "rank", np.arange(1, len(orange) + 1))
    else:
        orange = pd.DataFrame()
    if not bowldf.empty:
        purple = bowldf.groupby("player").agg(
            projected_wickets=("wickets", "mean"),
            wickets_p10=("wickets", lambda x: x.quantile(.1)),
            wickets_p90=("wickets", lambda x: x.quantile(.9)),
            mean_runs=("runs", "mean"),
            mean_balls=("balls", "mean"),
        ).reset_index().sort_values("projected_wickets", ascending=False)
        purple["projected_overs"] = purple.mean_balls / 6
        purple["projected_economy"] = purple.mean_runs / purple.projected_overs.clip(lower=.1)
        purple.insert(0, "rank", np.arange(1, len(purple) + 1))
    else:
        purple = pd.DataFrame()

    xi_all = pd.concat(xis.values(), ignore_index=True)
    matchup_rows = []
    for a in xis:
        for b in xis:
            if a != b:
                matchup_rows.extend(matchups_for_xis(xis[a], xis[b], hist, resolved))
    matchups = pd.DataFrame(matchup_rows)

    fixture_rows = []
    for fx in league_schedule.itertuples(index=False):
        for direction_a, direction_b in [(fx.team1, fx.team2), (fx.team2, fx.team1)]:
            for rec in matchups_for_xis(xis[direction_a], xis[direction_b], hist, resolved):
                rec.update({"match_id": fx.match_id, "date": fx.date, "venue": fx.venue})
                fixture_rows.append(rec)
    fixture_matchups = pd.DataFrame(fixture_rows)

    suitability_df = pd.concat(suitability, ignore_index=True)
    do_die = do_or_die_proxy(league_schedule, tables, match_sims)

    # Coach-facing decision-support overlay. Existing analytical segments and
    # their underlying predictions remain unchanged.
    dss_outputs = build_decision_support_outputs(
        out, xis, suitability_df, xi_all, tlge_status, fixture_matchups,
        mp, st, orange, purple, do_die, hist, batmap, bowlmap
    )

    # Save prediction outputs.
    write_csv(squads, out / "SQUADS_USED.csv")
    write_csv(schedule, out / "SCHEDULE_USED.csv")
    write_csv(resolved, out / "SQUAD_IDENTITY_AUDIT.csv")
    write_csv(suitability_df, out / "PLAYER_SUITABILITY_ALL_TEAMS.csv")
    write_csv(xi_all, out / "PREDICTED_PLAYING_XI_ALL_TEAMS.csv")
    write_csv(tlge_status, out / "TLGE_TEAM_POLICY_STATUS.csv")
    write_csv(matchups, out / "HISTORICAL_MATCHUPS_ALL_TEAM_COMBINATIONS.csv")
    write_csv(fixture_matchups, out / "FIXTURE_MATCHUP_EVIDENCE.csv")
    write_csv(mp, out / "MATCH_PREDICTIONS.csv")
    write_csv(st, out / "PREDICTED_POINTS_TABLE.csv")
    write_csv(orange.head(20), out / "PREDICTED_ORANGE_CAP_TOP20.csv")
    write_csv(purple.head(20), out / "PREDICTED_PURPLE_CAP_TOP20.csv")
    write_csv(do_die, out / "DO_OR_DIE_LEVERAGE_PRESEASON_PROXY.csv")
    write_csv(match_sims, out / "MATCH_SIMULATION_DRAWS.csv")

    xi_summary = xi_all.groupby("team").agg(
        xi_score=("xi_score", "first"),
        players=("Player", "count"),
        overseas=("overseas", "sum"),
        keepers=("role", lambda x: (x == "WK").sum()),
        batting_options=("role", lambda x: x.isin(["BAT", "WK", "AR"]).sum()),
        bowling_options=("role", lambda x: x.isin(["BOWL", "AR"]).sum()),
        evidence_backed_bowling_overs=("evidence_backed_bowling_overs", "first"),
        selection_status=("selection_status", "first"),
    ).reset_index()
    write_csv(xi_summary, out / "TEAM_XI_SUMMARY.csv")

    # Evaluation is strictly post-prediction: actual 2026 files are opened only now.
    evaluation = evaluate_against_actual_2026(
        root, league_schedule, mp, st, orange, purple, resolved, xi_all,
        fixture_matchups, do_die, out, args.simulations)

    limitations = [
        "Empirical Monte Carlo engine is a prototype, not calibrated to match outcomes.",
        "Player batting/bowling rates are simplified; direct historical batter-bowler adjustments are applied only with >=20 balls and are shrunk toward neutral.",
        "TL-GE-Entropy uses logged historical over rewards in a 108-state adapter; it does not estimate counterfactual bowler outcomes.",
        "Matchup modifiers are descriptive shrinkage adjustments, not causal estimates; unsupported pairs receive no adjustment.",
        "Team XI scoring weights are heuristic and not coach-calibrated.",
        "Do-or-die labels are pre-season leverage proxies, not actual in-season qualification states.",
        "Ties, rain, DLS and super overs are not fully modelled; playoffs use a simplified IPL top-four bracket.",
        "Simulated NRR uses legal balls faced and is an approximation of official NRR rules.",
    ]
    quality = {
        "historical_ingestion": ingest,
        "squad_file": str(squad_path),
        "squad_rows": int(len(resolved)),
        "teams": sorted(resolved.TeamCanonical.unique().tolist()),
        "identity_status_counts": resolved.identity_status.value_counts().to_dict(),
        "fixtures": int(len(league_schedule)),
        "playoff_slots": int(0 if playoff_slots is None else len(playoff_slots)),
        "simulations": int(args.simulations),
        "training_cutoff": "2025-12-31",
        "schedule_source": str(schedule_source),
        "schedule_validation": {
            "total_fixtures": int(len(schedule)),
            "league_fixtures": int(len(league_schedule)),
            "playoff_slots": int(len(playoff_slots)),
            "league_appearances_per_team": league_appearances.to_dict(),
            "playoff_participants_exposed": False,
        },
        "schedule_fields_used": ["match_id", "date", "teams", "venue", "city", "stage"],
        "actual_2026_results_used_for_prediction": False,
        "injury_availability_modelled": False,
        "fallbacks": [
            "IPL-only primary profiles",
            "cross-league fallback discounted 0.65",
            "neutral prior for no-history players",
            "TL-GE-Entropy support-gated to >=2 logged overs for >=2 available bowlers; empirical economy fallback otherwise",
        ],
        "limitations": limitations,
    }
    (out / "DATA_QUALITY_AND_METHOD.json").write_text(json.dumps(quality, indent=2, default=str))

    report = [
        "# IPL 2026 Pre-season Simulation",
        "",
        "- Training cutoff: **2025-12-31**",
        f"- Historical matches parsed: **{ingest['parsed_matches']:,}**",
        f"- Historical deliveries: **{ingest['delivery_rows']:,}**",
        f"- IPL delivery rows: **{ingest['ipl_rows']:,}**",
        f"- League fixtures simulated: **{len(league_schedule)}**",
        f"- Playoff slots simulated from predicted top four: **{0 if playoff_slots is None else len(playoff_slots)}**",
        f"- Monte Carlo seasons: **{args.simulations}**",
        f"- Squad identity counts: **{resolved.identity_status.value_counts().to_dict()}**",
        "",
        "## Decision-support algorithm outputs\n- - DSS_COACH_BRIEF.md: coach-facing summary\n- DSS_ACTION_FEED.csv: prioritized suggestion feed\n- DSS_TEAM_STRENGTH_CARDS.csv: team-level strength summaries\n- DSS_PLAYER_SHORTLISTS.csv: role-specific player options\n- DSS_MATCHUP_SUGGESTIONS.csv: supported historical matchup angles\n- DSS_FIXTURE_BRIEFS.csv: match scenario and leverage briefs\n- DSS_PHASE_SCORING_REFERENCE.csv: historical phase tempo reference\n\n## Predicted points table",
        st.to_string(index=False),
        "",
        "## Top Orange Cap candidates",
        orange.head(10).to_string(index=False) if not orange.empty else "No batting totals generated.",
        "",
        "## Top Purple Cap candidates",
        purple.head(10).to_string(index=False) if not purple.empty else "No bowling totals generated.",
        "",
        "## Retrospective evaluation",
        json.dumps(evaluation, indent=2, default=str),
        "",
        "## Important limitations",
        *["- " + x for x in limitations],
        "",
        "Predictions were frozen before the retrospective evaluator opened actual 2026 results.",
    ]
    (out / "IPL_2026_SIMULATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    if not args.no_zip:
        zpath = out.parent / "IPL_2026_SIMULATION_RESULTS.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            for p in out.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(Path(out.name) / p.relative_to(out)))
        print("Results ZIP:", zpath)
    print(f"Completed in {(time.time() - start) / 60:.1f} minutes. Outputs: {out}")


def notebook_main(
    input_dir="/kaggle/input",
    output_dir="/kaggle/working/IPL_2026_Simulation",
    squad_file=None,
    schedule_zip=None,
    schedule_file=None,
    simulations=300,
    seed=42,
    max_json=None,
    no_zip=False,
):
    """
    Notebook-native runner. Import this file and call notebook_main(...).
    No shell magic, IPython exit helper, sys.exit(), or quit() is used.
    Returns the output directory as a Path after a successful run.
    """
    argv = [
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--simulations", str(int(simulations)),
        "--seed", str(int(seed)),
    ]
    if squad_file:
        argv.extend(["--squad-file", str(squad_file)])
    if schedule_zip:
        argv.extend(["--schedule-zip", str(schedule_zip)])
    if schedule_file:
        argv.extend(["--schedule-file", str(schedule_file)])
    if max_json is not None:
        argv.extend(["--max-json", str(int(max_json))])
    if no_zip:
        argv.append("--no-zip")
    main(argv)
    return Path(output_dir)


if __name__ == "__main__":
    main()