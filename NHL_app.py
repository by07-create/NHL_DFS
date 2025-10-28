# NHL_app.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import math
from st_aggrid import AgGrid, GridOptionsBuilder

# --- Config ---
FANTASYDATA_SCHEDULE_URL = "https://fantasydata.com/nhl/schedule"
REQUESTS_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
REQUEST_DELAY = 0.5
HOME_ADV = 1.05
MIN_LAMBDA = 0.35
MAX_GOALIE_REDUCTION = 0.15  # max +/-15% effect
GOALIE_SHRINK_K = 20.0       # shrinkage strength for goalie effects
GOALIE_PRIOR_TEAM_WEIGHT = 0.85  # team vs league prior weighting (1.0 = team-only)

# --- Team Abbreviations ---
TEAM_ABBR_MAP = {
    "Anaheim Ducks": "ANA","Arizona Coyotes": "ARI","Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF","Calgary Flames": "CGY","Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI","Colorado Avalanche": "COL","Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL","Detroit Red Wings": "DET","Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA","Los Angeles Kings": "LAK","Minnesota Wild": "MIN",
    "Montreal Canadiens": "MTL","Nashville Predators": "NSH","New Jersey Devils": "NJD",
    "New York Islanders": "NYI","New York Rangers": "NYR","Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI","Pittsburgh Penguins": "PIT","San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA","St. Louis Blues": "STL","Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR","Vancouver Canucks": "VAN","Utah Mammoth": "UTA",
    "Vegas Golden Knights": "VGK","Washington Capitals": "WSH","Winnipeg Jets": "WPG"
}

# --- Data URLs ---
urls = {
    'Lines':   'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/lines.csv',
    'Goalies': 'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv',
    'Teams':   'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv'
}

# --- Data Helpers ---
@st.cache_data
def load_csv(url):
    try:
        df = pd.read_csv(url)
        df.columns = [c.strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Failed to load {url}: {e}")
        return pd.DataFrame()

def safe_div(a, b):
    try: return a / b
    except Exception: return None

def per_game_from_team_row(team_row):
    if team_row is None or team_row.empty: return None
    gp = team_row.get('games_played', None)
    if not pd.notna(gp) or gp == 0: return None
    g = {}
    if 'scoreVenueAdjustedxGoalsFor' in team_row and 'scoreVenueAdjustedxGoalsAgainst' in team_row:
        g['xg_for_pg'] = safe_div(team_row['scoreVenueAdjustedxGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['scoreVenueAdjustedxGoalsAgainst'], gp)
    elif 'highDangerxGoalsFor' in team_row and 'highDangerxGoalsAgainst' in team_row:
        g['xg_for_pg'] = safe_div(team_row['highDangerxGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['highDangerxGoalsAgainst'], gp)
    elif 'xGoalsFor' in team_row and 'xGoalsAgainst' in team_row:
        g['xg_for_pg'] = safe_div(team_row['xGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['xGoalsAgainst'], gp)
    else:
        return None
    return g

# --- Goalie Modeling ---
def _ratio(row, goals_col, xg_col, min_xg_required):
    xg = row.get(xg_col, None)
    g = row.get(goals_col, None)
    if pd.notna(xg) and pd.notna(g) and xg > 0 and float(xg) >= float(min_xg_required):
        return 1.0 - float(g) / float(xg)
    return None

def goalie_effect_from_row(row):
    if row is None or len(row) == 0:
        return 0.0, 'NONE'
    v = _ratio(row, 'highDangerGoals', 'highDangerxGoals', 4.0)
    if v is not None: return float(v), 'HD'
    v = _ratio(row, 'goals', 'xGoals', 8.0)
    if v is not None: return float(v), 'ALL'
    return 0.0, 'NONE'

def adjusted_goalie_effect(row, max_reduction=MAX_GOALIE_REDUCTION, goalies_ref=None):
    if row is None or len(row) == 0:
        return 0.0, {'source': 'NONE'}
    raw, source = goalie_effect_from_row(row)
    gp = int(row.get('games_played', 0)) if pd.notna(row.get('games_played', 0)) else 0
    k = GOALIE_SHRINK_K

    league_prior, team_prior = 0.0, None
    league_wsum = team_wsum = team_acc = 0.0
    team_code = str(row.get('team', '')).upper()

    if goalies_ref is not None and not goalies_ref.empty:
        for _, r in goalies_ref.iterrows():
            eff_r, _ = goalie_effect_from_row(r)
            gp_r = int(r.get('games_played', 0)) if pd.notna(r.get('games_played', 0)) else 0
            league_prior += eff_r * gp_r
            league_wsum += gp_r
            if str(r.get('team', '')).upper() == team_code:
                team_acc += eff_r * gp_r
                team_wsum += gp_r
        league_prior = league_prior / league_wsum if league_wsum > 0 else 0.0
        team_prior = team_acc / team_wsum if team_wsum > 0 else None

    prior = (GOALIE_PRIOR_TEAM_WEIGHT * team_prior + (1.0 - GOALIE_PRIOR_TEAM_WEIGHT) * league_prior) if team_prior is not None else league_prior
    weight = gp / (gp + k) if (gp + k) > 0 else 0.0
    shrunk = weight * raw + (1.0 - weight) * prior
    clipped = max(min(shrunk, max_reduction), -max_reduction)

    debug = {
        'source': source, 'raw': raw, 'prior': prior, 'gp': gp,
        'weight': weight, 'shrunk': shrunk, 'clipped': clipped,
        'team_prior': team_prior, 'league_prior': league_prior, 'team': team_code
    }
    return clipped, debug

def aggregate_goalie_effect(goalies_df, team_abbr):
    if goalies_df is None or goalies_df.empty: return 0.0, []
    g = goalies_df[goalies_df['team'].astype(str).str.upper() == team_abbr.upper()]
    if g.empty: return 0.0, []
    effects, dbg = [], []
    for _, r in g.iterrows():
        eff, info = adjusted_goalie_effect(r, goalies_ref=goalies_df)
        effects.append(eff)
        dbg.append({'name': r.get('name', ''), **info})
    return (sum(effects) / len(effects)) if effects else 0.0, dbg

def get_goalie_selection_effect(sel_idx, team_abbr, team_df, all_goalies):
    if sel_idx == -1 or team_df is None or team_df.empty:
        eff, dbg = aggregate_goalie_effect(all_goalies, team_abbr)
        return eff, {'mode': 'AGGREGATE', 'details': dbg}
    row = team_df.loc[sel_idx]
    eff, info = adjusted_goalie_effect(row, goalies_ref=all_goalies)
    info['mode'] = 'SINGLE'
    info['name'] = row.get('name', '')
    return eff, info

# --- Game Math ---
def poisson_pmf(k, lam):
    lam = max(0.001, float(lam))
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def win_probs_regulation(lam_home, lam_away, max_goals=12):
    pmf_h = [poisson_pmf(k, lam_home) for k in range(max_goals + 1)]
    pmf_a = [poisson_pmf(k, lam_away) for k in range(max_goals + 1)]
    p_home = p_away = p_tie = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            p = pmf_h[i] * pmf_a[j]
            if i > j: p_home += p
            elif j > i: p_away += p
            else: p_tie += p
    total = p_home + p_away + p_tie
    return {'home_reg': p_home/total, 'away_reg': p_away/total, 'tie_reg': p_tie/total}

def win_probs_with_ot(lam_home, lam_away):
    r = win_probs_regulation(lam_home, lam_away)
    home = r['home_reg'] + r['tie_reg'] * 0.5
    away = r['away_reg'] + r['tie_reg'] * 0.5
    return {**r, 'home_overall': home, 'away_overall': away}

def expected_goals(team_stats, opp_stats, home=False, goalie_adj=0.0):
    if team_stats is None or opp_stats is None: return None
    lam = (team_stats.get('xg_for_pg', 0) + opp_stats.get('xg_against_pg', 0)) / 2.0
    if home: lam *= HOME_ADV
    lam *= (1.0 - goalie_adj)
    return max(MIN_LAMBDA, min(lam, 12.0))

def fetch_schedule():
    try:
        r = requests.get(FANTASYDATA_SCHEDULE_URL, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        st.error(f"Failed to fetch schedule: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if not table: return pd.DataFrame()
    rows = []
    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 2: continue
        match = re.search(r'(.+?)\s*(?:@|vs\.?)\s*(.+?)\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)', " ".join(tds), re.I)
        if match:
            away, home, time_text = match.groups()
        else:
            away, home = tds[0], tds[1]
            time_text = tds[-1] if re.search(r'\d{1,2}:\d{2}', tds[-1]) else None
        rows.append({
            "away": away.strip(), "home": home.strip(),
            "game_time": time_text.strip() if time_text else None,
            "home_abbr": TEAM_ABBR_MAP.get(home.strip(), home.strip()),
            "away_abbr": TEAM_ABBR_MAP.get(away.strip(), away.strip())
        })
    return pd.DataFrame(rows)

# --- Streamlit App ---
st.set_page_config(page_title="NHL High-Danger Lines + Matchup Predictions", layout="wide")
st.title("NHL Schedule → High-Danger Lines + Matchup Predictions")

lines = load_csv(urls['Lines'])
goalies = load_csv(urls['Goalies'])
teams = load_csv(urls['Teams'])

teams_all = teams[teams.get('situation','all').str.lower()=='all'] if not teams.empty else pd.DataFrame()
goalies_all = goalies[goalies.get('situation','all').str.lower()=='all'] if not goalies.empty else pd.DataFrame()

schedule_df = fetch_schedule()
if not schedule_df.empty:
    st.header("Today's Schedule")
    st.dataframe(schedule_df)

    matchups = [f"{row['away_abbr']} @ {row['home_abbr']} ({row['game_time']})" for _, row in schedule_df.iterrows()]
    selected_matchup = st.selectbox("Select a Matchup", matchups)

    m = re.match(r'(\w+)\s*@\s*(\w+)', selected_matchup)
    if m:
        away_team, home_team = m.groups()

        def get_team_row(abbr):
            if teams_all.empty: return pd.Series()
            for c in ['name','team','abbr']:
                if c in teams_all.columns:
                    mask = teams_all[c].astype(str).str.upper().str.contains(abbr.upper(), na=False)
                    if mask.any(): return teams_all[mask].iloc[0]
            return pd.Series()

        home_stats = per_game_from_team_row(get_team_row(home_team))
        away_stats = per_game_from_team_row(get_team_row(away_team))

        def team_goalies(team_abbr):
            if goalies.empty: return pd.DataFrame()
            return goalies[(goalies['team'].str.upper()==team_abbr.upper()) & (goalies['situation'].str.lower()=='all')]

        home_df = team_goalies(home_team)
        away_df = team_goalies(away_team)

        home_options = [-1]+home_df.index.tolist()
        away_options = [-1]+away_df.index.tolist()

        col1,col2 = st.columns(2)
        sel_home = col1.selectbox(f"Select Home Goalie ({home_team})", options=home_options, index=0,
            format_func=lambda x: "Season aggregate" if x==-1 else f"{home_df.loc[x,'name']} ({int(home_df.loc[x,'games_played'])} gp)")
        sel_away = col2.selectbox(f"Select Away Goalie ({away_team})", options=away_options, index=0,
            format_func=lambda x: "Season aggregate" if x==-1 else f"{away_df.loc[x,'name']} ({int(away_df.loc[x,'games_played'])} gp)")

        home_goalie_effect, home_dbg = get_goalie_selection_effect(sel_home, home_team, home_df, goalies_all)
        away_goalie_effect, away_dbg = get_goalie_selection_effect(sel_away, away_team, away_df, goalies_all)

        lam_home = expected_goals(home_stats, away_stats, True, away_goalie_effect)
        lam_away = expected_goals(away_stats, home_stats, False, home_goalie_effect)

        if lam_home is None or lam_away is None:
            st.warning("Insufficient data to produce predictions for this matchup.")
        else:
            probs = win_probs_with_ot(lam_home, lam_away)
            exp_total = lam_home + lam_away

            st.markdown("### Model Output")
            c1,c2,c3 = st.columns(3)
            c1.metric("Home Expected Goals (λ)", f"{lam_home:.2f}")
            c2.metric("Away Expected Goals (λ)", f"{lam_away:.2f}")
            c3.metric("Total Expected Goals", f"{exp_total:.2f}")

            st.caption(f"Goalie effects — Home: {home_goalie_effect:+.3f} (affects AWAY λ), Away: {away_goalie_effect:+.3f} (affects HOME λ)")

            st.markdown("### Win Probabilities (Regulation + OT/SO)")
            st.write(f"Regulation — Home: **{probs['home_reg']*100:.1f}%**, Away: **{probs['away_reg']*100:.1f}%**, Tie: **{probs['tie_reg']*100:.1f}%**")
            st.write(f"Overall — Home: **{probs['home_overall']*100:.1f}%**, Away: **{probs['away_overall']*100:.1f}%**")

            with st.expander("Goalie Impact Debug"):
                st.write("**Home goalie debug (affects AWAY λ):**")
                st.json(home_dbg)
                st.write("**Away goalie debug (affects HOME λ):**")
                st.json(away_dbg)