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

# Team abbreviations
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
    "Toronto Maple Leafs": "TOR","Vancouver Canucks": "VAN","Utah Mammoth": "UTA","Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH","Winnipeg Jets": "WPG"
}

# Data URLs
urls = {
    'Lines':   'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/lines.csv',
    'Goalies': 'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv',
    'Teams':   'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv'
}

# --- Helpers ---
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
    """
    Build per-game metrics from a team row.
    Priority for xG sources:
      1) scoreVenueAdjustedxGoalsFor / Against
      2) highDangerxGoalsFor / Against
      3) xGoalsFor / xGoalsAgainst
    Returns dict with xg_for_pg and xg_against_pg (and sog if available)
    """
    if team_row is None or team_row.empty: return None
    gp = team_row.get('games_played', None)
    if not pd.notna(gp) or gp == 0: return None
    g = {}

    # prefer scoreVenueAdjusted if present
    if 'scoreVenueAdjustedxGoalsFor' in team_row and 'scoreVenueAdjustedxGoalsAgainst' in team_row and pd.notna(team_row.get('scoreVenueAdjustedxGoalsFor', None)):
        g['xg_for_pg'] = safe_div(team_row['scoreVenueAdjustedxGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['scoreVenueAdjustedxGoalsAgainst'], gp)
    elif 'highDangerxGoalsFor' in team_row and 'highDangerxGoalsAgainst' in team_row and pd.notna(team_row.get('highDangerxGoalsFor', None)):
        g['xg_for_pg'] = safe_div(team_row['highDangerxGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['highDangerxGoalsAgainst'], gp)
    elif 'xGoalsFor' in team_row and 'xGoalsAgainst' in team_row:
        g['xg_for_pg'] = safe_div(team_row['xGoalsFor'], gp)
        g['xg_against_pg'] = safe_div(team_row['xGoalsAgainst'], gp)
    else:
        return None

    if 'shotsOnGoalFor' in team_row: g['sog_for_pg'] = safe_div(team_row['shotsOnGoalFor'], gp)
    if 'shotsOnGoalAgainst' in team_row: g['sog_against_pg'] = safe_div(team_row['shotsOnGoalAgainst'], gp)
    return g

def goalie_effect_from_row(row):
    """
    Raw goalie effect as proportion:
      prefer high-danger fields:
         raw = 1 - (highDangerGoals / highDangerxGoals)
      else fallback:
         raw = 1 - (goals / xGoals)
    Positive -> goalie allowed fewer goals than expected (good); negative -> worse than expected.
    """
    if row is None or len(row) == 0: return 0.0
    # prefer high danger
    hd_xg = row.get('highDangerxGoals', None)
    hd_g = row.get('highDangerGoals', None)
    if pd.notna(hd_xg) and hd_xg > 0 and pd.notna(hd_g):
        return float(1.0 - (hd_g / hd_xg))
    xg = row.get('xGoals', None)
    g = row.get('goals', None)
    if pd.notna(xg) and xg > 0 and pd.notna(g):
        return float(1.0 - (g / xg))
    return 0.0

def adjusted_goalie_effect(row, max_reduction=MAX_GOALIE_REDUCTION, goalies_ref=None):
    """
    Shrunk goalie effect using empirical Bayes shrinkage toward league prior.
    Returns clipped shrunk effect in same units as goalie_effect_from_row.
    """
    if row is None or len(row) == 0:
        return 0.0

    raw = goalie_effect_from_row(row)
    gp = int(row.get('games_played', 0)) if pd.notna(row.get('games_played', None)) else 0
    k = GOALIE_SHRINK_K

    # league prior (weighted by gp)
    prior = 0.0
    if goalies_ref is not None and not goalies_ref.empty:
        vals = []
        total_gp = 0
        for _, r in goalies_ref.iterrows():
            gp_r = int(r.get('games_played', 0)) if pd.notna(r.get('games_played', None)) else 0
            eff = goalie_effect_from_row(r)
            vals.append((eff, gp_r))
            total_gp += gp_r
        if total_gp > 0:
            prior = sum(e * g for e, g in vals) / total_gp

    w = gp / (gp + k) if (gp + k) > 0 else 0.0
    shrunk = w * raw + (1.0 - w) * prior
    shrunk = max(min(shrunk, max_reduction), -max_reduction)
    return float(shrunk)

def aggregate_goalie_effect(goalies_df, team_abbr):
    """
    Average the adjusted goalie effects for a team (for the 'Season aggregate' fallback).
    Expects goalies_df to be filtered to situation == 'all' for proper averaging.
    """
    if goalies_df is None or goalies_df.empty:
        return 0.0
    g = goalies_df[goalies_df['team'].astype(str).str.upper() == str(team_abbr).upper()]
    if g.empty:
        g = goalies_df[goalies_df['team'].astype(str).str.upper().str.contains(str(team_abbr).upper(), na=False)]
    if g.empty:
        return 0.0

    effects = []
    for _, r in g.iterrows():
        effects.append(adjusted_goalie_effect(r, max_reduction=MAX_GOALIE_REDUCTION, goalies_ref=goalies_df))
    return float(sum(effects) / len(effects)) if effects else 0.0

def get_goalie_selection_effect(display_idx, team_abbr, per_team_df, goalies_all_df):
    """
    Given the selected index (or -1 for aggregate), return the shrunk effect.
    - display_idx: selected index from selectbox (DataFrame index or -1)
    - team_abbr: 3-letter team code
    - per_team_df: the filtered per-team goalie df (situation == 'all')
    - goalies_all_df: all goalies filtered to 'all' (for league prior)
    """
    if display_idx == -1 or per_team_df is None or per_team_df.empty:
        return aggregate_goalie_effect(goalies_all_df, team_abbr)
    row = per_team_df.loc[display_idx]
    return adjusted_goalie_effect(row, max_reduction=MAX_GOALIE_REDUCTION, goalies_ref=goalies_all_df)

def poisson_pmf(k, lam):
    lam = max(0.001, float(lam))
    return (lam**k) * math.exp(-lam) / math.factorial(k)

def win_probs_regulation(lam_home, lam_away, max_goals=12):
    pmf_h = [poisson_pmf(k, lam_home) for k in range(max_goals+1)]
    pmf_a = [poisson_pmf(k, lam_away) for k in range(max_goals+1)]
    p_home = p_away = p_tie = 0.0
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            p = pmf_h[i] * pmf_a[j]
            if i > j: p_home += p
            elif j > i: p_away += p
            else: p_tie += p
    total = p_home + p_away + p_tie
    return {'home_reg': p_home/total, 'away_reg': p_away/total, 'tie_reg': p_tie/total} if total>0 else {'home_reg':0.0,'away_reg':0.0,'tie_reg':0.0}

def win_probs_with_ot(lam_home, lam_away, ot_home_win_prob=0.5, max_goals=12):
    regs = win_probs_regulation(lam_home, lam_away, max_goals=max_goals)
    home_overall = regs['home_reg'] + regs['tie_reg']*ot_home_win_prob
    away_overall = regs['away_reg'] + regs['tie_reg']*(1.0 - ot_home_win_prob)
    return {**regs, 'home_overall': home_overall, 'away_overall': away_overall}

def fetch_schedule():
    try:
        r = requests.get(FANTASYDATA_SCHEDULE_URL, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        st.error(f"Failed to fetch schedule: {e}")
        return pd.DataFrame()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if not table: return pd.DataFrame()
    rows_data = []
    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds)<2: continue
        game_str = " ".join(tds)
        match = re.search(r'(.+?)\s*(?:@|vs\.?)\s*(.+?)\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)', game_str, re.I)
        if match:
            away, home, time_text = match.groups()
        else:
            away, home = tds[0], tds[1]
            time_text = tds[-1] if re.search(r'\d{1,2}:\d{2}', tds[-1]) else None
        rows_data.append({
            "away": away.strip(), "home": home.strip(),
            "game_time": time_text.strip() if time_text else None,
            "home_abbr": TEAM_ABBR_MAP.get(home.strip(), home.strip()),
            "away_abbr": TEAM_ABBR_MAP.get(away.strip(), away.strip())
        })
    return pd.DataFrame(rows_data)

def expected_goals(team_stats, opp_stats, home=False, goalie_adj=0.0):
    """
    team_stats and opp_stats are per-game dicts from per_game_from_team_row()
    goalie_adj is the shrunk effect for the opposing goalie (proportion-like)
    Returns lambda per game.
    """
    if team_stats is None or opp_stats is None:
        return None

    base_team = team_stats.get('xg_for_pg', 0.0)
    base_opp_allowed = opp_stats.get('xg_against_pg', 0.0)

    lam = (base_team + base_opp_allowed) / 2.0

    if home:
        lam *= HOME_ADV

    # positive goalie_adj = good goalie => reduce opponent scoring
    lam *= (1.0 - goalie_adj)

    # clamp
    lam = max(MIN_LAMBDA, min(lam, 12.0))
    return lam

# ---------- Streamlit App ----------
st.set_page_config(page_title="NHL High-Danger Lines + Matchup Predictions", layout="wide")
st.title("NHL Schedule → High-Danger Lines + Matchup Predictions")

# Load data
lines = load_csv(urls['Lines'])
goalies = load_csv(urls['Goalies'])
teams = load_csv(urls['Teams'])

# Filter 'all' situation for team/goalie reference data
teams_all = teams[teams.get('situation','all').str.lower()=='all'] if not teams.empty else pd.DataFrame()
goalies_all = goalies[goalies.get('situation','all').str.lower()=='all'] if not goalies.empty else pd.DataFrame()

# If lines loaded, create team_abbr
if not lines.empty:
    lines['team_abbr'] = lines.iloc[:, 3].astype(str).str.upper() if lines.shape[1]>=4 else ""

# Schedule
schedule_df = fetch_schedule()
if not schedule_df.empty:
    st.header("Today's Schedule")
    st.dataframe(schedule_df)

    # Raw Data
    with st.expander("Raw Data (Full stats)"):
        tabs = st.tabs(["Lines","Goalies","Teams"])
        with tabs[0]:
            gb = GridOptionsBuilder.from_dataframe(lines)
            gb.configure_default_column(frozen=False)
            if len(lines.columns) >= 2:
                gb.configure_column(lines.columns[0], pinned='left')
                gb.configure_column(lines.columns[1], pinned='left')
            AgGrid(lines, gridOptions=gb.build(), height=600)
        with tabs[1]:
            gb = GridOptionsBuilder.from_dataframe(goalies_all)
            gb.configure_default_column(frozen=False)
            if len(goalies_all.columns) >= 2:
                gb.configure_column(goalies_all.columns[0], pinned='left')
                gb.configure_column(goalies_all.columns[1], pinned='left')
            AgGrid(goalies_all, gridOptions=gb.build(), height=600)
        with tabs[2]:
            gb = GridOptionsBuilder.from_dataframe(teams_all)
            gb.configure_default_column(frozen=False)
            if len(teams_all.columns) >= 2:
                gb.configure_column(teams_all.columns[0], pinned='left')
                gb.configure_column(teams_all.columns[1], pinned='left')
            AgGrid(teams_all, gridOptions=gb.build(), height=600)

    # Lines playing today
    st.subheader("All Lines Playing Today – DFS Scores (per 60)")
    teams_today = pd.unique(schedule_df[['home_abbr','away_abbr']].values.ravel('K'))
    lines_today = lines[lines['team_abbr'].isin(teams_today)].copy() if not lines.empty else pd.DataFrame()
    st.dataframe(lines_today, height=400)

    # Matchups
    matchups = [f"{row['away_abbr']} @ {row['home_abbr']} ({row['game_time']})" for _, row in schedule_df.iterrows()]
    selected_matchup = st.selectbox("Select a Matchup", matchups)

    m = re.match(r'(\w+)\s*@\s*(\w+)', selected_matchup)
    if m:
        away_team, home_team = m.groups()
        st.subheader(f"Prediction: {home_team} vs {away_team}")

        def get_team_row_by_abbr(abbr):
            if teams_all.empty: return pd.Series()
            for col in ['name','team','abbr']:
                if col in teams_all.columns:
                    mask = teams_all[col].astype(str).str.upper().str.contains(str(abbr).upper(), na=False)
                    if mask.any(): return teams_all[mask].iloc[0]
            return pd.Series()

        home_row = get_team_row_by_abbr(home_team)
        away_row = get_team_row_by_abbr(away_team)
        home_stats = per_game_from_team_row(home_row)
        away_stats = per_game_from_team_row(away_row)

        # --- Goalie Selection (index-bound, no string matching) ---
        def goalies_for_team(team_abbr):
            """Return filtered goalie DataFrame for a team (exact 3-letter match, only situation == 'all')."""
            if goalies.empty:
                return pd.DataFrame()
            mask_team = goalies['team'].astype(str).str.upper() == str(team_abbr).upper()
            mask_sit = goalies['situation'].astype(str).str.lower() == 'all'
            g = goalies[mask_team & mask_sit].copy()
            return g

        home_goalies_df = goalies_for_team(home_team)
        away_goalies_df = goalies_for_team(away_team)

        def goalie_label(df, idx):
            if idx == -1:
                return "Season aggregate"
            r = df.loc[idx]
            gp = r.get('games_played', 0)
            try:
                gp = int(gp) if pd.notna(gp) else 0
            except Exception:
                gp = 0
            return f"{r.get('name','')} ({gp} gp)"

        home_options = [-1] + (home_goalies_df.index.tolist() if not home_goalies_df.empty else [])
        away_options = [-1] + (away_goalies_df.index.tolist() if not away_goalies_df.empty else [])

        colg1, colg2 = st.columns(2)
        with colg1:
            sel_home_idx = st.selectbox(
                f"Select Home Goalie ({home_team})",
                options=home_options,
                index=0,
                format_func=lambda x: goalie_label(home_goalies_df, x) if x != -1 else "Season aggregate"
            )
        with colg2:
            sel_away_idx = st.selectbox(
                f"Select Away Goalie ({away_team})",
                options=away_options,
                index=0,
                format_func=lambda x: goalie_label(away_goalies_df, x) if x != -1 else "Season aggregate"
            )

        # Effects from selections
        home_goalie_effect = get_goalie_selection_effect(sel_home_idx, home_team, home_goalies_df, goalies_all)
        away_goalie_effect = get_goalie_selection_effect(sel_away_idx, away_team, away_goalies_df, goalies_all)

        # Optional: Goalie impact analysis
        show_goalie_debug = st.checkbox("Show Goalie Impact Analysis", value=False)
        if show_goalie_debug:
            st.write("### Goalie impact breakdown")
            if sel_home_idx == -1:
                st.write(f"Home goalie: Season aggregate (team {home_team}) → effect = **{home_goalie_effect:+.4f}**")
            else:
                st.write(f"Home goalie: {goalie_label(home_goalies_df, sel_home_idx)} → effect = **{home_goalie_effect:+.4f}**")
            if sel_away_idx == -1:
                st.write(f"Away goalie: Season aggregate (team {away_team}) → effect = **{away_goalie_effect:+.4f}**")
            else:
                st.write(f"Away goalie: {goalie_label(away_goalies_df, sel_away_idx)} → effect = **{away_goalie_effect:+.4f}**")

        # Compute lambdas (apply opposing goalie effect)
        lam_home = expected_goals(home_stats, away_stats, home=True, goalie_adj=away_goalie_effect)
        lam_away = expected_goals(away_stats, home_stats, home=False, goalie_adj=home_goalie_effect)

        if lam_home is None or lam_away is None:
            st.warning("Insufficient data to produce predictions for this matchup.")
        else:
            probs = win_probs_with_ot(lam_home, lam_away, ot_home_win_prob=0.5)
            exp_total = lam_home + lam_away

            st.markdown("### Model Output (analytic)")
            c1,c2,c3 = st.columns(3)
            c1.metric("Home Expected Goals (λ)", f"{lam_home:.2f}")
            c2.metric("Away Expected Goals (λ)", f"{lam_away:.2f}")
            c3.metric("Total Expected Goals", f"{exp_total:.2f}")

            st.caption(f"Goalie effects — Home: {home_goalie_effect:+.3f} (applies to AWAY λ), "
                       f"Away: {away_goalie_effect:+.3f} (applies to HOME λ)")

            st.markdown("### Win probabilities (Regulation / Overall incl. OT/SO)")
            st.write(f"Regulation — Home: **{probs['home_reg']*100:.1f}%**, "
                     f"Away: **{probs['away_reg']*100:.1f}%**, "
                     f"Tie: **{probs['tie_reg']*100:.1f}%**")
            st.write(f"Overall (after OT/SO) — Home: **{probs['home_overall']*100:.1f}%**, "
                     f"Away: **{probs['away_overall']*100:.1f}%**")