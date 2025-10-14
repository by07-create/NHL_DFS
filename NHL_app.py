# NHL_schedule_money_puck_lines_matchup.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

# --- Config ---
FANTASYDATA_SCHEDULE_URL = "https://fantasydata.com/nhl/schedule"
REQUESTS_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
REQUEST_DELAY = 0.5  # polite delay

# --- MoneyPuck CSV URLs ---
urls = {
    'Lines':    'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/lines.csv',
    'Goalies':  'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/goalies.csv',
    'Teams':    'https://moneypuck.com/moneypuck/playerData/seasonSummary/2025/regular/teams.csv'
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
    "Toronto Maple Leafs": "TOR","Vancouver Canucks": "VAN","Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH","Winnipeg Jets": "WPG"
}

def fetch_schedule():
    """Scrape FantasyData NHL schedule and parse away/home and time"""
    try:
        r = requests.get(FANTASYDATA_SCHEDULE_URL, headers=HEADERS, timeout=REQUESTS_TIMEOUT)
        r.raise_for_status()
        time.sleep(REQUEST_DELAY)
    except Exception as e:
        st.error(f"Failed to fetch FantasyData schedule: {e}")
        return pd.DataFrame()

    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table")
    if not table:
        st.warning("Could not find schedule table on page.")
        return pd.DataFrame()

    rows_data = []
    for tr in table.find_all("tr"):
        tds = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(tds) < 2:
            continue
        game_str = " ".join(tds).replace("\n"," ").strip()
        match = re.search(r'(.+?)\s*(?:@|vs\.?)\s*(.+?)\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)', game_str, re.I)
        if match:
            away, home, time_text = match.groups()
        else:
            away, home = tds[0], tds[1]
            time_text = tds[-1] if re.search(r'\d{1,2}:\d{2}', tds[-1]) else None
        rows_data.append({
            "away": away.strip(),
            "home": home.strip(),
            "game_time": time_text.strip() if time_text else None,
            "home_abbr": TEAM_ABBR_MAP.get(home.strip(), home.strip()),
            "away_abbr": TEAM_ABBR_MAP.get(away.strip(), away.strip())
        })
    return pd.DataFrame(rows_data)

# --- Streamlit App ---
st.set_page_config(page_title="NHL High-Danger Lines", layout="wide")
st.title("NHL Schedule → High-Danger Lines")

# Load MoneyPuck data
lines = load_csv(urls['Lines'])
goalies = load_csv(urls['Goalies'])
teams = load_csv(urls['Teams'])

if lines.empty:
    st.warning("Lines data missing.")
else:
    st.success("Loaded Lines data.")
    # ensure team abbr exists (column 4 in original CSV)
    if lines.shape[1] >= 4:
        lines['team_abbr'] = lines.iloc[:, 3].astype(str).str.upper()
    else:
        lines['team_abbr'] = ""

# Fetch schedule
schedule_df = fetch_schedule()
if schedule_df.empty:
    st.warning("No games today.")
else:
    st.header("Today's Schedule")
    st.dataframe(schedule_df)

    # -------------------------
    # Raw Data (expander with tabs)
    # -------------------------
    with st.expander("Raw Data (Full stats)"):
        tabs = st.tabs(["Lines", "Goalies", "Teams"])

        # ---------- LINES TAB ----------
        with tabs[0]:
            st.subheader("Lines (raw totals)")
            # columns requested for lines (only show those present)
            line_cols = [
                'name','team','position','situation','games_played','icetime',
                'xGoalsPercentage','xGoalsFor','xReboundsFor','shotsOnGoalFor','goalsFor',
                'reboundGoalsFor','highDangerShotsFor','highDangerxGoalsFor','highDangerGoalsFor',
                'reboundxGoalsFor','xGoalsAgainst','xReboundsAgainst','shotsOnGoalAgainst',
                'goalsAgainst','highDangerShotsAgainst','highDangerxGoalsAgainst',
                'highDangerGoalsAgainst','reboundxGoalsAgainst'
            ]
            line_cols = [c for c in line_cols if c in lines.columns]
            if lines.empty:
                st.write("Lines CSV didn't load or is empty.")
            else:
                # separate rows by position label "line" vs "pairing" if that exists
                if 'position' in lines.columns:
                    # position values are 'line'/'pairing' per your note — show both grouped
                    lines_copy = lines.copy()
                    lines_copy['position'] = lines_copy['position'].astype(str)
                    # For readability, show lines first then pairings
                    lines_lines = lines_copy[lines_copy['position'].str.lower().str.contains('line', na=False)]
                    lines_pair = lines_copy[lines_copy['position'].str.lower().str.contains('pair', na=False)]
                    others = lines_copy[~(lines_copy.index.isin(lines_lines.index) | lines_copy.index.isin(lines_pair.index))]

                    if not lines_lines.empty:
                        st.markdown("**Forward Lines / Line rows**")
                        st.dataframe(lines_lines[line_cols].fillna("N/A"))
                    if not lines_pair.empty:
                        st.markdown("**Defensive Pairings / Pairing rows**")
                        st.dataframe(lines_pair[line_cols].fillna("N/A"))
                    if not others.empty:
                        st.markdown("**Other / Unlabeled Positions**")
                        st.dataframe(others[line_cols].fillna("N/A"))
                else:
                    st.dataframe(lines[line_cols].fillna("N/A"))

        # ---------- GOALIES TAB ----------
        with tabs[1]:
            st.subheader("Goalies (raw totals)")
            goalie_cols = [
                'name','team','situation','games_played','xGoals','goals',
                'xRebounds','rebounds','highDangerShots','highDangerxGoals','highDangerGoals'
            ]
            goalie_cols = [c for c in goalie_cols if c in goalies.columns]
            if goalies.empty:
                st.write("Goalies CSV didn't load or is empty.")
            else:
                st.dataframe(goalies[goalie_cols].fillna("N/A"))

        # ---------- TEAMS TAB ----------
        with tabs[2]:
            st.subheader("Teams (raw totals)")
            team_cols = [
                'name','situation','games_played','xGoalsFor','xReboundsFor','shotsOnGoalFor',
                'goalsFor','reboundGoalsFor','penaltiesFor','highDangerShotsFor','highDangerxGoalsFor',
                'highDangerGoalsFor','shotsOnGoalAgainst','goalsAgainst','reboundGoalsAgainst',
                'highDangerShotsAgainst','highDangerxGoalsAgainst','highDangerGoalsAgainst'
            ]
            team_cols = [c for c in team_cols if c in teams.columns]
            if teams.empty:
                st.write("Teams CSV didn't load or is empty.")
            else:
                st.dataframe(teams[team_cols].fillna("N/A"))

    # -------------------------
    # All Teams DFS Table
    # -------------------------
    st.subheader("All Lines Playing Today – DFS Scores (per-game rates used for scoring only)")

    # teams playing today
    teams_today = pd.unique(schedule_df[['home_abbr', 'away_abbr']].values.ravel('K'))
    lines_today = lines[lines['team_abbr'].isin(teams_today)].copy()

    # Ice time per game (icetime assumed to be seconds total)
    if 'icetime' in lines_today.columns and 'games_played' in lines_today.columns:
        lines_today['iceTimePerGame'] = lines_today['icetime'] / lines_today['games_played']
        lines_today['IceTimePerGame_Minutes'] = lines_today['iceTimePerGame'] / 60
    else:
        lines_today['iceTimePerGame'] = lines_today.get('iceTimeRank', 0)
        lines_today['IceTimePerGame_Minutes'] = lines_today['iceTimePerGame']

    # totals that should be converted to per-game for scoring only
    totals_to_rate = [
        'highDangerShotsFor','highDangerGoalsFor','shotsOnGoalFor','shotAttemptsFor',
        'unblockedShotAttemptsFor','reboundsFor','reboundGoalsFor','xGoalsFor',
        'xGoalsFromActualReboundsOfShotsFor'
    ]

    scoring_df = lines_today.copy()
    if 'games_played' in scoring_df.columns:
        for col in totals_to_rate:
            if col in scoring_df.columns:
                # convert totals -> per-game rates for scoring
                scoring_df[col] = scoring_df[col] / scoring_df['games_played']

    # DFS stats (same inputs as you previously used)
    dfs_stats = [
        'iceTimePerGame', 'highDangerxGoalsFor', 'highDangerShotsFor',
        'shotsOnGoalFor', 'reboundsFor', 'reboundGoalsFor'
    ]
    existing_dfs_cols = [c for c in dfs_stats if c in scoring_df.columns]

    display_cols_all = ['team_abbr','position','name','IceTimePerGame_Minutes'] + existing_dfs_cols
    # guard if some of those columns are missing
    display_cols_all = [c for c in display_cols_all if c in scoring_df.columns]

    all_dfs_df = scoring_df[display_cols_all].copy()

    # normalize numeric columns used for scoring (exclude team/name/position/display minutes)
    numeric_cols_all = [c for c in all_dfs_df.columns if c not in ['team_abbr','name','position','IceTimePerGame_Minutes']]
    def normalize(series):
        if series.empty or series.max() == series.min(): 
            return series
        return (series - series.min())/(series.max() - series.min())
    if numeric_cols_all:
        # apply normalization safely
        all_dfs_df[numeric_cols_all] = all_dfs_df[numeric_cols_all].apply(lambda s: normalize(s))

    # Compute DFS_Score using same weights you specified
    all_dfs_df['DFS_Score'] = (
        0.3*all_dfs_df.get('iceTimePerGame', 0) +
        0.35*all_dfs_df.get('highDangerxGoalsFor', 0) +
        0.15*all_dfs_df.get('highDangerShotsFor', 0) +
        0.1*all_dfs_df.get('shotsOnGoalFor', 0) +
        0.05*all_dfs_df.get('reboundsFor', 0) +
        0.05*all_dfs_df.get('reboundGoalsFor', 0)
    )

    all_dfs_df = all_dfs_df.sort_values('DFS_Score', ascending=False)
    st.dataframe(all_dfs_df.fillna("N/A"))

    # -------------------------
    # Matchup DFS Table
    # -------------------------
    matchups = [f"{row['away_abbr']} @ {row['home_abbr']} ({row['game_time']})" for _, row in schedule_df.iterrows()]
    selected_matchup = st.selectbox("Select a Matchup", matchups)

    m = re.match(r'(\w+)\s*@\s*(\w+)', selected_matchup)
    if m:
        away_team, home_team = m.groups()
        st.subheader(f"DFS Ratings: {home_team} vs {away_team}")

        home_lines = lines[lines['team_abbr']==home_team].copy()
        away_lines = lines[lines['team_abbr']==away_team].copy()

        # per-game icetime and display minutes (seconds->minutes)
        for df in [home_lines, away_lines]:
            if 'icetime' in df.columns and 'games_played' in df.columns:
                df['iceTimePerGame'] = df['icetime'] / df['games_played']
                df['IceTimePerGame_Minutes'] = df['iceTimePerGame'] / 60
            else:
                df['iceTimePerGame'] = df.get('iceTimeRank', 0)
                df['IceTimePerGame_Minutes'] = df['iceTimePerGame']

        # Build scoring frames for both teams (rate-adjust totals by games_played)
        def build_scoring_frame(df):
            scoring = df.copy()
            if 'games_played' in scoring.columns:
                for col in totals_to_rate:
                    if col in scoring.columns:
                        scoring[col] = scoring[col] / scoring['games_played']
            return scoring

        home_scoring = build_scoring_frame(home_lines)
        away_scoring = build_scoring_frame(away_lines)

        combined_df = pd.concat([home_scoring[['position','name','IceTimePerGame_Minutes'] + [c for c in dfs_stats if c in home_scoring.columns]],
                                 away_scoring[['position','name','IceTimePerGame_Minutes'] + [c for c in dfs_stats if c in away_scoring.columns]]],
                                ignore_index=True, sort=False)

        # find numeric columns present
        dfs_numeric_cols = [c for c in dfs_stats if c in combined_df.columns]
        if dfs_numeric_cols:
            combined_df[dfs_numeric_cols] = combined_df[dfs_numeric_cols].apply(lambda s: normalize(s))

        combined_df['DFS_Score'] = (
            0.3*combined_df.get('iceTimePerGame',0) +
            0.35*combined_df.get('highDangerxGoalsFor',0) +
            0.15*combined_df.get('highDangerShotsFor',0) +
            0.1*combined_df.get('shotsOnGoalFor',0) +
            0.05*combined_df.get('reboundsFor',0) +
            0.05*combined_df.get('reboundGoalsFor',0)
        )

        # Show grouped by position (values are "line" or "pairing")
        pos_col = 'position'
        if pos_col in combined_df.columns:
            combined_df[pos_col] = combined_df[pos_col].astype(str)
            forwards_mask = combined_df[pos_col].str.lower().str.contains('line', na=False)
            defense_mask = combined_df[pos_col].str.lower().str.contains('pair', na=False)

            if forwards_mask.any():
                st.markdown("**Forward Lines (line)**")
                st.dataframe(combined_df[forwards_mask][['position','name','IceTimePerGame_Minutes'] + dfs_numeric_cols + ['DFS_Score']].fillna("N/A"))
            if defense_mask.any():
                st.markdown("**Defensive Pairings (pairing)**")
                st.dataframe(combined_df[defense_mask][['position','name','IceTimePerGame_Minutes'] + dfs_numeric_cols + ['DFS_Score']].fillna("N/A"))
            others = combined_df[~(forwards_mask | defense_mask)]
            if others.shape[0] > 0:
                st.markdown("**Other / Unlabeled**")
                st.dataframe(others[['position','name','IceTimePerGame_Minutes'] + dfs_numeric_cols + ['DFS_Score']].fillna("N/A"))
        else:
            st.dataframe(combined_df[['name','IceTimePerGame_Minutes'] + dfs_numeric_cols + ['DFS_Score']].fillna("N/A"))
