# NHL_schedule_money_puck_lines_matchup.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time, re

# --- Config ---
FANTASYDATA_SCHEDULE_URL = "https://fantasydata.com/nhl/schedule"
REQUESTS_TIMEOUT = 10
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StreamlitApp/1.0)"}
REQUEST_DELAY = 0.5

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

# --- Conversion helpers ---
def per_60(df):
    if 'icetime' not in df.columns:
        return df
    numeric_cols = df.select_dtypes(include=['number']).columns
    df = df.copy()
    for col in numeric_cols:
        if col not in ['icetime', 'games_played'] and df['icetime'].sum() > 0:
            df[col] = (df[col] / df['icetime']) * 3600
    return df

def per_game(df):
    if 'games_played' not in df.columns:
        return df
    numeric_cols = df.select_dtypes(include=['number']).columns
    df = df.copy()
    for col in numeric_cols:
        if col != 'games_played' and df['games_played'].sum() > 0:
            df[col] = df[col] / df['games_played']
    return df

# --- Streamlit ---
st.set_page_config(page_title="NHL Schedule & High-Danger Lines", layout="wide")
st.title("NHL Schedule → High-Danger Lines (Per 60 Normalized)")

# Load Data
lines = load_csv(urls['Lines'])
goalies = load_csv(urls['Goalies'])
teams = load_csv(urls['Teams'])

if not lines.empty and lines.shape[1] >= 4:
    lines['team_abbr'] = lines.iloc[:, 3].astype(str).str.upper()
else:
    lines['team_abbr'] = ""

schedule_df = fetch_schedule()

if schedule_df.empty:
    st.warning("No games today.")
else:
    st.header("Today's Schedule")
    st.dataframe(schedule_df)

    # --- Raw Data ---
    with st.expander("Raw Data (Per 60 and Per Game Adjusted)"):
        tabs = st.tabs(["Lines (per 60)", "Goalies (per game)", "Teams (per game)"])

        # Lines
        with tabs[0]:
            st.subheader("Lines – Per 60 (using icetime)")
            line_cols = [
                'name','team','position','situation','games_played','icetime',
                'xGoalsPercentage','xGoalsFor','xReboundsFor','shotsOnGoalFor','goalsFor',
                'reboundGoalsFor','highDangerShotsFor','highDangerxGoalsFor','highDangerGoalsFor',
                'reboundxGoalsFor','xGoalsAgainst','xReboundsAgainst','shotsOnGoalAgainst',
                'goalsAgainst','highDangerShotsAgainst','highDangerxGoalsAgainst',
                'highDangerGoalsAgainst','reboundxGoalsAgainst'
            ]
            lines_per60 = per_60(lines)
            show_cols = [c for c in line_cols if c in lines_per60.columns]
            st.dataframe(lines_per60[show_cols].fillna("N/A"))

        # Goalies
        with tabs[1]:
            st.subheader("Goalies – Per Game (by games_played)")
            goalie_cols = [
                'name','team','situation','games_played','xGoals','goals',
                'xRebounds','rebounds','highDangerShots','highDangerxGoals','highDangerGoals'
            ]
            goalies_pg = per_game(goalies)
            show_cols = [c for c in goalie_cols if c in goalies_pg.columns]
            st.dataframe(goalies_pg[show_cols].fillna("N/A"))

        # Teams
        with tabs[2]:
            st.subheader("Teams – Per Game (by games_played)")
            team_cols = [
                'name','situation','games_played','xGoalsFor','xReboundsFor','shotsOnGoalFor',
                'goalsFor','reboundGoalsFor','penaltiesFor','highDangerShotsFor','highDangerxGoalsFor',
                'highDangerGoalsFor','shotsOnGoalAgainst','goalsAgainst','reboundGoalsAgainst',
                'highDangerShotsAgainst','highDangerxGoalsAgainst','highDangerGoalsAgainst'
            ]
            teams_pg = per_game(teams)
            show_cols = [c for c in team_cols if c in teams_pg.columns]
            st.dataframe(teams_pg[show_cols].fillna("N/A"))

    # --- DFS TABLES (All Per 60) ---
    st.subheader("All Lines Playing Today – DFS Scores (Per 60)")

    teams_today = pd.unique(schedule_df[['home_abbr','away_abbr']].values.ravel('K'))
    lines_today = lines[lines['team_abbr'].isin(teams_today)].copy()

    if 'icetime' in lines_today.columns and 'games_played' in lines_today.columns:
        lines_today['iceTimePerGame'] = lines_today['icetime'] / lines_today['games_played']
        lines_today['IceTimePerGame_Minutes'] = lines_today['iceTimePerGame'] / 60

    scoring_df = per_60(lines_today)

    dfs_stats = [
        'iceTimePerGame','highDangerxGoalsFor','highDangerShotsFor',
        'shotsOnGoalFor','reboundsFor','reboundGoalsFor'
    ]
    existing_cols = [c for c in dfs_stats if c in scoring_df.columns]

    def normalize(series):
        if series.max() == series.min():
            return series
        return (series - series.min()) / (series.max() - series.min())

    numeric_cols = [c for c in existing_cols if c in scoring_df.columns]
    scoring_df[numeric_cols] = scoring_df[numeric_cols].apply(normalize)

    scoring_df['DFS_Score'] = (
        0.3*scoring_df.get('iceTimePerGame',0) +
        0.35*scoring_df.get('highDangerxGoalsFor',0) +
        0.15*scoring_df.get('highDangerShotsFor',0) +
        0.1*scoring_df.get('shotsOnGoalFor',0) +
        0.05*scoring_df.get('reboundsFor',0) +
        0.05*scoring_df.get('reboundGoalsFor',0)
    )

    scoring_df = scoring_df.sort_values('DFS_Score', ascending=False)
    st.dataframe(scoring_df.fillna("N/A"))

    # --- Matchup Table (Per 60) ---
    matchups = [f"{r['away_abbr']} @ {r['home_abbr']} ({r['game_time']})" for _,r in schedule_df.iterrows()]
    selected_matchup = st.selectbox("Select a Matchup", matchups)

    m = re.match(r'(\w+)\s*@\s*(\w+)', selected_matchup)
    if m:
        away_team, home_team = m.groups()
        st.subheader(f"DFS Ratings: {home_team} vs {away_team}")

        home_lines = per_60(lines[lines['team_abbr']==home_team])
        away_lines = per_60(lines[lines['team_abbr']==away_team])

        def build_scoring(df):
            df = df.copy()
            if 'icetime' in df.columns:
                df = per_60(df)
            return df

        home = build_scoring(home_lines)
        away = build_scoring(away_lines)
        combined = pd.concat([home, away])

        dfs_cols = [c for c in dfs_stats if c in combined.columns]
        combined[dfs_cols] = combined[dfs_cols].apply(normalize)

        combined['DFS_Score'] = (
            0.3*combined.get('iceTimePerGame',0) +
            0.35*combined.get('highDangerxGoalsFor',0) +
            0.15*combined.get('highDangerShotsFor',0) +
            0.1*combined.get('shotsOnGoalFor',0) +
            0.05*combined.get('reboundsFor',0) +
            0.05*combined.get('reboundGoalsFor',0)
        )

        st.dataframe(combined[['team','position','name'] + dfs_cols + ['DFS_Score']].fillna("N/A"))