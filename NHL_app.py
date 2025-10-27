# NHL_schedule_money_puck_lines_matchup.py
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import datetime as dt  # ✅ added for date parsing/filtering

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
    "Toronto Maple Leafs": "TOR","Vancouver Canucks": "VAN","Utah Mammoth": "UTA","Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH","Winnipeg Jets": "WPG"
}

def fetch_schedule():
    """Scrape FantasyData NHL schedule, parse away/home/time and attach the date from header rows"""
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
    current_date = None

    for tr in table.find_all("tr"):
        th = tr.find("th")
        if th:
            header_text = th.get_text(strip=True)
            parsed = None
            for fmt in ("%A, %b %d", "%A, %B %d", "%b %d, %Y", "%B %d, %Y", "%b %d", "%B %d"):
                try:
                    parsed_dt = dt.datetime.strptime(header_text, fmt)
                    if parsed_dt.year == 1900:
                        parsed_dt = parsed_dt.replace(year=dt.datetime.now().year)
                    parsed = parsed_dt
                    break
                except Exception:
                    continue
            if parsed:
                current_date = parsed.date()
            continue

        tds = tr.find_all("td")
        if len(tds) == 0:
            row_text = tr.get_text(strip=True)
            if row_text:
                parsed = None
                for fmt in ("%A, %b %d", "%A, %B %d", "%b %d, %Y", "%B %d, %Y", "%b %d", "%B %d"):
                    try:
                        parsed_dt = dt.datetime.strptime(row_text, fmt)
                        if parsed_dt.year == 1900:
                            parsed_dt = parsed_dt.replace(year=dt.datetime.now().year)
                        parsed = parsed_dt
                        break
                    except Exception:
                        continue
                if parsed:
                    current_date = parsed.date()
            continue

        tds_text = [td.get_text(strip=True) for td in tds]
        if len(tds_text) < 2:
            continue
        game_str = " ".join(tds_text).replace("\n"," ").strip()

        match = re.search(r'(.+?)\s*(?:@|vs\.?)\s*(.+?)\s+(\d{1,2}:\d{2}\s*(?:AM|PM)?)', game_str, re.I)
        if match:
            away, home, time_text = match.groups()
        else:
            away, home = tds_text[0], tds_text[1]
            time_text = tds_text[-1] if re.search(r'\d{1,2}:\d{2}', tds_text[-1]) else None

        rows_data.append({
            "away": away.strip(),
            "home": home.strip(),
            "game_time": time_text.strip() if time_text else None,
            "home_abbr": TEAM_ABBR_MAP.get(home.strip(), home.strip()),
            "away_abbr": TEAM_ABBR_MAP.get(away.strip(), away.strip()),
            "date": current_date
        })

    df = pd.DataFrame(rows_data)
    if "date" in df.columns and df["date"].isna().any():
        if df["date"].notna().any():
            known_dates = df["date"].dropna().unique()
            if len(known_dates) == 1:
                df["date"] = df["date"].fillna(known_dates[0])
    return df

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
    if lines.shape[1] >= 4:
        lines['team_abbr'] = lines.iloc[:, 3].astype(str).str.upper()
    else:
        lines['team_abbr'] = ""

# Fetch schedule
schedule_df = fetch_schedule()

# ✅ UPDATED SECTION: Calendar-style date selector
if not schedule_df.empty and 'date' in schedule_df.columns:
    schedule_df = schedule_df.dropna(subset=['date'])
    unique_dates = sorted(schedule_df['date'].unique())

    selected_date = st.date_input(
        "Select a date to view games:",
        value=dt.datetime.now().date(),
        min_value=min(unique_dates),
        max_value=max(unique_dates)
    )

    schedule_df = schedule_df[schedule_df['date'] == selected_date]
else:
    st.warning("No schedule data found.")
    schedule_df = pd.DataFrame()

if schedule_df.empty:
    st.warning("No games on this date.")
else:
    st.header(f"Games on {selected_date.strftime('%A, %b %d')}")
    st.dataframe(schedule_df)

    # (Everything below this point is identical — unchanged)
    # -------------------------
    # Raw Data (expander with tabs)
    # -------------------------
    with st.expander("Raw Data (Full stats)"):
        tabs = st.tabs(["Lines", "Goalies", "Teams"])
        # ... (same as your original code for tabs and tables)
        # no edits beyond this point