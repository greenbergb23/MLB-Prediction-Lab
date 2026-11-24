from shiny import App, module, reactive, render, ui
from pathlib import Path
from functools import lru_cache
import json
import unicodedata

import pandas as pd
import numpy as np
import joblib
import requests


# ============================================================
# FILE PATHS
# ============================================================

APP_DIR = Path(__file__).parent

HOF_MODEL_PATH = APP_DIR / "hof_model.joblib"
HOF_PLAYERS_PATH = APP_DIR / "hof_players.joblib"

ALLMLB_MODEL_PATH = APP_DIR / "allmlb_model.joblib"
ALLMLB_LOOKUP_PATH = APP_DIR / "allmlb_player_lookup.csv"
ALLMLB_2026_PATH = APP_DIR / "allmlb_2026.csv"
ALLMLB_2027_PROJECTION_PATH = APP_DIR / "2027_steamer_merged.csv"

POSTSEASON_MODEL_PATH = APP_DIR / "postseason_model.joblib"
POSTSEASON_DATA_PATH = APP_DIR / "postseason_team_dataset.csv"


# ============================================================
# LOAD HALL OF FAME MODEL
# ============================================================

hof_bundle = joblib.load(HOF_MODEL_PATH)
players_df = joblib.load(HOF_PLAYERS_PATH)

hof_model = hof_bundle["model"]
HOF_THRESHOLD = hof_bundle["threshold"]
hof_features = hof_bundle["features"]

DATA_YEAR = int(
    players_df["last_season"].max()
)


# ============================================================
# LOAD ALL-MLB MODEL
# ============================================================

allmlb_bundle = joblib.load(
    ALLMLB_MODEL_PATH
)

allmlb_model = allmlb_bundle["model"]
ALLMLB_THRESHOLD = allmlb_bundle["threshold"]
allmlb_features = allmlb_bundle["features"]


# ============================================================
# LOAD POSTSEASON MODEL
# ============================================================

postseason_bundle = joblib.load(
    POSTSEASON_MODEL_PATH
)

postseason_threshold_models = (
    postseason_bundle["threshold_models"]
)

postseason_features = (
    postseason_bundle["features"]
)

POSTSEASON_OUTCOME_LABELS = (
    postseason_bundle["outcome_labels"]
)

POSTSEASON_THRESHOLD_LABELS = (
    postseason_bundle["threshold_labels"]
)


# ============================================================
# LOAD POSTSEASON DATA
# ============================================================

postseason_dataset = pd.read_csv(
    POSTSEASON_DATA_PATH
)

postseason_dataset["yearID"] = pd.to_numeric(
    postseason_dataset["yearID"],
    errors="coerce",
)

postseason_dataset = postseason_dataset[
    postseason_dataset["yearID"].notna()
].copy()

postseason_dataset["yearID"] = (
    postseason_dataset["yearID"]
    .astype(int)
)


# ============================================================
# NORMALIZE LEAGUE
# ============================================================

if "lgID" in postseason_dataset.columns:

    postseason_dataset["league"] = (
        postseason_dataset["lgID"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

elif "league" in postseason_dataset.columns:

    postseason_dataset["league"] = (
        postseason_dataset["league"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

else:

    raise ValueError(
        "postseason_team_dataset.csv must contain lgID or league."
    )


postseason_dataset["league"] = (
    postseason_dataset["league"]
    .replace(
        {
            "AMERICAN": "AL",
            "AMERICAN LEAGUE": "AL",
            "NATIONAL": "NL",
            "NATIONAL LEAGUE": "NL",
        }
    )
)


# ============================================================
# NORMALIZE DIVISION
# ============================================================

if "divID" not in postseason_dataset.columns:

    raise ValueError(
        "postseason_team_dataset.csv must contain divID."
    )


postseason_dataset["divID"] = (
    postseason_dataset["divID"]
    .astype(str)
    .str.upper()
    .str.strip()
    .replace(
        {
            "EAST": "E",
            "EASTERN": "E",
            "CENTRAL": "C",
            "WEST": "W",
            "WESTERN": "W",
        }
    )
)


# ============================================================
# ACTUAL POSTSEASON RESULTS
# ============================================================

if "postseason_result" not in postseason_dataset.columns:
    postseason_dataset["postseason_result"] = pd.NA

if "postseason_outcome" not in postseason_dataset.columns:
    postseason_dataset["postseason_outcome"] = pd.NA


POSTSEASON_DATA_YEAR = int(
    postseason_dataset["yearID"].max()
)


# ============================================================
# MLB TEAM LOGOS
# ============================================================

TEAM_MLBAM = {
    "ARI": 109,
    "ATL": 144,
    "BAL": 110,
    "BOS": 111,

    "CHA": 145,
    "CHW": 145,

    "CHN": 112,
    "CHC": 112,

    "CIN": 113,
    "CLE": 114,
    "COL": 115,
    "DET": 116,
    "HOU": 117,

    "KCA": 118,
    "KC": 118,

    "ANA": 108,
    "CAL": 108,
    "LAA": 108,

    "LAN": 119,
    "LAD": 119,

    "MIA": 146,
    "FLO": 146,

    "MIL": 158,
    "MIN": 142,

    "NYA": 147,
    "NYY": 147,

    "NYN": 121,
    "NYM": 121,

    "OAK": 133,
    "ATH": 133,

    "PHI": 143,
    "PIT": 134,

    "SDN": 135,
    "SDP": 135,

    "SEA": 136,

    "SFN": 137,
    "SFG": 137,

    "SLN": 138,
    "STL": 138,

    "TBA": 139,
    "TBD": 139,
    "TBR": 139,

    "TEX": 140,
    "TOR": 141,

    "WAS": 120,
    "WSN": 120,
    "MON": 120,
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_int(value):

    if pd.isna(value):
        return "N/A"

    return f"{int(round(float(value))):,}"


def safe_rate(value):

    if pd.isna(value):
        return "N/A"

    return f"{float(value):.3f}"


def format_stat(label, value):

    if pd.isna(value):
        return "N/A"

    if label in [
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "ISO",
    ]:

        return f"{float(value):.3f}"

    return f"{int(round(float(value))):,}"


def score_cell(label, value):

    return ui.div(

        ui.div(
            label,
            class_="score-cell-label"
        ),

        ui.div(
            value,
            class_="score-cell-value"
        ),

        class_="score-cell",
    )


# ============================================================
# LOGOS
# ============================================================

def team_logo_url(team_id):

    if pd.isna(team_id):
        return None

    team_id = (
        str(team_id)
        .upper()
        .strip()
    )

    mlbam = TEAM_MLBAM.get(team_id)

    if mlbam is None:
        return None

    return (
        f"https://www.mlbstatic.com/"
        f"team-logos/{mlbam}.svg"
    )


def team_logo(
    team_id,
    size="36px"
):

    url = team_logo_url(
        team_id
    )

    if url:

        return ui.tags.img(
            src=url,
            class_="team-logo",
            style=(
                f"width:{size};"
                f"height:{size};"
            ),
        )

    return ui.div(
        str(team_id),
        class_="team-logo-fallback",
        style=(
            f"width:{size};"
            f"height:{size};"
        ),
    )


# ============================================================
# PLAYER HEADSHOTS
# ============================================================

@lru_cache(maxsize=1000)
def get_mlbam_id(player_name):

    if not player_name:
        return None

    try:

        response = requests.get(
            "https://statsapi.mlb.com/api/v1/people/search",
            params={
                "names": player_name,
                "sportIds": 1,
            },
            timeout=5,
        )

        response.raise_for_status()

        people = (
            response
            .json()
            .get("people", [])
        )

        if not people:
            return None

        exact = [
            person
            for person in people
            if (
                person.get(
                    "fullName",
                    ""
                )
                .lower()
                .strip()
                ==
                player_name
                .lower()
                .strip()
            )
        ]

        return (
            exact[0]
            if exact
            else people[0]
        ).get("id")

    except Exception:

        return None


@lru_cache(maxsize=1000)
def get_headshot_url(player_name):

    mlbam_id = get_mlbam_id(
        player_name
    )

    if mlbam_id is None:
        return None

    return (
        "https://img.mlbstatic.com/"
        "mlb-photos/image/upload/"
        "d_people:generic:headshot:67:current.png/"
        "w_300,q_auto:best/"
        f"v1/people/{mlbam_id}/"
        "headshot/67/current"
    )


def headshot_panel(player_name):

    if player_name == "Custom Player":

        return ui.div(
            "CUSTOM",
            class_="headshot-placeholder"
        )

    url = get_headshot_url(
        player_name
    )

    if url is None:

        return ui.div(
            "NO PHOTO",
            class_="headshot-placeholder"
        )

    return ui.div(

        ui.tags.img(
            src=url,
            class_="player-headshot"
        ),

        class_="headshot-frame"
    )


# ============================================================
# ============================================================
# HALL OF FAME MODULE
# ============================================================
# ============================================================

eligible_players = players_df[
    (
        players_df["service_years"] >= 10
    )
    &
    (
        players_df["last_season"]
        <= DATA_YEAR - 5
    )
].copy()


recently_retired_players = players_df[
    (
        players_df["service_years"] >= 10
    )
    &
    (
        players_df["last_season"]
        < DATA_YEAR
    )
    &
    (
        players_df["last_season"]
        > DATA_YEAR - 5
    )
].copy()


current_players = players_df[
    players_df["last_season"]
    == DATA_YEAR
].copy()


eligible_names = sorted(
    eligible_players["Name"]
    .dropna()
    .unique()
)

recent_names = sorted(
    recently_retired_players["Name"]
    .dropna()
    .unique()
)

current_names = sorted(
    current_players["Name"]
    .dropna()
    .unique()
)


# ============================================================
# 2027 BBWAA HALL OF FAME BALLOT
# ============================================================

HOF_2027_BALLOT = [
    {"Name": "Jake Arrieta", "Ballot_Status": "First-Time"},
    {"Name": "Jay Bruce", "Ballot_Status": "First-Time"},
    {"Name": "Asdrúbal Cabrera", "Ballot_Status": "First-Time"},
    {"Name": "Starlin Castro", "Ballot_Status": "First-Time"},
    {"Name": "Wade Davis", "Ballot_Status": "First-Time"},
    {"Name": "Brett Gardner", "Ballot_Status": "First-Time"},
    {"Name": "Jon Lester", "Ballot_Status": "First-Time"},
    {"Name": "Andrew Miller", "Ballot_Status": "First-Time"},
    {"Name": "Buster Posey", "Ballot_Status": "First-Time"},
    {"Name": "Pablo Sandoval", "Ballot_Status": "First-Time"},
    {"Name": "Ervin Santana", "Ballot_Status": "First-Time"},
    {"Name": "Kyle Seager", "Ballot_Status": "First-Time"},
    {"Name": "Joakim Soria", "Ballot_Status": "First-Time"},
    {"Name": "Ryan Zimmerman", "Ballot_Status": "First-Time"},
    {"Name": "Bobby Abreu", "Ballot_Status": "Returning"},
    {"Name": "Mark Buehrle", "Ballot_Status": "Returning"},
    {"Name": "Cole Hamels", "Ballot_Status": "Returning"},
    {"Name": "Félix Hernández", "Ballot_Status": "Returning"},
    {"Name": "Torii Hunter", "Ballot_Status": "Returning"},
    {"Name": "Dustin Pedroia", "Ballot_Status": "Returning"},
    {"Name": "Andy Pettitte", "Ballot_Status": "Returning"},
    {"Name": "Alex Rodriguez", "Ballot_Status": "Returning"},
    {"Name": "Francisco Rodríguez", "Ballot_Status": "Returning"},
    {"Name": "Jimmy Rollins", "Ballot_Status": "Returning"},
    {"Name": "Chase Utley", "Ballot_Status": "Returning"},
    {"Name": "Omar Vizquel", "Ballot_Status": "Returning"},
    {"Name": "David Wright", "Ballot_Status": "Returning"},
]


def normalize_hof_name(name):

    if pd.isna(name):
        return ""

    value = str(name).strip().lower()
    value = unicodedata.normalize("NFKD", value)
    value = "".join(
        character
        for character in value
        if not unicodedata.combining(character)
    )

    # The Lahman-style player file generally does not preserve accents.
    # Normalizing both sides makes names such as Félix/Felix match.
    value = (
        value
        .replace(".", "")
        .replace("'", "")
        .replace("’", "")
        .replace("-", " ")
    )
    value = " ".join(value.split())
    return value


HOF_2027_BALLOT_DF = pd.DataFrame(HOF_2027_BALLOT)

_hof_player_name_lookup = {
    normalize_hof_name(name): name
    for name in players_df["Name"].dropna().astype(str).unique()
}


def get_2027_ballot_player(display_name):

    lookup_name = _hof_player_name_lookup.get(
        normalize_hof_name(display_name)
    )

    if lookup_name is None:
        return None

    player = players_df[
        players_df["Name"].astype(str) == lookup_name
    ].copy()

    if player.empty:
        return None

    # players_df is career-level, but iloc[[0]] also protects the UI
    # if a future source contains duplicate rows.
    return player.iloc[[0]]


def score_2027_ballot_candidate(display_name):

    player = get_2027_ballot_player(display_name)

    if player is None:
        return None, None, None

    try:
        probability, prediction = score_hof_player(player)
        return player, probability, prediction
    except Exception:
        return player, None, None


def hof_ballot_headshot(player_name):

    url = get_headshot_url(player_name)

    if url is None:
        return ui.div(
            "NO PHOTO",
            class_="hof-ballot-headshot-placeholder"
        )

    return ui.div(
        ui.tags.img(
            src=url,
            class_="hof-ballot-headshot-img"
        ),
        class_="hof-ballot-headshot"
    )


def make_hof_2027_ballot_board(scored_candidates):

    if scored_candidates.empty:
        return ui.div(
            "No 2027 ballot hitters match the current filters.",
            class_="allmlb-explanation-unavailable"
        )

    cards = []

    for rank, (_, candidate) in enumerate(
        scored_candidates.iterrows(),
        start=1,
    ):

        probability = float(candidate["probability"])
        status = str(candidate["Ballot_Status"])
        designation = str(candidate["Designation"])
        position_group = str(candidate["position_group"])

        designation_class = (
            "hof-ballot-positive"
            if probability >= HOF_THRESHOLD
            else "hof-ballot-negative"
        )

        cards.append(
            ui.div(
                ui.div(
                    hof_ballot_headshot(
                        candidate["Name"]
                    ),
                    ui.div(
                        ui.div(
                            ui.div(
                                f"#{rank}",
                                class_="hof-ballot-rank"
                            ),
                            ui.div(
                                status.upper(),
                                class_=(
                                    "hof-ballot-status "
                                    + (
                                        "hof-ballot-first"
                                        if status == "First-Time"
                                        else "hof-ballot-returning"
                                    )
                                )
                            ),
                            ui.div(
                                position_group,
                                class_="hof-ballot-position"
                            ),
                            class_="hof-ballot-card-top"
                        ),
                        ui.div(
                            candidate["Name"],
                            class_="hof-ballot-player"
                        ),
                        class_="hof-ballot-card-info"
                    ),
                    class_="hof-ballot-player-header"
                ),
                ui.div(
                    ui.div(
                        "MODEL HOF PROBABILITY",
                        class_="hof-ballot-prob-label"
                    ),
                    ui.div(
                        f"{probability * 100:.1f}%",
                        class_="hof-ballot-prob"
                    ),
                    class_="hof-ballot-prob-wrap"
                ),
                ui.div(
                    designation,
                    class_=(
                        "hof-ballot-designation "
                        f"{designation_class}"
                    )
                ),
                class_="hof-ballot-card"
            )
        )

    return ui.div(
        ui.div(
            "MODEL RANKING OF 2027 BBWAA BALLOT HITTERS",
            class_="hof-ballot-board-title"
        ),
        ui.div(
            (
                "Only hitters with a valid saved-model probability are "
                "shown. The percentage reflects career-profile similarity "
                "to historical Hall of Fame outcomes; it is not a forecast "
                "of BBWAA vote share."
            ),
            class_="hof-ballot-board-note"
        ),
        ui.div(
            *cards,
            class_="hof-ballot-grid"
        ),
        class_="hof-ballot-board"
    )


custom_reference = players_df[
    players_df["service_years"] >= 10
].copy()


custom_mean_cols = [
    "service_years",
    "AB",
    "H",
    "HR",
    "BB",
    "Doubles",
    "Triples",
    "SB",
    "AVG",
    "OBP",
    "SLG",
    "MVP",
    "Gold_Glove",
    "Platinum_Glove",
    "Silver_Slugger",
]


position_means = (
    custom_reference
    .groupby("position")[
        custom_mean_cols
    ]
    .mean()
    .round(3)
)


def prepare_player(player):

    player = player.copy()

    player["3000_H"] = (
        player["H"] >= 3000
    ).astype(int)

    return player


def score_hof_player(player):

    player = prepare_player(
        player
    )

    probability = (
        hof_model
        .predict_proba(
            player[
                hof_features
            ]
        )[:, 1][0]
    )

    return (
        probability,
        int(
            probability
            >= HOF_THRESHOLD
        )
    )


def hof_designation(probability):

    if probability >= .75:
        return "STRONG HOF PROFILE"

    if probability >= HOF_THRESHOLD:
        return "HOF RANGE"

    if probability >= .10:
        return "BORDERLINE"

    return "LONG SHOT"


def get_hof_2027_position_group(position):

    if pd.isna(position):
        return "N/A"

    position = str(position).upper().strip()

    if position in ["LF", "CF", "RF", "OF"]:
        return "OF"

    return position


def get_hof_2027_scored_ballot():

    rows = []

    for candidate in HOF_2027_BALLOT:

        player, probability, prediction = (
            score_2027_ballot_candidate(
                candidate["Name"]
            )
        )

        # This ballot view is intentionally hitter-only.
        # Pitchers / unscorable careers produce no usable probability
        # with the saved hitter Hall-of-Fame model and are excluded.
        if (
            player is None
            or probability is None
            or not np.isfinite(float(probability))
        ):
            continue

        player_row = player.iloc[0]
        raw_position = player_row.get(
            "position",
            "N/A"
        )
        position_group = get_hof_2027_position_group(
            raw_position
        )

        # Additional protection against pitcher-style position labels.
        if position_group in [
            "P", "SP", "RP", "RHP", "LHP"
        ]:
            continue

        probability = float(probability)

        rows.append(
            {
                "Name": candidate["Name"],
                "Ballot_Status": candidate["Ballot_Status"],
                "position": raw_position,
                "position_group": position_group,
                "probability": probability,
                "prediction": int(prediction),
                "Designation": hof_designation(probability),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Name",
                "Ballot_Status",
                "position",
                "position_group",
                "probability",
                "prediction",
                "Designation",
            ]
        )

    return pd.DataFrame(rows)


def make_hof_scoreboard(
    player,
    probability,
    prediction,
    note
):

    player = prepare_player(
        player
    )

    row = player.iloc[0]

    name = row.get(
        "Name",
        "CUSTOM PLAYER"
    )

    position = row.get(
        "position",
        "N/A"
    )

    service = row.get(
        "service_years",
        "N/A"
    )

    last_season = row.get(
        "last_season",
        np.nan
    )


    if pd.isna(last_season):

        subtitle = (
            f"POSITION {position} • "
            f"SERVICE {service} YEARS"
        )

    else:

        subtitle = (
            f"POSITION {position} • "
            f"SERVICE {service} YEARS • "
            f"LAST SEASON {int(last_season)}"
        )


    stats = [
        ("AB", row.get("AB")),
        ("H", row.get("H")),
        ("HR", row.get("HR")),
        ("BB", row.get("BB")),
        ("SB", row.get("SB")),
        ("2B", row.get("Doubles")),
        ("3B", row.get("Triples")),
        ("AVG", row.get("AVG")),
        ("OBP", row.get("OBP")),
        ("SLG", row.get("SLG")),
        ("MVP", row.get("MVP")),
        (
            "GOLD GLOVE",
            row.get("Gold_Glove")
        ),
        (
            "PLATINUM GLOVE",
            row.get("Platinum_Glove")
        ),
        (
            "SILVER SLUGGER",
            row.get("Silver_Slugger")
        ),
        (
            "3000 H",
            (
                "YES"
                if row.get("3000_H", 0) == 1
                else "NO"
            )
        ),
    ]


    return ui.div(

        ui.div(

            ui.div(
                headshot_panel(
                    name
                ),
                class_="marquee-photo-column"
            ),

            ui.div(

                ui.div(
                    "HALL OF FAME SCOREBOARD",
                    class_="marquee-small"
                ),

                ui.div(
                    str(name).upper(),
                    class_="marquee-name"
                ),

                ui.div(
                    subtitle,
                    class_="marquee-subtitle"
                ),

                class_="marquee-info-column"
            ),

            class_="player-marquee"
        ),

        ui.div(
            "CAREER STATISTICS",
            class_="score-section-title"
        ),

        ui.div(
            *[
                score_cell(
                    label,
                    (
                        value
                        if label == "3000 H"
                        else format_stat(
                            label,
                            value
                        )
                    )
                )
                for label, value
                in stats
            ],
            class_="score-grid"
        ),

        ui.div(
            "MODEL OUTPUT",
            class_="score-section-title"
        ),

        ui.div(

            ui.div(

                ui.div(
                    "HOF MODEL %",
                    class_="big-panel-label"
                ),

                ui.div(
                    f"{probability * 100:.1f}%",
                    class_="big-panel-number"
                ),

                class_="big-score-panel"
            ),

            ui.div(

                ui.div(
                    "DESIGNATION",
                    class_="big-panel-label"
                ),

                ui.div(
                    hof_designation(
                        probability
                    ),
                    class_=(
                        "designation-on"
                        if prediction == 1
                        else "designation-off"
                    ),
                ),

                class_="big-score-panel"
            ),

            class_="big-score-grid"
        ),

        ui.div(
            (
                "MODEL CLASSIFICATION THRESHOLD: "
                f"{HOF_THRESHOLD:.2f}"
            ),
            class_="threshold-board"
        ),

        ui.div(
            note,
            class_="scoreboard-note"
        ),

        class_="scoreboard"
    )


@module.ui
def hof_ui():

    return ui.div(

        ui.div(

            ui.h1(
                "HALL OF FAME PREDICTOR",
                class_="dashboard-title"
            ),

            ui.p(
                (
                    "2027 BBWAA ballot • "
                    "Historical careers • "
                    "Recently retired • "
                    "Current players • "
                    "Custom simulation"
                ),
                class_="dashboard-subtitle"
            ),

            class_="dashboard-header"
        ),

        ui.div(

            ui.h2(
                "HOW TO USE THE HALL OF FAME PREDICTOR",
                class_="instruction-title"
            ),

            ui.p(
                "This dashboard evaluates MLB career profiles "
                "using a logistic regression model trained on "
                "historical Hall of Fame outcomes."
            ),

            ui.div(

                ui.div(
                    ui.strong(
                        "1. Choose a player group"
                    ),
                    ui.p(
                        "Explore eligible, recently retired, "
                        "current, or custom careers."
                    ),
                ),

                ui.div(
                    ui.strong(
                        "2. Select a player"
                    ),
                    ui.p(
                        "The saved model scores the player's "
                        "career profile."
                    ),
                ),

                ui.div(
                    ui.strong(
                        "3. Review production"
                    ),
                    ui.p(
                        "See career statistics and awards."
                    ),
                ),

                ui.div(
                    ui.strong(
                        "4. Model probability"
                    ),
                    ui.p(
                        "The percentage measures similarity "
                        "to historical Hall of Fame careers."
                    ),
                ),

                class_="instruction-grid"
            ),

            class_="instruction-panel"
        ),

        ui.navset_tab(

            ui.nav_panel(

                "2027 BBWAA Ballot",

                ui.div(

                    ui.h3(
                        "2027 HALL OF FAME BALLOT — HITTERS",
                        class_="hof-ballot-section-title"
                    ),

                    ui.p(
                        (
                            "Explore the hitter candidates on the 2027 "
                            "BBWAA ballot through the saved Hall of Fame "
                            "model. Pitchers and candidates without a valid "
                            "model probability are excluded."
                        ),
                        class_="hof-ballot-section-description"
                    ),

                    ui.div(

                        ui.layout_columns(

                            ui.input_select(
                                "hof_2027_status_filter",
                                "Ballot Status",
                                choices=[
                                    "All",
                                    "First-Time",
                                    "Returning",
                                ],
                                selected="All",
                            ),

                            ui.input_select(
                                "hof_2027_position_filter",
                                "Position",
                                choices=[
                                    "All",
                                    "C",
                                    "1B",
                                    "2B",
                                    "3B",
                                    "SS",
                                    "OF",
                                    "DH",
                                ],
                                selected="All",
                            ),

                            ui.input_select(
                                "hof_2027_designation_filter",
                                "Model Designation",
                                choices=[
                                    "All",
                                    "STRONG HOF PROFILE",
                                    "HOF RANGE",
                                    "BORDERLINE",
                                    "LONG SHOT",
                                ],
                                selected="All",
                            ),

                            ui.input_select(
                                "hof_2027_sort",
                                "Sort By",
                                choices={
                                    "prob_desc": "Probability: High → Low",
                                    "prob_asc": "Probability: Low → High",
                                    "name_asc": "Name: A → Z",
                                    "name_desc": "Name: Z → A",
                                },
                                selected="prob_desc",
                            ),

                            col_widths=[3, 3, 3, 3],
                        ),

                        ui.output_ui(
                            "hof_2027_filter_summary"
                        ),

                        class_="hof-ballot-filter-panel"
                    ),

                    ui.output_ui(
                        "hof_2027_ballot_board"
                    ),

                    class_="hof-ballot-intro"
                ),

                ui.div(

                    ui.h3(
                        "CANDIDATE DETAIL",
                        class_="hof-ballot-detail-title"
                    ),

                    ui.output_ui(
                        "hof_2027_candidate_selector"
                    ),

                    class_="selection-panel"
                ),

                ui.output_ui(
                    "hof_2027_scoreboard"
                ),
            ),

            ui.nav_panel(

                "Hall Eligible",

                ui.div(

                    ui.input_selectize(
                        "eligible_player",
                        "Select Player",
                        choices=eligible_names,
                        selected=(
                            eligible_names[0]
                            if eligible_names
                            else None
                        ),
                    ),

                    class_="selection-panel"
                ),

                ui.output_ui(
                    "eligible_scoreboard"
                ),
            ),

            ui.nav_panel(

                "Recently Retired",

                ui.div(

                    ui.input_selectize(
                        "recent_player",
                        "Select Player",
                        choices=recent_names,
                        selected=(
                            recent_names[0]
                            if recent_names
                            else None
                        ),
                    ),

                    class_="selection-panel"
                ),

                ui.output_ui(
                    "recent_scoreboard"
                ),
            ),

            ui.nav_panel(

                "Current Players",

                ui.div(

                    ui.input_selectize(
                        "current_player",
                        "Select Player",
                        choices=current_names,
                        selected=(
                            current_names[0]
                            if current_names
                            else None
                        ),
                    ),

                    class_="selection-panel"
                ),

                ui.output_ui(
                    "current_scoreboard"
                ),
            ),

            ui.nav_panel(

                "Custom Career",

                ui.div(

                    ui.layout_columns(

                        ui.input_select(
                            "custom_position",
                            "Position",
                            choices=[
                                "C",
                                "1B",
                                "2B",
                                "3B",
                                "SS",
                                "LF",
                                "CF",
                                "RF",
                                "DH",
                            ],
                            selected="RF",
                        ),

                        ui.input_numeric(
                            "custom_service",
                            "Service Years",
                            value=15,
                            min=1,
                        ),

                        ui.input_numeric(
                            "custom_ab",
                            "At-Bats",
                            value=7500,
                            min=0,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_h",
                            "Hits",
                            value=2200,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_hr",
                            "Home Runs",
                            value=350,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_bb",
                            "Walks",
                            value=900,
                            min=0,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_doubles",
                            "Doubles",
                            value=400,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_triples",
                            "Triples",
                            value=30,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_sb",
                            "Stolen Bases",
                            value=100,
                            min=0,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_avg",
                            "AVG",
                            value=.290,
                            step=.001,
                        ),

                        ui.input_numeric(
                            "custom_obp",
                            "OBP",
                            value=.370,
                            step=.001,
                        ),

                        ui.input_numeric(
                            "custom_slg",
                            "SLG",
                            value=.500,
                            step=.001,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_mvp",
                            "MVP Awards",
                            value=0,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_gold",
                            "Gold Gloves",
                            value=0,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_platinum",
                            "Platinum Gloves",
                            value=0,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_silver",
                            "Silver Sluggers",
                            value=0,
                            min=0,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.input_action_button(
                        "predict_custom",
                        "RUN HALL OF FAME MODEL",
                        class_="predict-button",
                    ),

                    class_="custom-panel"
                ),

                ui.output_ui(
                    "custom_scoreboard"
                ),
            ),
        ),
    )


@module.server
def hof_server(
    input,
    output,
    session
):

    def get_player(
        dataframe,
        name
    ):

        if not name:
            return None

        result = dataframe[
            dataframe["Name"]
            == name
        ].copy()

        if result.empty:
            return None

        return result.iloc[[0]]


    # ========================================================
    # 2027 BBWAA BALLOT
    # ========================================================

    @reactive.calc
    def hof_2027_scored_ballot():

        return get_hof_2027_scored_ballot()


    @reactive.calc
    def hof_2027_filtered_ballot():

        df = hof_2027_scored_ballot().copy()

        status = input.hof_2027_status_filter()
        position = input.hof_2027_position_filter()
        designation = input.hof_2027_designation_filter()
        sort_mode = input.hof_2027_sort()

        if status and status != "All":
            df = df[
                df["Ballot_Status"] == status
            ].copy()

        if position and position != "All":
            df = df[
                df["position_group"] == position
            ].copy()

        if designation and designation != "All":
            df = df[
                df["Designation"] == designation
            ].copy()

        if sort_mode == "prob_asc":
            df = df.sort_values(
                ["probability", "Name"],
                ascending=[True, True],
            )

        elif sort_mode == "name_asc":
            df = df.sort_values(
                "Name",
                ascending=True,
            )

        elif sort_mode == "name_desc":
            df = df.sort_values(
                "Name",
                ascending=False,
            )

        else:
            df = df.sort_values(
                ["probability", "Name"],
                ascending=[False, True],
            )

        return df.reset_index(drop=True)


    @output
    @render.ui
    def hof_2027_filter_summary():

        total = len(hof_2027_scored_ballot())
        shown = len(hof_2027_filtered_ballot())

        return ui.div(
            f"Showing {shown} of {total} hitter candidates",
            class_="hof-ballot-filter-summary"
        )


    @output
    @render.ui
    def hof_2027_ballot_board():

        return make_hof_2027_ballot_board(
            hof_2027_filtered_ballot()
        )


    @output
    @render.ui
    def hof_2027_candidate_selector():

        df = hof_2027_scored_ballot().copy()

        if df.empty:
            return ui.div(
                "No hitter candidates are available.",
                class_="allmlb-explanation-unavailable"
            )

        names = sorted(
            df["Name"]
            .astype(str)
            .unique()
        )

        default_name = (
            "Buster Posey"
            if "Buster Posey" in names
            else names[0]
        )

        return ui.input_selectize(
            session.ns("hof_2027_player"),
            "Select 2027 Ballot Hitter",
            choices=names,
            selected=default_name,
        )


    @output
    @render.ui
    def hof_2027_scoreboard():

        try:
            selected_name = input.hof_2027_player()
        except Exception:
            selected_name = None

        valid_df = hof_2027_scored_ballot()

        if valid_df.empty:
            return None

        valid_names = (
            valid_df["Name"]
            .astype(str)
            .tolist()
        )

        if (
            not selected_name
            or str(selected_name) not in valid_names
        ):
            selected_name = (
                "Buster Posey"
                if "Buster Posey" in valid_names
                else valid_names[0]
            )

        candidate = next(
            (
                row
                for row in HOF_2027_BALLOT
                if row["Name"] == selected_name
            ),
            None,
        )

        player, probability, prediction = (
            score_2027_ballot_candidate(
                selected_name
            )
        )

        if (
            player is None
            or probability is None
            or not np.isfinite(float(probability))
        ):
            return None

        ballot_status = (
            candidate["Ballot_Status"]
            if candidate is not None
            else "2027 Ballot"
        )

        return make_hof_scoreboard(
            player,
            probability,
            prediction,
            (
                f"2027 BBWAA ballot hitter • {ballot_status}. "
                "The model percentage reflects career-profile similarity "
                "to historical Hall of Fame outcomes; it is not a "
                "prediction of BBWAA vote percentage."
            ),
        )


    @output
    @render.ui
    def eligible_scoreboard():

        player = get_player(
            eligible_players,
            input.eligible_player(),
        )

        if player is None:
            return None

        probability, prediction = (
            score_hof_player(
                player
            )
        )

        return make_hof_scoreboard(
            player,
            probability,
            prediction,
            "Hall of Fame eligible player.",
        )


    @output
    @render.ui
    def recent_scoreboard():

        player = get_player(
            recently_retired_players,
            input.recent_player(),
        )

        if player is None:
            return None

        probability, prediction = (
            score_hof_player(
                player
            )
        )

        return make_hof_scoreboard(
            player,
            probability,
            prediction,
            "Recently retired player.",
        )


    @output
    @render.ui
    def current_scoreboard():

        player = get_player(
            current_players,
            input.current_player(),
        )

        if player is None:
            return None

        probability, prediction = (
            score_hof_player(
                player
            )
        )

        return make_hof_scoreboard(
            player,
            probability,
            prediction,
            f"Career-to-date through {DATA_YEAR}.",
        )


    @reactive.effect
    def update_custom_position():

        position = input.custom_position()

        if (
            not position
            or position
            not in position_means.index
        ):
            return

        means = position_means.loc[
            position
        ]

        mapping = {
            "custom_service":
                int(round(means["service_years"])),

            "custom_ab":
                int(round(means["AB"])),

            "custom_h":
                int(round(means["H"])),

            "custom_hr":
                int(round(means["HR"])),

            "custom_bb":
                int(round(means["BB"])),

            "custom_doubles":
                int(round(means["Doubles"])),

            "custom_triples":
                int(round(means["Triples"])),

            "custom_sb":
                int(round(means["SB"])),

            "custom_avg":
                round(means["AVG"], 3),

            "custom_obp":
                round(means["OBP"], 3),

            "custom_slg":
                round(means["SLG"], 3),

            "custom_mvp":
                round(means["MVP"], 2),

            "custom_gold":
                round(means["Gold_Glove"], 2),

            "custom_platinum":
                round(means["Platinum_Glove"], 2),

            "custom_silver":
                round(means["Silver_Slugger"], 2),
        }

        for input_id, value in mapping.items():

            ui.update_numeric(
                input_id,
                value=value,
                session=session,
            )


    @reactive.calc
    @reactive.event(
        input.predict_custom
    )
    def custom_player():

        return pd.DataFrame(
            {
                "Name": [
                    "Custom Player"
                ],

                "position": [
                    input.custom_position()
                ],

                "service_years": [
                    input.custom_service()
                ],

                "last_season": [
                    np.nan
                ],

                "AB": [
                    input.custom_ab()
                ],

                "H": [
                    input.custom_h()
                ],

                "HR": [
                    input.custom_hr()
                ],

                "BB": [
                    input.custom_bb()
                ],

                "Doubles": [
                    input.custom_doubles()
                ],

                "Triples": [
                    input.custom_triples()
                ],

                "SB": [
                    input.custom_sb()
                ],

                "AVG": [
                    input.custom_avg()
                ],

                "OBP": [
                    input.custom_obp()
                ],

                "SLG": [
                    input.custom_slg()
                ],

                "MVP": [
                    input.custom_mvp()
                ],

                "Gold_Glove": [
                    input.custom_gold()
                ],

                "Platinum_Glove": [
                    input.custom_platinum()
                ],

                "Silver_Slugger": [
                    input.custom_silver()
                ],
            }
        )


    @output
    @render.ui
    def custom_scoreboard():

        player = custom_player()

        probability, prediction = (
            score_hof_player(
                player
            )
        )

        return make_hof_scoreboard(
            player,
            probability,
            prediction,
            "Custom career simulation.",
        )


# ============================================================
# ============================================================
# ALL-MLB MODULE
# ============================================================
# ============================================================

def get_allmlb_position_group(position):

    if pd.isna(position):
        return "N/A"

    position = (
        str(position)
        .upper()
        .strip()
    )

    if position in [
        "LF",
        "CF",
        "RF",
        "OF",
    ]:

        return "OF"

    return position


ALLMLB_POSITION_ORDER = [
    "C",
    "1B",
    "2B",
    "3B",
    "SS",
    "OF",
    "DH",
]


def normalize_allmlb_dataframe(
    dataframe,
    source_name="All-MLB"
):

    dataframe = dataframe.copy()

    required = [
        "Name",
        "yearID",
        "primary_position",
        "All_MLB",
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{source_name} missing columns: {missing}"
        )


    dataframe["Name"] = (
        dataframe["Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["yearID"] = pd.to_numeric(
        dataframe["yearID"],
        errors="coerce",
    )

    dataframe = dataframe[
        dataframe["yearID"].notna()
    ].copy()

    dataframe["yearID"] = (
        dataframe["yearID"]
        .astype(int)
    )

    dataframe["All_MLB"] = (
        pd.to_numeric(
            dataframe["All_MLB"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )

    dataframe["primary_position"] = (
        dataframe["primary_position"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    dataframe["position_group"] = (
        dataframe["primary_position"]
        .apply(
            get_allmlb_position_group
        )
    )


    if "All_MLB_Team" not in dataframe.columns:

        dataframe["All_MLB_Team"] = np.where(
            dataframe["All_MLB"] == 1,
            "Selected",
            "Not Selected",
        )


    if "award_position" not in dataframe.columns:

        dataframe["award_position"] = pd.NA


    dataframe["award_position"] = (
        dataframe["award_position"]
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "None": pd.NA,
            }
        )
    )


    dataframe["selection_position"] = (
        dataframe["position_group"]
    )


    award_mask = (
        (
            dataframe["All_MLB"] == 1
        )
        &
        dataframe[
            "award_position"
        ].notna()
    )


    dataframe.loc[
        award_mask,
        "selection_position"
    ] = (
        dataframe.loc[
            award_mask,
            "award_position"
        ]
        .apply(
            get_allmlb_position_group
        )
    )


    return dataframe


allmlb_lookup = normalize_allmlb_dataframe(
    pd.read_csv(
        ALLMLB_LOOKUP_PATH
    ),
    "allmlb_player_lookup.csv",
)


if ALLMLB_2026_PATH.exists():

    allmlb_2026 = normalize_allmlb_dataframe(
        pd.read_csv(
            ALLMLB_2026_PATH
        ),
        "allmlb_2026.csv",
    )

else:

    allmlb_2026 = pd.DataFrame()


historical_allmlb = allmlb_lookup[
    allmlb_lookup["yearID"] != 2026
].copy()


if not allmlb_2026.empty:

    allmlb_all_seasons = pd.concat(
        [
            historical_allmlb,
            allmlb_2026,
        ],
        ignore_index=True,
        sort=False,
    )

else:

    allmlb_all_seasons = (
        historical_allmlb.copy()
    )


# ============================================================
# LOAD 2027 STEAMER ALL-MLB PROJECTION DATA
# ============================================================

PROJECTED_ALLMLB_MIN_PA = 300


def normalize_allmlb_projection_dataframe(dataframe):

    dataframe = dataframe.copy()

    required = [
        "Name",
        "Team",
        "primary_position",
        "PA",
        *allmlb_features,
    ]

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            "2027_steamer_merged.csv is missing columns: "
            f"{sorted(set(missing))}"
        )

    dataframe["Name"] = (
        dataframe["Name"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    dataframe["Team"] = (
        dataframe["Team"]
        .fillna("FA")
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": "FA", "NAN": "FA", "NONE": "FA"})
    )

    dataframe["primary_position"] = (
        dataframe["primary_position"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    dataframe["position_group"] = (
        dataframe["primary_position"]
        .apply(get_allmlb_position_group)
    )

    dataframe["yearID"] = 2027

    # These columns make projected rows compatible with the
    # existing All-MLB card/explanation helpers.
    dataframe["All_MLB"] = 0
    dataframe["All_MLB_Team"] = "Projected"
    dataframe["selection_position"] = dataframe["position_group"]

    return dataframe


if ALLMLB_2027_PROJECTION_PATH.exists():

    allmlb_2027_projection = (
        normalize_allmlb_projection_dataframe(
            pd.read_csv(ALLMLB_2027_PROJECTION_PATH)
        )
    )

else:

    allmlb_2027_projection = pd.DataFrame()


def score_allmlb_player(player):

    probability = (
        allmlb_model
        .predict_proba(
            player[
                allmlb_features
            ]
        )[:, 1][0]
    )

    return (
        probability,
        int(
            probability
            >= ALLMLB_THRESHOLD
        )
    )


def score_allmlb_dataframe(
    dataframe
):

    if dataframe.empty:
        return dataframe.copy()

    scored = dataframe.copy()

    scored["Model_Probability"] = (
        allmlb_model
        .predict_proba(
            scored[
                allmlb_features
            ]
        )[:, 1]
    )

    scored["Model_Prediction"] = (
        scored["Model_Probability"]
        >= ALLMLB_THRESHOLD
    ).astype(int)

    return scored

# ============================================================
# ALL-MLB MODEL EXPLANATION HELPERS
# ============================================================

ALLMLB_FEATURE_DISPLAY = {
    "PA": "Plate Appearances",
    "H": "Hits",
    "HR": "Home Runs",
    "RBI": "RBI",
    "BB": "Walks",
    "SB": "Stolen Bases",
    "AVG": "AVG",
    "OBP": "OBP",
    "SLG": "SLG",
    "OPS": "OPS",
    "ISO": "ISO",
    "HR_rate": "HR Rate",
    "BB_rate": "BB Rate",
    "SO_rate": "SO Rate",
    "primary_position": "Position",
}


def allmlb_feature_label(feature):

    return ALLMLB_FEATURE_DISPLAY.get(
        feature,
        str(feature)
        .replace("_", " ")
        .title()
    )


def format_allmlb_explanation_value(
    feature,
    value
):

    if pd.isna(value):
        return "N/A"

    if feature == "primary_position":
        return str(value)

    if feature in [
        "PA",
        "H",
        "HR",
        "RBI",
        "BB",
        "SB",
    ]:

        return f"{int(round(float(value))):,}"

    if feature in [
        "AVG",
        "OBP",
        "SLG",
        "OPS",
        "ISO",
    ]:

        return f"{float(value):.3f}"

    # All-MLB rate features are stored as percentage-point values
    # in the All-MLB dataset/model.
    if feature in [
        "HR_rate",
        "BB_rate",
        "SO_rate",
    ]:

        return f"{float(value):.1f}%"

    try:

        return f"{float(value):.3f}"

    except Exception:

        return str(value)


def allmlb_pairwise_feature_effects(
    selected_row,
    snub_row
):

    """
    Measures how each feature difference contributes to the
    probability separation between two players.

    For each feature:
      1. Give the selected player the snub's value.
      2. Give the snub the selected player's value.
      3. Re-score both counterfactual seasons.
      4. Average the two changes in the probability gap.

    Positive effect:
        the selected player's value is more favorable.

    Negative effect:
        the snub's value is more favorable.

    This is a model sensitivity explanation, not a causal claim.
    """

    base_selected_probability = float(
        selected_row.get(
            "Model_Probability",
            0
        )
    )

    base_snub_probability = float(
        snub_row.get(
            "Model_Probability",
            0
        )
    )

    results = []


    for feature in allmlb_features:

        if (
            feature not in selected_row.index
            or feature not in snub_row.index
        ):
            continue


        selected_value = selected_row.get(
            feature
        )

        snub_value = snub_row.get(
            feature
        )


        # Skip features that are identical.
        try:

            if (
                pd.isna(selected_value)
                and pd.isna(snub_value)
            ):
                continue

        except Exception:

            pass


        try:

            if selected_value == snub_value:
                continue

        except Exception:

            pass


        try:

            # ---------------------------------------------
            # SELECTED PLAYER WITH SNUB'S FEATURE VALUE
            # ---------------------------------------------

            selected_counterfactual = (
                selected_row.copy()
            )

            selected_counterfactual[
                feature
            ] = snub_value


            selected_counterfactual_df = (
                pd.DataFrame(
                    [
                        selected_counterfactual
                    ]
                )
            )


            selected_swapped_probability = float(
                allmlb_model
                .predict_proba(
                    selected_counterfactual_df[
                        allmlb_features
                    ]
                )[:, 1][0]
            )


            # ---------------------------------------------
            # SNUB WITH SELECTED PLAYER'S FEATURE VALUE
            # ---------------------------------------------

            snub_counterfactual = (
                snub_row.copy()
            )

            snub_counterfactual[
                feature
            ] = selected_value


            snub_counterfactual_df = (
                pd.DataFrame(
                    [
                        snub_counterfactual
                    ]
                )
            )


            snub_swapped_probability = float(
                allmlb_model
                .predict_proba(
                    snub_counterfactual_df[
                        allmlb_features
                    ]
                )[:, 1][0]
            )


            # ---------------------------------------------
            # SYMMETRIC EFFECT ON PROBABILITY GAP
            # ---------------------------------------------

            selected_side_effect = (
                base_selected_probability
                -
                selected_swapped_probability
            )


            snub_side_effect = (
                snub_swapped_probability
                -
                base_snub_probability
            )


            effect = (
                selected_side_effect
                +
                snub_side_effect
            ) / 2


            results.append(
                {
                    "feature":
                        feature,

                    "selected_value":
                        selected_value,

                    "snub_value":
                        snub_value,

                    "effect":
                        effect,

                    "abs_effect":
                        abs(effect),
                }
            )


        except Exception:

            # If one feature cannot be counterfactually
            # evaluated, simply omit it from the explanation.
            continue


    if not results:

        return pd.DataFrame()


    return (
        pd.DataFrame(
            results
        )
        .sort_values(
            "abs_effect",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def make_allmlb_model_explanation(
    actual,
    snubs,
    explanation_subtitle=None,
    snub_label="#1 MODEL SNUB",
    selected_label="ACTUAL SELECTION",
):

    if (
        actual.empty
        or snubs.empty
    ):

        return ui.div()


    if explanation_subtitle is None:

        explanation_subtitle = (
            "The comparison automatically uses the "
            "lowest-probability actual selection and "
            "the #1 model snub at this position."
        )


    # ========================================================
    # EXPLANATION MATCHUP
    #
    # Compare:
    #   lowest-probability actual selection
    # vs
    #   highest-probability non-selected player
    #
    # This is generally the most interesting comparison.
    # ========================================================

    selected_player = (
        actual
        .sort_values(
            "Model_Probability",
            ascending=True,
        )
        .iloc[0]
    )


    snub_player = (
        snubs
        .sort_values(
            "Model_Probability",
            ascending=False,
        )
        .iloc[0]
    )


    selected_probability = float(
        selected_player[
            "Model_Probability"
        ]
    )


    snub_probability = float(
        snub_player[
            "Model_Probability"
        ]
    )


    probability_gap = (
        snub_probability
        -
        selected_probability
    )


    drivers = (
        allmlb_pairwise_feature_effects(
            selected_player,
            snub_player,
        )
    )


    # ========================================================
    # MODEL FEATURE CHIPS
    # ========================================================

    feature_chips = [
        ui.span(
            allmlb_feature_label(
                feature
            ),
            class_="allmlb-feature-chip"
        )

        for feature in allmlb_features
    ]


    # ========================================================
    # DRIVER ROWS
    # ========================================================

    driver_rows = []


    if not drivers.empty:

        top_drivers = drivers.head(6)

        max_effect = max(
            float(
                top_drivers[
                    "abs_effect"
                ].max()
            ),
            1e-9,
        )


        for _, driver in (
            top_drivers.iterrows()
        ):

            feature = driver[
                "feature"
            ]


            effect = float(
                driver[
                    "effect"
                ]
            )


            width = (
                abs(effect)
                /
                max_effect
                *
                100
            )


            if effect > 0:

                edge_name = selected_player[
                    "Name"
                ]

                edge_class = (
                    "allmlb-driver-selected"
                )

            elif effect < 0:

                edge_name = snub_player[
                    "Name"
                ]

                edge_class = (
                    "allmlb-driver-snub"
                )

            else:

                edge_name = "EVEN"

                edge_class = (
                    "allmlb-driver-even"
                )


            driver_rows.append(

                ui.div(

                    # FEATURE
                    ui.div(
                        allmlb_feature_label(
                            feature
                        ),
                        class_="allmlb-driver-feature"
                    ),

                    # SELECTED PLAYER VALUE
                    ui.div(
                        format_allmlb_explanation_value(
                            feature,
                            driver[
                                "selected_value"
                            ],
                        ),
                        class_="allmlb-driver-value"
                    ),

                    # SNUB VALUE
                    ui.div(
                        format_allmlb_explanation_value(
                            feature,
                            driver[
                                "snub_value"
                            ],
                        ),
                        class_="allmlb-driver-value"
                    ),

                    # MODEL EFFECT
                    ui.div(

                        ui.div(

                            ui.div(
                                class_=(
                                    "allmlb-driver-bar-fill "
                                    f"{edge_class}"
                                ),
                                style=(
                                    f"width:{width:.1f}%;"
                                ),
                            ),

                            class_="allmlb-driver-bar-track"
                        ),

                        ui.div(
                            (
                                f"{edge_name} "
                                f"({abs(effect) * 100:.1f} pts)"
                            ),
                            class_=(
                                "allmlb-driver-edge "
                                f"{edge_class}"
                            ),
                        ),

                        class_="allmlb-driver-effect"
                    ),

                    class_="allmlb-driver-row"
                )
            )


    else:

        driver_rows = [

            ui.div(
                (
                    "Feature-level explanation is unavailable "
                    "for this matchup, but the probabilities "
                    "above still come from the saved model."
                ),
                class_="allmlb-explanation-unavailable"
            )
        ]


    # ========================================================
    # SUMMARY TEXT
    # ========================================================

    if probability_gap > 0:

        summary_text = (
            f"The model gives {snub_player['Name']} "
            f"a {probability_gap * 100:.1f}-point higher "
            f"selection probability than "
            f"{selected_player['Name']}. "
            "The drivers below show which differences "
            "between their seasons most influence that gap."
        )

    elif probability_gap < 0:

        summary_text = (
            f"The model gives {selected_player['Name']} "
            f"{abs(probability_gap) * 100:.1f} points more "
            f"selection probability than "
            f"{snub_player['Name']}. "
            "The drivers below show which differences "
            "between their seasons most influence that gap."
        )

    else:

        summary_text = (
            "The model gives these players nearly identical "
            "selection probabilities. The drivers below show "
            "which features push the prediction in either "
            "direction."
        )


    return ui.div(

        # ====================================================
        # HEADER
        # ====================================================

        ui.div(

            ui.div(
                "MODEL EXPLANATION",
                class_="allmlb-explanation-kicker"
            ),

            ui.div(
                "WHY DOES THE MODEL SEE THESE PLAYERS DIFFERENTLY?",
                class_="allmlb-explanation-title"
            ),

            ui.div(
                explanation_subtitle,
                class_="allmlb-explanation-subtitle"
            ),

            class_="allmlb-explanation-header"
        ),

        # ====================================================
        # TWO PLAYER SUMMARY
        # ====================================================

        ui.div(

            ui.div(

                ui.div(
                    selected_label,
                    class_="allmlb-explain-player-label"
                ),

                ui.div(
                    selected_player[
                        "Name"
                    ],
                    class_="allmlb-explain-player-name"
                ),

                ui.div(
                    (
                        f"{selected_probability * 100:.1f}%"
                    ),
                    class_=(
                        "allmlb-explain-prob "
                        "selected-explain-prob"
                    )
                ),

                class_=(
                    "allmlb-explain-player-card "
                    "actual-explain-card"
                )
            ),

            ui.div(
                "VS",
                class_="allmlb-explain-vs"
            ),

            ui.div(

                ui.div(
                    snub_label,
                    class_="allmlb-explain-player-label"
                ),

                ui.div(
                    snub_player[
                        "Name"
                    ],
                    class_="allmlb-explain-player-name"
                ),

                ui.div(
                    (
                        f"{snub_probability * 100:.1f}%"
                    ),
                    class_=(
                        "allmlb-explain-prob "
                        "snub-explain-prob"
                    )
                ),

                class_=(
                    "allmlb-explain-player-card "
                    "snub-explain-card"
                )
            ),

            class_="allmlb-explain-player-grid"
        ),

        # ====================================================
        # SUMMARY
        # ====================================================

        ui.div(
            summary_text,
            class_="allmlb-explanation-summary"
        ),

        # ====================================================
        # MODEL FEATURE SET
        # ====================================================

        ui.div(
            "FEATURES USED BY THE MODEL",
            class_="allmlb-explanation-section-title"
        ),

        ui.div(
            (
                "These are the inputs passed into the saved "
                "All-MLB model when it calculates a player's "
                "selection probability."
            ),
            class_="allmlb-feature-description"
        ),

        ui.div(
            *feature_chips,
            class_="allmlb-feature-chip-grid"
        ),

        # ====================================================
        # LOCAL DRIVERS
        # ====================================================

        ui.div(
            "BIGGEST DRIVERS OF THIS PROBABILITY GAP",
            class_="allmlb-explanation-section-title"
        ),

        ui.div(

            ui.div(
                "FEATURE"
            ),

            ui.div(
                selected_player[
                    "Name"
                ]
            ),

            ui.div(
                snub_player[
                    "Name"
                ]
            ),

            ui.div(
                "MODEL EDGE"
            ),

            class_="allmlb-driver-header"
        ),

        ui.div(
            *driver_rows,
            class_="allmlb-driver-list"
        ),

        # ====================================================
        # INTERPRETATION NOTE
        # ====================================================

        ui.div(
            (
                "How to read this: the model is re-scored "
                "after swapping one feature at a time between "
                "the two players. A larger effect means that "
                "difference has a larger influence on the "
                "probability gap for this specific comparison. "
                "These are model sensitivity effects, not "
                "causal effects and not a statement that one "
                "stat is universally more important."
            ),
            class_="allmlb-explanation-note"
        ),

        class_="allmlb-model-explanation"
    )


def allmlb_player_card(
    row,
    card_type="selection",
    rank=None
):

    probability = float(
        row.get(
            "Model_Probability",
            0
        )
    )


    if card_type == "selection":

        badge_text = (
            str(
                row.get(
                    "All_MLB_Team",
                    "Selected"
                )
            )
            .upper()
        )

        card_class = "selected-card"

    else:

        badge_text = "SNUB CANDIDATE"
        card_class = "snub-card"


    if rank is not None:

        badge_text = (
            f"#{rank} {badge_text}"
        )


    stats = [
        ("PA", safe_int(row.get("PA"))),
        ("H", safe_int(row.get("H"))),
        ("HR", safe_int(row.get("HR"))),
        ("RBI", safe_int(row.get("RBI"))),
        ("AVG", safe_rate(row.get("AVG"))),
        ("OBP", safe_rate(row.get("OBP"))),
        ("SLG", safe_rate(row.get("SLG"))),
        ("OPS", safe_rate(row.get("OPS"))),
    ]


    return ui.div(

        ui.div(

            ui.div(

                ui.div(
                    badge_text,
                    class_="comparison-badge"
                ),

                ui.div(
                    row.get(
                        "Name",
                        "Unknown"
                    ),
                    class_="comparison-player-name"
                ),

                ui.div(
                    (
                        "Primary Position: "
                        f"{row.get('primary_position', 'N/A')}"
                    ),
                    class_="comparison-player-position"
                ),
            ),

            ui.div(

                ui.div(
                    "MODEL PROBABILITY",
                    class_="comparison-prob-label"
                ),

                ui.div(
                    f"{probability * 100:.1f}%",
                    class_="comparison-prob-value"
                ),
            ),

            class_="comparison-card-header"
        ),

        ui.div(
            *[
                ui.div(

                    ui.div(
                        label,
                        class_="comparison-stat-label"
                    ),

                    ui.div(
                        value,
                        class_="comparison-stat-value"
                    ),

                    class_="comparison-stat"
                )
                for label, value
                in stats
            ],
            class_="comparison-stat-grid"
        ),

        class_=(
            "allmlb-comparison-card "
            f"{card_class}"
        )
    )


def make_position_comparison(
    scored_df,
    position
):

    # ========================================================
    # ACTUAL ALL-MLB SELECTIONS
    # ========================================================

    actual = scored_df[
        (
            scored_df["All_MLB"] == 1
        )
        &
        (
            scored_df["selection_position"]
            == position
        )
    ].copy()


    actual = actual.sort_values(
        "Model_Probability",
        ascending=False,
    )


    # ========================================================
    # NON-SELECTED PLAYERS / SNUBS
    # ========================================================

    snubs = scored_df[
        (
            scored_df["All_MLB"] == 0
        )
        &
        (
            scored_df["position_group"]
            == position
        )
    ].copy()


    snubs = (
        snubs
        .sort_values(
            "Model_Probability",
            ascending=False,
        )
        .head(
            6
            if position == "OF"
            else 3
        )
    )


    # ========================================================
    # ACTUAL SELECTION CARDS
    # ========================================================

    actual_cards = [

        allmlb_player_card(
            row,
            "selection",
            rank,
        )

        for rank, (_, row)
        in enumerate(
            actual.iterrows(),
            start=1,
        )
    ]


    # ========================================================
    # SNUB CARDS
    # ========================================================

    snub_cards = [

        allmlb_player_card(
            row,
            "snub",
            rank,
        )

        for rank, (_, row)
        in enumerate(
            snubs.iterrows(),
            start=1,
        )
    ]


    if not actual_cards:

        actual_cards = [

            ui.div(
                "No actual selection found.",
                class_="empty-comparison-message"
            )
        ]


    if not snub_cards:

        snub_cards = [

            ui.div(
                "No snub candidates found.",
                class_="empty-comparison-message"
            )
        ]


    # ========================================================
    # COMPLETE POSITION PANEL
    # ========================================================

    return ui.div(

        # POSITION TITLE

        ui.div(

            ui.div(
                position,
                class_="selected-position-title"
            ),

            ui.div(
                "ACTUAL SELECTIONS VS MODEL SNUBS",
                class_="selected-position-subtitle"
            ),

            class_="selected-position-header"
        ),

        # ====================================================
        # ORIGINAL SIDE-BY-SIDE CARDS
        # ====================================================

        ui.div(

            # ACTUAL SELECTIONS

            ui.div(

                ui.div(
                    "ACTUAL ALL-MLB SELECTIONS",
                    class_="comparison-section-title"
                ),

                *actual_cards,
            ),

            # MODEL SNUBS

            ui.div(

                ui.div(
                    (
                        "TOP 6 SNUBS"
                        if position == "OF"
                        else "TOP 3 SNUBS"
                    ),
                    class_="comparison-section-title"
                ),

                *snub_cards,
            ),

            class_="selection-vs-snub-grid"
        ),

        class_="position-comparison-panel"
    )


def make_actual_roster(
    scored_df
):

    selected = scored_df[
        scored_df["All_MLB"]
        == 1
    ].copy()


    if selected.empty:

        return ui.div(
            "No selections found."
        )


    cards = []


    for position in ALLMLB_POSITION_ORDER:

        position_players = selected[
            selected["selection_position"]
            == position
        ].copy()


        if position_players.empty:
            continue


        position_players = (
            position_players
            .sort_values(
                "Model_Probability",
                ascending=False,
            )
        )


        rows = []


        for _, row in (
            position_players.iterrows()
        ):

            rows.append(

                ui.div(

                    ui.div(
                        row["Name"],
                        class_="roster-player-name"
                    ),

                    ui.div(
                        (
                            f"{row['All_MLB_Team']} • "
                            f"{row['Model_Probability'] * 100:.1f}%"
                        ),
                        class_="roster-player-prob"
                    ),

                    class_="actual-roster-player-row"
                )
            )


        cards.append(

            ui.div(

                ui.div(
                    position,
                    class_="actual-roster-position"
                ),

                *rows,

                class_="actual-roster-card"
            )
        )


    return ui.div(
        *cards,
        class_="actual-roster-grid"
    )


def build_projected_allmlb_roster(scored_df):

    if scored_df.empty:
        return pd.DataFrame()

    eligible = scored_df[
        scored_df["position_group"].isin(ALLMLB_POSITION_ORDER)
        &
        (pd.to_numeric(scored_df["PA"], errors="coerce") >= PROJECTED_ALLMLB_MIN_PA)
    ].copy()

    roster_rows = []

    for position in ALLMLB_POSITION_ORDER:

        pool = (
            eligible[eligible["position_group"] == position]
            .sort_values("Model_Probability", ascending=False)
            .copy()
        )

        first_count = 3 if position == "OF" else 1
        second_count = 3 if position == "OF" else 1

        first_team = pool.head(first_count).copy()
        second_team = pool.iloc[
            first_count:first_count + second_count
        ].copy()

        for rank, (_, row) in enumerate(first_team.iterrows(), start=1):
            item = row.copy()
            item["Projected_Team"] = "First Team"
            item["Projected_Rank"] = rank
            item["All_MLB_Team"] = "Projected First Team"
            roster_rows.append(item)

        for rank, (_, row) in enumerate(second_team.iterrows(), start=1):
            item = row.copy()
            item["Projected_Team"] = "Second Team"
            item["Projected_Rank"] = rank
            item["All_MLB_Team"] = "Projected Second Team"
            roster_rows.append(item)

    if not roster_rows:
        return pd.DataFrame()

    return pd.DataFrame(roster_rows).reset_index(drop=True)


def make_projected_allmlb_roster(scored_df):

    roster = build_projected_allmlb_roster(scored_df)

    if roster.empty:
        return ui.div(
            "No eligible 2027 projection roster is available.",
            class_="allmlb-explanation-unavailable"
        )

    cards = []

    for position in ALLMLB_POSITION_ORDER:

        position_players = roster[
            roster["position_group"] == position
        ].copy()

        if position_players.empty:
            continue

        rows = []

        for _, row in position_players.iterrows():

            team = str(row.get("Projected_Team", "Projected"))
            club = str(row.get("Team", "FA"))
            probability = float(row.get("Model_Probability", 0))

            rows.append(
                ui.div(
                    ui.div(
                        row["Name"],
                        class_="roster-player-name"
                    ),
                    ui.div(
                        f"{team} • {club} • {probability * 100:.1f}%",
                        class_="roster-player-prob"
                    ),
                    class_="actual-roster-player-row"
                )
            )

        cards.append(
            ui.div(
                ui.div(
                    position,
                    class_="actual-roster-position"
                ),
                *rows,
                class_="actual-roster-card"
            )
        )

    return ui.div(
        *cards,
        class_="actual-roster-grid"
    )



def _projection_team_rows(roster, position, team_name):

    rows = roster[
        (roster["position_group"] == position)
        &
        (roster["Projected_Team"] == team_name)
    ].copy()

    return rows.sort_values(
        ["Projected_Rank", "Model_Probability"],
        ascending=[True, False],
    )


def _projection_team_names(roster, position, team_name):

    rows = _projection_team_rows(
        roster,
        position,
        team_name,
    )

    if rows.empty:
        return ["—"]

    return [
        str(name)
        for name in rows["Name"].tolist()
    ]


def make_projection_position_button(
    roster,
    position,
    input_id,
    css_position_class,
    active=False,
):

    first_names = _projection_team_names(
        roster,
        position,
        "First Team",
    )

    second_names = _projection_team_names(
        roster,
        position,
        "Second Team",
    )

    first_players = ui.div(
        *[
            ui.div(
                name,
                class_="projection-diamond-player-name"
            )
            for name in first_names
        ],
        class_="projection-diamond-player-list"
    )

    second_players = ui.div(
        *[
            ui.div(
                name,
                class_="projection-diamond-player-name"
            )
            for name in second_names
        ],
        class_="projection-diamond-player-list"
    )

    label = ui.div(
        ui.div(
            position,
            class_="projection-diamond-position-label"
        ),
        ui.div(
            "FIRST TEAM",
            class_="projection-diamond-team-label projection-first-label"
        ),
        first_players,
        ui.div(
            "SECOND TEAM",
            class_="projection-diamond-team-label projection-second-label"
        ),
        second_players,
        class_="projection-diamond-button-content"
    )

    active_class = (
        " projection-position-card-active"
        if active
        else ""
    )

    return ui.input_action_button(
        input_id,
        label,
        class_=(
            "projection-position-card "
            f"{css_position_class}"
            f"{active_class}"
        ),
    )


def make_projected_allmlb_diamond(
    scored_df,
    ns,
    selected_position="SS",
):

    roster = build_projected_allmlb_roster(
        scored_df
    )

    if roster.empty:
        return ui.div(
            "No eligible 2027 projection roster is available.",
            class_="allmlb-explanation-unavailable"
        )

    return ui.div(

        ui.div(
            ui.div(
                "2027 PROJECTED ALL-MLB",
                class_="projection-diamond-kicker"
            ),
            ui.h3(
                "FIRST & SECOND TEAM DIAMOND",
                class_="diamond-title"
            ),
            ui.p(
                (
                    "Each position shows the projected First Team "
                    "player(s) followed immediately by the projected "
                    "Second Team player(s). Click any position card "
                    "to open that position's race and next-in-line "
                    "players below."
                ),
                class_="diamond-description"
            ),
            class_="diamond-heading"
        ),

        ui.div(

            ui.div(
                "OUTFIELD",
                class_=(
                    "field-zone-label "
                    "field-outfield-label"
                )
            ),

            make_projection_position_button(
                roster,
                "OF",
                ns("proj_pos_of"),
                "projection-card-of",
                selected_position == "OF",
            ),

            make_projection_position_button(
                roster,
                "SS",
                ns("proj_pos_ss"),
                "projection-card-ss",
                selected_position == "SS",
            ),

            make_projection_position_button(
                roster,
                "2B",
                ns("proj_pos_2b"),
                "projection-card-2b",
                selected_position == "2B",
            ),

            make_projection_position_button(
                roster,
                "3B",
                ns("proj_pos_3b"),
                "projection-card-3b",
                selected_position == "3B",
            ),

            make_projection_position_button(
                roster,
                "1B",
                ns("proj_pos_1b"),
                "projection-card-1b",
                selected_position == "1B",
            ),

            make_projection_position_button(
                roster,
                "C",
                ns("proj_pos_c"),
                "projection-card-c",
                selected_position == "C",
            ),

            make_projection_position_button(
                roster,
                "DH",
                ns("proj_pos_dh"),
                "projection-card-dh",
                selected_position == "DH",
            ),

            ui.div(
                class_=(
                    "base-marker "
                    "second-base-marker"
                )
            ),
            ui.div(
                class_=(
                    "base-marker "
                    "third-base-marker"
                )
            ),
            ui.div(
                class_=(
                    "base-marker "
                    "first-base-marker"
                )
            ),
            ui.div(
                class_="home-plate-marker"
            ),
            ui.div(
                class_="pitcher-mound"
            ),
            ui.div(
                "INFIELD",
                class_=(
                    "field-zone-label "
                    "field-infield-label"
                )
            ),

            class_=(
                "baseball-field "
                "projection-baseball-field"
            )
        ),

        ui.div(
            ui.span(
                "FIRST TEAM",
                class_="projection-diamond-legend-first"
            ),
            ui.span(
                "Top projected player(s) at the position"
            ),
            ui.span(
                "SECOND TEAM",
                class_="projection-diamond-legend-second"
            ),
            ui.span(
                "Next projected roster player(s) at the position"
            ),
            class_="projection-diamond-legend"
        ),

        class_="diamond-panel projection-diamond-panel"
    )


def make_projection_position_comparison(scored_df, position):

    roster = build_projected_allmlb_roster(scored_df)

    selected = roster[
        roster["position_group"] == position
    ].copy()

    pool = (
        scored_df[
            (scored_df["position_group"] == position)
            &
            (pd.to_numeric(scored_df["PA"], errors="coerce") >= PROJECTED_ALLMLB_MIN_PA)
        ]
        .sort_values("Model_Probability", ascending=False)
        .copy()
    )

    selected_names = set(selected["Name"].astype(str)) if not selected.empty else set()

    next_in_line = pool[
        ~pool["Name"].astype(str).isin(selected_names)
    ].head(6 if position == "OF" else 3).copy()

    if selected.empty:
        return ui.div(
            "No projected roster players are available at this position.",
            class_="allmlb-explanation-unavailable"
        )

    selected_cards = [
        allmlb_player_card(row, "selection", rank)
        for rank, (_, row) in enumerate(selected.iterrows(), start=1)
    ]

    next_cards = [
        allmlb_player_card(row, "snub", rank)
        for rank, (_, row) in enumerate(next_in_line.iterrows(), start=1)
    ]

    return ui.div(
        ui.div(
            ui.h3(
                f"2027 PROJECTED {position} RACE",
                class_="position-comparison-title"
            ),
            ui.p(
                (
                    "Projected First/Second Team players are chosen by "
                    "model probability within position. The right side "
                    "shows the strongest eligible players currently "
                    "outside the projected roster."
                ),
                class_="position-comparison-description"
            ),
        ),
        ui.layout_columns(
            ui.div(
                ui.h4("PROJECTED TEAM"),
                *selected_cards,
            ),
            ui.div(
                ui.h4("NEXT IN LINE"),
                *next_cards,
            ),
            col_widths=[6, 6],
        ),
        class_="position-comparison-panel"
    )


def make_custom_allmlb_scoreboard(
    player
):

    probability, prediction = (
        score_allmlb_player(
            player
        )
    )

    row = player.iloc[0]


    stats = [
        ("AB", safe_int(row.get("AB"))),
        ("PA", safe_int(row.get("PA"))),
        ("H", safe_int(row.get("H"))),
        ("HR", safe_int(row.get("HR"))),
        ("RBI", safe_int(row.get("RBI"))),
        ("BB", safe_int(row.get("BB"))),
        ("SB", safe_int(row.get("SB"))),
        ("AVG", safe_rate(row.get("AVG"))),
        ("OBP", safe_rate(row.get("OBP"))),
        ("SLG", safe_rate(row.get("SLG"))),
        ("OPS", safe_rate(row.get("OPS"))),
        ("ISO", safe_rate(row.get("ISO"))),
    ]


    return ui.div(

        ui.div(

            ui.div(
                "CUSTOM ALL-MLB SEASON",
                class_="marquee-small"
            ),

            ui.div(
                row.get(
                    "Name",
                    "Custom Player"
                ),
                class_="marquee-name"
            ),

            ui.div(
                (
                    "POSITION "
                    f"{row.get('primary_position', '')}"
                ),
                class_="marquee-subtitle"
            ),

            class_="custom-allmlb-header"
        ),

        ui.div(
            "SEASON STATISTICS",
            class_="score-section-title"
        ),

        ui.div(
            *[
                score_cell(
                    label,
                    value
                )
                for label, value
                in stats
            ],
            class_="score-grid"
        ),

        ui.div(
            "MODEL OUTPUT",
            class_="score-section-title"
        ),

        ui.div(

            ui.div(

                ui.div(
                    "ALL-MLB MODEL %",
                    class_="big-panel-label"
                ),

                ui.div(
                    f"{probability * 100:.1f}%",
                    class_="big-panel-number"
                ),

                class_="big-score-panel"
            ),

            ui.div(

                ui.div(
                    "DESIGNATION",
                    class_="big-panel-label"
                ),

                ui.div(
                    (
                        "ALL-MLB RANGE"
                        if prediction == 1
                        else
                        "BELOW THRESHOLD"
                    ),
                    class_=(
                        "designation-on"
                        if prediction == 1
                        else
                        "designation-off"
                    ),
                ),

                class_="big-score-panel"
            ),

            class_="big-score-grid"
        ),

        class_="scoreboard"
    )


@module.ui
def allmlb_ui():

    years = sorted(
        allmlb_all_seasons["yearID"]
        .dropna()
        .astype(int)
        .unique(),
        reverse=True,
    )

    choices = [
        str(year)
        for year in years
    ]

    default = (
        "2026"
        if "2026" in choices
        else choices[0]
    )

    return ui.div(

        # ====================================================
        # HEADER
        # ====================================================

        ui.div(

            ui.h1(
                "ALL-MLB TEAM EXPLORER",
                class_="dashboard-title"
            ),

            ui.p(
                (
                    "Interactive position explorer • "
                    "Actual selections • "
                    "Model probabilities • "
                    "Snub analysis"
                ),
                class_="dashboard-subtitle"
            ),

            class_="dashboard-header"
        ),

        # ====================================================
        # TABS
        # ====================================================

        ui.navset_tab(

            # =================================================
            # SEASON EXPLORER
            # =================================================

            ui.nav_panel(

                "Season Explorer",

                # ---------------------------------------------
                # SEASON / POSITION CONTROLS
                # ---------------------------------------------

                ui.div(

                    ui.layout_columns(

                        ui.div(

                            ui.h3(
                                "SELECT SEASON"
                            ),

                            ui.input_select(
                                "allmlb_season",
                                "MLB Season",
                                choices=choices,
                                selected=default,
                            ),

                            class_="season-control-box"
                        ),

                        ui.div(

                            ui.h3(
                                "CURRENT POSITION"
                            ),

                            ui.output_ui(
                                "selected_position_display"
                            ),

                            ui.p(
                                (
                                    "Click a position on the "
                                    "baseball field below."
                                ),
                                class_="allmlb-position-help"
                            ),

                            class_="season-control-box"
                        ),

                        col_widths=[
                            6,
                            6,
                        ],
                    ),

                    class_="selection-panel"
                ),

                # ---------------------------------------------
                # FIELD + ACTUAL ROSTER
                # ---------------------------------------------

                ui.layout_columns(

                    # =========================================
                    # LEFT SIDE
                    # =========================================

                    ui.div(

                        # =====================================
                        # BASEBALL DIAMOND
                        # =====================================

                        ui.div(

                            ui.div(

                                ui.h3(
                                    "POSITION EXPLORER",
                                    class_="diamond-title"
                                ),

                                ui.p(
                                    (
                                        "Click a position on the field "
                                        "to compare the actual All-MLB "
                                        "selections with the strongest "
                                        "players the model believed "
                                        "were snubs."
                                    ),
                                    class_="diamond-description"
                                ),

                                class_="diamond-heading"
                            ),

                            # =================================
                            # BASEBALL FIELD
                            # =================================

                            ui.div(

                                # OUTFIELD LABEL
                                ui.div(
                                    "OUTFIELD",
                                    class_=(
                                        "field-zone-label "
                                        "field-outfield-label"
                                    )
                                ),

                                # OUTFIELD
                                ui.input_action_button(
                                    "pos_of",
                                    "OF",
                                    class_=(
                                        "field-position-button "
                                        "field-of"
                                    ),
                                ),

                                # SHORTSTOP
                                ui.input_action_button(
                                    "pos_ss",
                                    "SS",
                                    class_=(
                                        "field-position-button "
                                        "field-ss"
                                    ),
                                ),

                                # SECOND BASE
                                ui.input_action_button(
                                    "pos_2b",
                                    "2B",
                                    class_=(
                                        "field-position-button "
                                        "field-2b"
                                    ),
                                ),

                                # THIRD BASE
                                ui.input_action_button(
                                    "pos_3b",
                                    "3B",
                                    class_=(
                                        "field-position-button "
                                        "field-3b"
                                    ),
                                ),

                                # FIRST BASE
                                ui.input_action_button(
                                    "pos_1b",
                                    "1B",
                                    class_=(
                                        "field-position-button "
                                        "field-1b"
                                    ),
                                ),

                                # CATCHER
                                ui.input_action_button(
                                    "pos_c",
                                    "C",
                                    class_=(
                                        "field-position-button "
                                        "field-c"
                                    ),
                                ),

                                # DESIGNATED HITTER
                                ui.input_action_button(
                                    "pos_dh",
                                    "DH",
                                    class_=(
                                        "field-position-button "
                                        "field-dh"
                                    ),
                                ),

                                # SECOND BASE MARKER
                                ui.div(
                                    class_=(
                                        "base-marker "
                                        "second-base-marker"
                                    )
                                ),

                                # THIRD BASE MARKER
                                ui.div(
                                    class_=(
                                        "base-marker "
                                        "third-base-marker"
                                    )
                                ),

                                # FIRST BASE MARKER
                                ui.div(
                                    class_=(
                                        "base-marker "
                                        "first-base-marker"
                                    )
                                ),

                                # HOME PLATE
                                ui.div(
                                    class_="home-plate-marker"
                                ),

                                # PITCHER'S MOUND
                                ui.div(
                                    class_="pitcher-mound"
                                ),

                                # INFIELD LABEL
                                ui.div(
                                    "INFIELD",
                                    class_=(
                                        "field-zone-label "
                                        "field-infield-label"
                                    )
                                ),

                                class_="baseball-field"
                            ),

                            # =================================
                            # LEGEND
                            # =================================

                            ui.div(

                                ui.div(
                                    ui.span(
                                        "C",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Catcher"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "1B",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "First Base"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "2B",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Second Base"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "3B",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Third Base"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "SS",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Shortstop"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "OF",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Outfield"
                                    ),
                                    class_="field-legend-item"
                                ),

                                ui.div(
                                    ui.span(
                                        "DH",
                                        class_="field-legend-code"
                                    ),
                                    ui.span(
                                        "Designated Hitter"
                                    ),
                                    class_="field-legend-item"
                                ),

                                class_="field-legend"
                            ),

                            class_="diamond-panel"
                        ),

                        # =====================================
                        # POSITION COMPARISON
                        # =====================================

                        ui.output_ui(
                            "position_comparison"
                        ),

                        # =====================================
                        # PLAYER-SELECTED MODEL COMPARISON
                        # =====================================

                        ui.output_ui(
                            "allmlb_comparison_controls"
                        ),

                        ui.output_ui(
                            "allmlb_selected_model_explanation"
                        ),

                        class_="diamond-explorer-column"
                    ),

                    # =========================================
                    # RIGHT SIDE — ACTUAL ROSTER
                    # =========================================

                    ui.div(

                        ui.div(

                            ui.h3(
                                "ACTUAL ALL-MLB SELECTIONS",
                                class_="actual-roster-title"
                            ),

                            ui.p(
                                (
                                    "Official All-MLB selections "
                                    "for the chosen season with "
                                    "their model probability."
                                ),
                                class_="actual-roster-description"
                            ),

                            ui.output_ui(
                                "allmlb_actual_roster"
                            ),

                            class_="actual-roster-panel"
                        ),

                        class_="allmlb-roster-column"
                    ),

                    col_widths=[
                        8,
                        4,
                    ],
                ),
            ),

            # =================================================
            # 2027 PROJECTION
            # =================================================

            ui.nav_panel(

                "2027 Projection",

                ui.div(
                    ui.h3(
                        "2027 STEAMER PROJECTED ALL-MLB TEAM"
                    ),
                    ui.p(
                        (
                            "Steamer projected statistics are run through "
                            "the saved All-MLB model. The roster is built "
                            "by ranking eligible players within position: "
                            "one First Team and one Second Team player at "
                            "C, 1B, 2B, 3B, SS, and DH, plus three "
                            "outfielders on each team."
                        )
                    ),
                    ui.p(
                        (
                            f"Roster eligibility currently requires at least "
                            f"{PROJECTED_ALLMLB_MIN_PA} projected plate appearances. "
                            "Probabilities are generated by this All-MLB model, "
                            "not by Steamer itself."
                        ),
                        class_="allmlb-position-help"
                    ),
                    class_="selection-panel"
                ),

                ui.output_ui(
                    "allmlb_2027_projection_diamond"
                ),

                ui.output_ui(
                    "allmlb_2027_selected_position_header"
                ),

                ui.output_ui(
                    "allmlb_2027_position_comparison"
                ),

                ui.output_ui(
                    "allmlb_2027_comparison_controls"
                ),

                ui.output_ui(
                    "allmlb_2027_model_explanation"
                ),
            ),

            # =================================================
            # CUSTOM SEASON
            # =================================================

            ui.nav_panel(

                "Custom Season",

                ui.div(

                    ui.h3(
                        "BUILD A HYPOTHETICAL MLB SEASON"
                    ),

                    ui.p(
                        (
                            "Enter a player's offensive season "
                            "and run it through the All-MLB model."
                        )
                    ),

                    class_="selection-panel"
                ),

                ui.div(

                    ui.layout_columns(

                        ui.input_text(
                            "allmlb_custom_name",
                            "Player Name",
                            value="Custom Player",
                        ),

                        ui.input_select(
                            "allmlb_custom_position",
                            "Primary Position",
                            choices=[
                                "C",
                                "1B",
                                "2B",
                                "3B",
                                "SS",
                                "LF",
                                "CF",
                                "RF",
                                "DH",
                            ],
                            selected="RF",
                        ),

                        ui.input_numeric(
                            "allmlb_custom_ab",
                            "At-Bats",
                            value=575,
                            min=1,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "allmlb_custom_h",
                            "Hits",
                            value=160,
                            min=0,
                        ),

                        ui.input_numeric(
                            "allmlb_custom_hr",
                            "Home Runs",
                            value=30,
                            min=0,
                        ),

                        ui.input_numeric(
                            "allmlb_custom_rbi",
                            "RBI",
                            value=90,
                            min=0,
                        ),

                        ui.input_numeric(
                            "allmlb_custom_bb",
                            "Walks",
                            value=70,
                            min=0,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "allmlb_custom_sb",
                            "Stolen Bases",
                            value=10,
                            min=0,
                        ),

                        ui.input_numeric(
                            "allmlb_custom_so",
                            "Strikeouts",
                            value=130,
                            min=0,
                        ),

                        ui.input_numeric(
                            "allmlb_custom_slg",
                            "SLG",
                            value=.500,
                            min=0,
                            max=1.5,
                            step=.001,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.input_action_button(
                        "predict_allmlb_custom",
                        "RUN ALL-MLB MODEL",
                        class_="predict-button",
                    ),

                    class_="custom-panel"
                ),

                ui.output_ui(
                    "allmlb_custom_scoreboard"
                ),
            ),
        ),
    )

@module.server
def allmlb_server(
    input,
    output,
    session
):

    selected_position = reactive.Value(
        "C"
    )

    selected_projection_position = reactive.Value(
        "SS"
    )


    @reactive.effect
    @reactive.event(input.pos_c)
    def _():
        selected_position.set("C")


    @reactive.effect
    @reactive.event(input.pos_1b)
    def _():
        selected_position.set("1B")


    @reactive.effect
    @reactive.event(input.pos_2b)
    def _():
        selected_position.set("2B")


    @reactive.effect
    @reactive.event(input.pos_3b)
    def _():
        selected_position.set("3B")


    @reactive.effect
    @reactive.event(input.pos_ss)
    def _():
        selected_position.set("SS")


    @reactive.effect
    @reactive.event(input.pos_of)
    def _():
        selected_position.set("OF")


    @reactive.effect
    @reactive.event(input.pos_dh)
    def _():
        selected_position.set("DH")


    @reactive.calc
    def scored_season():

        selected_year = int(
            input.allmlb_season()
        )

        season = allmlb_all_seasons[
            allmlb_all_seasons["yearID"]
            == selected_year
        ].copy()

        return score_allmlb_dataframe(
            season
        )


    @output
    @render.ui
    def selected_position_display():

        return ui.div(
            selected_position.get(),
            class_="selected-position-chip"
        )


    @output
    @render.ui
    def position_comparison():

        return make_position_comparison(
            scored_season(),
            selected_position.get(),
        )


    # ========================================================
    # PLAYER-SELECTED COMPARISON CONTROLS
    # ========================================================

    @output
    @render.ui
    def allmlb_comparison_controls():

        position = selected_position.get()
        season_df = scored_season()

        # ----------------------------------------------------
        # ACTUAL ALL-MLB SELECTIONS AT THIS POSITION
        # ----------------------------------------------------

        actual_players = season_df[
            (season_df["All_MLB"] == 1)
            &
            (season_df["selection_position"] == position)
        ].copy()

        # ----------------------------------------------------
        # TOP MODEL SNUBS AT THIS POSITION
        # ----------------------------------------------------

        snub_players = season_df[
            (season_df["All_MLB"] == 0)
            &
            (season_df["position_group"] == position)
        ].copy()

        snub_players = (
            snub_players
            .sort_values(
                "Model_Probability",
                ascending=False,
            )
            .head(
                6
                if position == "OF"
                else 3
            )
        )

        if (
            actual_players.empty
            or snub_players.empty
        ):

            return ui.div(
                (
                    "Player comparison options are unavailable "
                    "for this position and season."
                ),
                class_="allmlb-explanation-unavailable"
            )

        # Sort selections so the strongest model score appears first.
        # The All-MLB team label remains visible in each dropdown option.
        actual_players = (
            actual_players
            .sort_values(
                "Model_Probability",
                ascending=False,
            )
        )

        selected_choices = {}

        for _, row in actual_players.iterrows():

            player_name = str(
                row["Name"]
            )

            team_name = str(
                row.get(
                    "All_MLB_Team",
                    "Selected"
                )
            )

            probability = float(
                row["Model_Probability"]
            )

            selected_choices[player_name] = (
                f"{player_name} — "
                f"{team_name} — "
                f"{probability * 100:.1f}%"
            )

        snub_choices = {}

        for rank, (_, row) in enumerate(
            snub_players.iterrows(),
            start=1,
        ):

            player_name = str(
                row["Name"]
            )

            probability = float(
                row["Model_Probability"]
            )

            snub_choices[player_name] = (
                f"#{rank} {player_name} — "
                f"{probability * 100:.1f}%"
            )

        # Default to the lowest-probability actual selection
        # against the highest-probability snub. The user can
        # immediately choose any other pair from the dropdowns.
        default_selected = str(
            actual_players
            .sort_values(
                "Model_Probability",
                ascending=True,
            )
            .iloc[0]["Name"]
        )

        default_snub = str(
            snub_players
            .sort_values(
                "Model_Probability",
                ascending=False,
            )
            .iloc[0]["Name"]
        )

        snub_count = (
            6
            if position == "OF"
            else 3
        )

        return ui.div(

            ui.div(

                ui.div(
                    f"CHOOSE A {position} COMPARISON",
                    class_="allmlb-comparison-title"
                ),

                ui.div(
                    (
                        "Choose the exact All-MLB selection and "
                        "model snub you want to compare. "
                        f"The snub menu contains the top {snub_count} "
                        "non-selected players by model probability "
                        "at this position."
                    ),
                    class_="allmlb-comparison-description"
                ),
            ),

            ui.layout_columns(

                ui.input_select(
                    session.ns(
                        "comparison_selected_player"
                    ),
                    "Selected All-MLB Player",
                    choices=selected_choices,
                    selected=default_selected,
                ),

                ui.input_select(
                    session.ns(
                        "comparison_snub_player"
                    ),
                    "Model Snub",
                    choices=snub_choices,
                    selected=default_snub,
                ),

                col_widths=[
                    6,
                    6,
                ],
            ),

            class_="allmlb-comparison-controls"
        )


    # ========================================================
    # PLAYER-SELECTED MODEL EXPLANATION
    # ========================================================

    @output
    @render.ui
    def allmlb_selected_model_explanation():

        position = selected_position.get()
        season_df = scored_season()

        actual_players = season_df[
            (season_df["All_MLB"] == 1)
            &
            (season_df["selection_position"] == position)
        ].copy()

        snub_players = season_df[
            (season_df["All_MLB"] == 0)
            &
            (season_df["position_group"] == position)
        ].copy()

        snub_players = (
            snub_players
            .sort_values(
                "Model_Probability",
                ascending=False,
            )
            .head(
                6
                if position == "OF"
                else 3
            )
        )

        if (
            actual_players.empty
            or snub_players.empty
        ):

            return None

        # Dynamic UI inputs can briefly be unavailable when a
        # new position is clicked, so safe fallbacks are used.
        try:
            selected_name = input.comparison_selected_player()
        except Exception:
            selected_name = None

        try:
            snub_name = input.comparison_snub_player()
        except Exception:
            snub_name = None

        actual_names = (
            actual_players["Name"]
            .astype(str)
            .values
        )

        snub_names = (
            snub_players["Name"]
            .astype(str)
            .values
        )

        if (
            not selected_name
            or str(selected_name) not in actual_names
        ):

            selected_name = str(
                actual_players
                .sort_values(
                    "Model_Probability",
                    ascending=True,
                )
                .iloc[0]["Name"]
            )

        if (
            not snub_name
            or str(snub_name) not in snub_names
        ):

            snub_name = str(
                snub_players
                .sort_values(
                    "Model_Probability",
                    ascending=False,
                )
                .iloc[0]["Name"]
            )

        selected_player_df = (
            actual_players[
                actual_players["Name"]
                .astype(str)
                == str(selected_name)
            ]
            .iloc[[0]]
            .copy()
        )

        snub_player_df = (
            snub_players[
                snub_players["Name"]
                .astype(str)
                == str(snub_name)
            ]
            .iloc[[0]]
            .copy()
        )

        return make_allmlb_model_explanation(
            selected_player_df,
            snub_player_df,
            explanation_subtitle=(
                "You selected the exact All-MLB player and "
                "model snub for this comparison."
            ),
            snub_label="SELECTED MODEL SNUB",
        )


    @output
    @render.ui
    def allmlb_actual_roster():

        return make_actual_roster(
            scored_season()
        )


    # ========================================================
    # 2027 STEAMER PROJECTION
    # ========================================================

    @reactive.calc
    def scored_2027_projection():

        if allmlb_2027_projection.empty:
            return pd.DataFrame()

        projection = allmlb_2027_projection.copy()

        # Rows without a usable All-MLB position are kept out of
        # scoring/roster construction rather than guessed.
        projection = projection[
            projection["position_group"].isin(ALLMLB_POSITION_ORDER)
        ].copy()

        return score_allmlb_dataframe(projection)


    @reactive.effect
    @reactive.event(input.proj_pos_c)
    def _():
        selected_projection_position.set("C")


    @reactive.effect
    @reactive.event(input.proj_pos_1b)
    def _():
        selected_projection_position.set("1B")


    @reactive.effect
    @reactive.event(input.proj_pos_2b)
    def _():
        selected_projection_position.set("2B")


    @reactive.effect
    @reactive.event(input.proj_pos_3b)
    def _():
        selected_projection_position.set("3B")


    @reactive.effect
    @reactive.event(input.proj_pos_ss)
    def _():
        selected_projection_position.set("SS")


    @reactive.effect
    @reactive.event(input.proj_pos_of)
    def _():
        selected_projection_position.set("OF")


    @reactive.effect
    @reactive.event(input.proj_pos_dh)
    def _():
        selected_projection_position.set("DH")


    @output
    @render.ui
    def allmlb_2027_projection_diamond():

        if allmlb_2027_projection.empty:
            return ui.div(
                (
                    "2027_steamer_merged.csv was not found next to the "
                    "Shiny app. Add that file to enable this projection."
                ),
                class_="allmlb-explanation-unavailable"
            )

        return make_projected_allmlb_diamond(
            scored_2027_projection(),
            session.ns,
            selected_projection_position.get(),
        )


    @output
    @render.ui
    def allmlb_2027_selected_position_header():

        if allmlb_2027_projection.empty:
            return None

        position = selected_projection_position.get()

        return ui.div(
            ui.div(
                "SELECTED POSITION",
                class_="projection-selected-kicker"
            ),
            ui.div(
                position,
                class_="selected-position-chip"
            ),
            ui.div(
                (
                    "The section below shows the projected First/Second "
                    "Team player(s) and the strongest eligible players "
                    "currently next in line."
                ),
                class_="projection-selected-description"
            ),
            class_="projection-selected-position-header"
        )


    @output
    @render.ui
    def allmlb_2027_position_comparison():

        if allmlb_2027_projection.empty:
            return None

        return make_projection_position_comparison(
            scored_2027_projection(),
            selected_projection_position.get(),
        )


    @output
    @render.ui
    def allmlb_2027_comparison_controls():

        scored = scored_2027_projection()

        if scored.empty:
            return None

        position = selected_projection_position.get()
        roster = build_projected_allmlb_roster(scored)

        projected_players = roster[
            roster["position_group"] == position
        ].copy()

        pool = (
            scored[
                (scored["position_group"] == position)
                &
                (pd.to_numeric(scored["PA"], errors="coerce") >= PROJECTED_ALLMLB_MIN_PA)
            ]
            .sort_values("Model_Probability", ascending=False)
            .copy()
        )

        projected_names = set(
            projected_players["Name"].astype(str)
        ) if not projected_players.empty else set()

        next_players = pool[
            ~pool["Name"].astype(str).isin(projected_names)
        ].head(6 if position == "OF" else 3).copy()

        if projected_players.empty or next_players.empty:
            return ui.div(
                "Projection comparison options are unavailable at this position.",
                class_="allmlb-explanation-unavailable"
            )

        projected_choices = {}

        for _, row in projected_players.iterrows():
            name = str(row["Name"])
            projected_choices[name] = (
                f"{name} — {row.get('Projected_Team', 'Projected')} — "
                f"{float(row['Model_Probability']) * 100:.1f}%"
            )

        next_choices = {}

        for rank, (_, row) in enumerate(next_players.iterrows(), start=1):
            name = str(row["Name"])
            next_choices[name] = (
                f"#{rank} {name} — "
                f"{float(row['Model_Probability']) * 100:.1f}%"
            )

        default_projected = str(
            projected_players
            .sort_values("Model_Probability", ascending=True)
            .iloc[0]["Name"]
        )

        default_next = str(
            next_players
            .sort_values("Model_Probability", ascending=False)
            .iloc[0]["Name"]
        )

        return ui.div(
            ui.div(
                ui.div(
                    f"CHOOSE A 2027 {position} COMPARISON",
                    class_="allmlb-comparison-title"
                ),
                ui.div(
                    (
                        "Compare any projected roster player with a "
                        "next-in-line player at the same position."
                    ),
                    class_="allmlb-comparison-description"
                ),
            ),
            ui.layout_columns(
                ui.input_select(
                    session.ns("projection_selected_player"),
                    "Projected All-MLB Player",
                    choices=projected_choices,
                    selected=default_projected,
                ),
                ui.input_select(
                    session.ns("projection_next_player"),
                    "Next In Line",
                    choices=next_choices,
                    selected=default_next,
                ),
                col_widths=[6, 6],
            ),
            class_="allmlb-comparison-controls"
        )


    @output
    @render.ui
    def allmlb_2027_model_explanation():

        scored = scored_2027_projection()

        if scored.empty:
            return None

        position = selected_projection_position.get()
        roster = build_projected_allmlb_roster(scored)

        projected_players = roster[
            roster["position_group"] == position
        ].copy()

        pool = (
            scored[
                (scored["position_group"] == position)
                &
                (pd.to_numeric(scored["PA"], errors="coerce") >= PROJECTED_ALLMLB_MIN_PA)
            ]
            .sort_values("Model_Probability", ascending=False)
            .copy()
        )

        projected_names = set(
            projected_players["Name"].astype(str)
        ) if not projected_players.empty else set()

        next_players = pool[
            ~pool["Name"].astype(str).isin(projected_names)
        ].head(6 if position == "OF" else 3).copy()

        if projected_players.empty or next_players.empty:
            return None

        try:
            selected_name = input.projection_selected_player()
        except Exception:
            selected_name = None

        try:
            next_name = input.projection_next_player()
        except Exception:
            next_name = None

        projected_name_values = projected_players["Name"].astype(str).values
        next_name_values = next_players["Name"].astype(str).values

        if not selected_name or str(selected_name) not in projected_name_values:
            selected_name = str(
                projected_players
                .sort_values("Model_Probability", ascending=True)
                .iloc[0]["Name"]
            )

        if not next_name or str(next_name) not in next_name_values:
            next_name = str(
                next_players
                .sort_values("Model_Probability", ascending=False)
                .iloc[0]["Name"]
            )

        selected_df = projected_players[
            projected_players["Name"].astype(str) == str(selected_name)
        ].iloc[[0]].copy()

        next_df = next_players[
            next_players["Name"].astype(str) == str(next_name)
        ].iloc[[0]].copy()

        return make_allmlb_model_explanation(
            selected_df,
            next_df,
            explanation_subtitle=(
                "This compares a projected 2027 All-MLB roster player "
                "with a next-in-line player using their Steamer projected "
                "statistics."
            ),
            snub_label="NEXT IN LINE",
            selected_label="PROJECTED SELECTION",
        )


    @reactive.calc
    @reactive.event(
        input.predict_allmlb_custom
    )
    def custom_allmlb_player():

        AB = max(
            float(
                input.allmlb_custom_ab()
                or 1
            ),
            1,
        )

        H = float(
            input.allmlb_custom_h()
            or 0
        )

        HR = float(
            input.allmlb_custom_hr()
            or 0
        )

        RBI = float(
            input.allmlb_custom_rbi()
            or 0
        )

        BB = float(
            input.allmlb_custom_bb()
            or 0
        )

        SB = float(
            input.allmlb_custom_sb()
            or 0
        )

        SO = float(
            input.allmlb_custom_so()
            or 0
        )

        SLG = float(
            input.allmlb_custom_slg()
            or 0
        )


        PA = AB + BB
        AVG = H / AB

        OBP = (
            (H + BB) / PA
            if PA > 0
            else 0
        )

        OPS = OBP + SLG
        ISO = SLG - AVG


        return pd.DataFrame(
            {
                "Name": [
                    input.allmlb_custom_name()
                    or "Custom Player"
                ],

                "primary_position": [
                    input.allmlb_custom_position()
                ],

                "AB": [AB],
                "PA": [PA],
                "H": [H],
                "HR": [HR],
                "RBI": [RBI],
                "BB": [BB],
                "SB": [SB],

                "AVG": [AVG],
                "OBP": [OBP],
                "SLG": [SLG],
                "OPS": [OPS],
                "ISO": [ISO],

                "HR_rate": [
                    HR / PA * 100
                    if PA > 0
                    else 0
                ],

                "BB_rate": [
                    BB / PA * 100
                    if PA > 0
                    else 0
                ],

                "SO_rate": [
                    SO / PA * 100
                    if PA > 0
                    else 0
                ],
            }
        )


    @output
    @render.ui
    def allmlb_custom_scoreboard():

        return make_custom_allmlb_scoreboard(
            custom_allmlb_player()
        )


# ============================================================
# ============================================================
# POSTSEASON MODULE
# ============================================================
# ============================================================

def postseason_cumulative_probabilities(
    dataframe
):

    if dataframe.empty:
        return np.empty((0, 5))


    X = dataframe[
        postseason_features
    ]


    probabilities = []


    for threshold in range(
        1,
        6
    ):

        probabilities.append(
            postseason_threshold_models[
                threshold
            ]
            .predict_proba(
                X
            )[:, 1]
        )


    cumulative = np.column_stack(
        probabilities
    )


    cumulative = np.minimum.accumulate(
        cumulative,
        axis=1,
    )


    return cumulative


def postseason_exact_probabilities(
    dataframe
):

    cumulative = (
        postseason_cumulative_probabilities(
            dataframe
        )
    )


    if len(cumulative) == 0:

        return np.empty(
            (0, 6)
        )


    exact = np.zeros(
        (
            len(dataframe),
            6,
        )
    )


    exact[:, 0] = (
        1 - cumulative[:, 0]
    )

    exact[:, 1] = (
        cumulative[:, 0]
        - cumulative[:, 1]
    )

    exact[:, 2] = (
        cumulative[:, 1]
        - cumulative[:, 2]
    )

    exact[:, 3] = (
        cumulative[:, 2]
        - cumulative[:, 3]
    )

    exact[:, 4] = (
        cumulative[:, 3]
        - cumulative[:, 4]
    )

    exact[:, 5] = (
        cumulative[:, 4]
    )


    exact = np.clip(
        exact,
        0,
        1,
    )


    row_sums = exact.sum(
        axis=1,
        keepdims=True,
    )


    row_sums[
        row_sums == 0
    ] = 1


    return exact / row_sums


def score_postseason_dataframe(
    dataframe
):

    if dataframe.empty:
        return dataframe.copy()


    scored = dataframe.copy()


    cumulative = (
        postseason_cumulative_probabilities(
            scored
        )
    )


    exact = (
        postseason_exact_probabilities(
            scored
        )
    )


    scored["prob_make_playoffs"] = cumulative[:, 0]
    scored["prob_reach_ds"] = cumulative[:, 1]
    scored["prob_reach_cs"] = cumulative[:, 2]
    scored["prob_reach_ws"] = cumulative[:, 3]
    scored["prob_win_ws"] = cumulative[:, 4]

    scored["prob_miss_playoffs"] = exact[:, 0]
    scored["prob_wc_exit"] = exact[:, 1]
    scored["prob_ds_exit"] = exact[:, 2]
    scored["prob_cs_exit"] = exact[:, 3]
    scored["prob_ws_loss"] = exact[:, 4]
    scored["prob_ws_champion"] = exact[:, 5]


    return scored


# ============================================================
# POSTSEASON FORMAT
# ============================================================

def postseason_era(year):

    year = int(year)

    if year <= 2011:
        return "1995-2011"

    if year == 2012:
        return "2012"

    if year <= 2019:
        return "2013-2019"

    if year == 2020:
        return "2020"

    if year == 2021:
        return "2021"

    return "2022+"


def postseason_format_label(year):

    return {
        "1995-2011":
            "8-Team Postseason",

        "2012":
            "10-Team Postseason",

        "2013-2019":
            "10-Team Postseason",

        "2020":
            "16-Team Postseason",

        "2021":
            "10-Team Postseason",

        "2022+":
            "12-Team Postseason",

    }[
        postseason_era(
            year
        )
    ]


# ============================================================
# SEEDING
# ============================================================

def add_record_tiebreakers(
    dataframe
):

    dataframe = dataframe.copy()


    if "win_pct" not in dataframe.columns:

        dataframe["win_pct"] = (
            dataframe["W"]
            /
            dataframe["G"]
        )


    if "run_diff_per_game" not in dataframe.columns:

        dataframe["run_diff_per_game"] = (
            (
                dataframe["R"]
                -
                dataframe["RA"]
            )
            /
            dataframe["G"]
        )


    return dataframe


def record_sort(dataframe):

    dataframe = add_record_tiebreakers(
        dataframe
    )


    return dataframe.sort_values(
        [
            "win_pct",
            "run_diff_per_game",
            "W",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )


def determine_division_winners(
    league_df
):

    winners = []


    for division in [
        "E",
        "C",
        "W",
    ]:

        group = league_df[
            league_df["divID"]
            == division
        ].copy()


        if group.empty:
            continue


        winners.append(
            record_sort(
                group
            ).head(1)
        )


    if not winners:

        return league_df.iloc[
            0:0
        ].copy()


    return pd.concat(
        winners
    )


def seed_league(
    scored_df,
    league,
    year
):

    league_df = scored_df[
        scored_df["league"]
        == league
    ].copy()


    league_df = add_record_tiebreakers(
        league_df
    )


    league_df["seed"] = np.nan
    league_df["seed_type"] = "Out"


    era = postseason_era(
        year
    )


    if era == "2020":

        qualifiers = []
        used_indices = set()


        for division in [
            "E",
            "C",
            "W",
        ]:

            top_two = record_sort(
                league_df[
                    league_df["divID"]
                    == division
                ]
            ).head(2)


            qualifiers.append(
                top_two
            )


            used_indices.update(
                top_two.index
            )


        first_six = pd.concat(
            qualifiers
        )


        remaining = record_sort(
            league_df[
                ~league_df.index.isin(
                    used_indices
                )
            ]
        ).head(2)


        playoff_field = pd.concat(
            [
                first_six,
                remaining,
            ]
        )


        playoff_field = record_sort(
            playoff_field
        )


        for seed, index in enumerate(
            playoff_field.index,
            start=1,
        ):

            league_df.loc[
                index,
                "seed"
            ] = seed

            league_df.loc[
                index,
                "seed_type"
            ] = "Playoff"


        return league_df


    division_winners = (
        determine_division_winners(
            league_df
        )
    )


    division_winners = record_sort(
        division_winners
    )


    division_indices = set(
        division_winners.index
    )


    for seed, index in enumerate(
        division_winners.index,
        start=1,
    ):

        league_df.loc[
            index,
            "seed"
        ] = seed

        league_df.loc[
            index,
            "seed_type"
        ] = "Division Winner"


    remaining = record_sort(
        league_df[
            ~league_df.index.isin(
                division_indices
            )
        ]
    )


    if era == "1995-2011":

        wildcard_count = 1

    elif era in [
        "2012",
        "2013-2019",
        "2021",
    ]:

        wildcard_count = 2

    else:

        wildcard_count = 3


    wildcards = remaining.head(
        wildcard_count
    )


    for seed, index in enumerate(
        wildcards.index,
        start=4,
    ):

        league_df.loc[
            index,
            "seed"
        ] = seed

        league_df.loc[
            index,
            "seed_type"
        ] = "Wild Card"


    return league_df


def seed_row(
    seeded_df,
    seed
):

    result = seeded_df[
        seeded_df["seed"]
        == seed
    ]


    if result.empty:
        return None


    return result.iloc[0]


# ============================================================
# WINNER LOGIC
# ============================================================

def model_series_winner(
    team_a,
    team_b,
    probability_column
):

    if team_a is None:
        return team_b

    if team_b is None:
        return team_a


    probability_a = float(
        team_a.get(
            probability_column,
            0
        )
    )


    probability_b = float(
        team_b.get(
            probability_column,
            0
        )
    )


    if probability_a > probability_b:
        return team_a

    if probability_b > probability_a:
        return team_b


    win_pct_a = float(
        team_a.get(
            "win_pct",
            0
        )
    )


    win_pct_b = float(
        team_b.get(
            "win_pct",
            0
        )
    )


    if win_pct_a >= win_pct_b:
        return team_a


    return team_b


def actual_stage_value(team):

    if team is None:
        return -1


    value = team.get(
        "postseason_outcome",
        np.nan
    )


    if pd.isna(value):
        return -1


    return int(value)


def actual_series_winner(
    team_a,
    team_b
):

    if team_a is None:
        return team_b

    if team_b is None:
        return team_a


    stage_a = actual_stage_value(
        team_a
    )

    stage_b = actual_stage_value(
        team_b
    )


    if stage_a > stage_b:
        return team_a

    if stage_b > stage_a:
        return team_b


    return model_series_winner(
        team_a,
        team_b,
        "prob_win_ws",
    )


def make_matchup(
    team_a,
    team_b,
    winner,
    round_name,
    probability_column
):

    return {
        "team_a": team_a,
        "team_b": team_b,
        "winner": winner,
        "round": round_name,
        "probability_column":
            probability_column,
    }


# ============================================================
# MODERN 2022+ BRACKET
# ============================================================

def simulate_modern_league(
    seeded_df,
    actual=False
):

    s1 = seed_row(seeded_df, 1)
    s2 = seed_row(seeded_df, 2)
    s3 = seed_row(seeded_df, 3)
    s4 = seed_row(seeded_df, 4)
    s5 = seed_row(seeded_df, 5)
    s6 = seed_row(seeded_df, 6)


    if actual:

        wc36_winner = (
            actual_series_winner(
                s3,
                s6,
            )
        )

        wc45_winner = (
            actual_series_winner(
                s4,
                s5,
            )
        )

    else:

        wc36_winner = (
            model_series_winner(
                s3,
                s6,
                "prob_reach_ds",
            )
        )

        wc45_winner = (
            model_series_winner(
                s4,
                s5,
                "prob_reach_ds",
            )
        )


    wc36 = make_matchup(
        s3,
        s6,
        wc36_winner,
        "Wild Card",
        "prob_reach_ds",
    )


    wc45 = make_matchup(
        s4,
        s5,
        wc45_winner,
        "Wild Card",
        "prob_reach_ds",
    )


    if actual:

        ds1_winner = (
            actual_series_winner(
                s1,
                wc45_winner,
            )
        )

        ds2_winner = (
            actual_series_winner(
                s2,
                wc36_winner,
            )
        )

    else:

        ds1_winner = (
            model_series_winner(
                s1,
                wc45_winner,
                "prob_reach_cs",
            )
        )

        ds2_winner = (
            model_series_winner(
                s2,
                wc36_winner,
                "prob_reach_cs",
            )
        )


    ds1 = make_matchup(
        s1,
        wc45_winner,
        ds1_winner,
        "Division Series",
        "prob_reach_cs",
    )


    ds2 = make_matchup(
        s2,
        wc36_winner,
        ds2_winner,
        "Division Series",
        "prob_reach_cs",
    )


    if actual:

        cs_winner = (
            actual_series_winner(
                ds1_winner,
                ds2_winner,
            )
        )

    else:

        cs_winner = (
            model_series_winner(
                ds1_winner,
                ds2_winner,
                "prob_reach_ws",
            )
        )


    cs = make_matchup(
        ds1_winner,
        ds2_winner,
        cs_winner,
        "Championship Series",
        "prob_reach_ws",
    )


    return {
        "seeded": seeded_df,

        "wild_card": [
            wc36,
            wc45,
        ],

        "division_series": [
            ds1,
            ds2,
        ],

        "championship": cs,
        "champion": cs_winner,
    }


# ============================================================
# 10 TEAM FORMAT
# ============================================================

def simulate_five_team_league(
    seeded_df,
    actual=False
):

    s1 = seed_row(seeded_df, 1)
    s2 = seed_row(seeded_df, 2)
    s3 = seed_row(seeded_df, 3)
    s4 = seed_row(seeded_df, 4)
    s5 = seed_row(seeded_df, 5)


    if actual:

        wc_winner = (
            actual_series_winner(
                s4,
                s5,
            )
        )

    else:

        wc_winner = (
            model_series_winner(
                s4,
                s5,
                "prob_reach_ds",
            )
        )


    wc = make_matchup(
        s4,
        s5,
        wc_winner,
        "Wild Card",
        "prob_reach_ds",
    )


    if actual:

        ds1_winner = (
            actual_series_winner(
                s1,
                wc_winner,
            )
        )

        ds2_winner = (
            actual_series_winner(
                s2,
                s3,
            )
        )

    else:

        ds1_winner = (
            model_series_winner(
                s1,
                wc_winner,
                "prob_reach_cs",
            )
        )

        ds2_winner = (
            model_series_winner(
                s2,
                s3,
                "prob_reach_cs",
            )
        )


    ds1 = make_matchup(
        s1,
        wc_winner,
        ds1_winner,
        "Division Series",
        "prob_reach_cs",
    )


    ds2 = make_matchup(
        s2,
        s3,
        ds2_winner,
        "Division Series",
        "prob_reach_cs",
    )


    if actual:

        cs_winner = (
            actual_series_winner(
                ds1_winner,
                ds2_winner,
            )
        )

    else:

        cs_winner = (
            model_series_winner(
                ds1_winner,
                ds2_winner,
                "prob_reach_ws",
            )
        )


    cs = make_matchup(
        ds1_winner,
        ds2_winner,
        cs_winner,
        "Championship Series",
        "prob_reach_ws",
    )


    return {
        "seeded": seeded_df,

        "wild_card": [
            wc
        ],

        "division_series": [
            ds1,
            ds2,
        ],

        "championship": cs,
        "champion": cs_winner,
    }


# ============================================================
# 8 TEAM FORMAT
# ============================================================

def simulate_four_team_league(
    seeded_df,
    actual=False
):

    s1 = seed_row(seeded_df, 1)
    s2 = seed_row(seeded_df, 2)
    s3 = seed_row(seeded_df, 3)
    s4 = seed_row(seeded_df, 4)


    if (
        s1 is not None
        and s4 is not None
        and s1.get("divID")
        != s4.get("divID")
    ):

        pair1 = (
            s1,
            s4
        )

        pair2 = (
            s2,
            s3
        )

    else:

        pair1 = (
            s2,
            s4
        )

        pair2 = (
            s1,
            s3
        )


    if actual:

        winner1 = (
            actual_series_winner(
                *pair1
            )
        )

        winner2 = (
            actual_series_winner(
                *pair2
            )
        )

    else:

        winner1 = (
            model_series_winner(
                pair1[0],
                pair1[1],
                "prob_reach_cs",
            )
        )

        winner2 = (
            model_series_winner(
                pair2[0],
                pair2[1],
                "prob_reach_cs",
            )
        )


    ds1 = make_matchup(
        pair1[0],
        pair1[1],
        winner1,
        "Division Series",
        "prob_reach_cs",
    )


    ds2 = make_matchup(
        pair2[0],
        pair2[1],
        winner2,
        "Division Series",
        "prob_reach_cs",
    )


    if actual:

        cs_winner = (
            actual_series_winner(
                winner1,
                winner2,
            )
        )

    else:

        cs_winner = (
            model_series_winner(
                winner1,
                winner2,
                "prob_reach_ws",
            )
        )


    cs = make_matchup(
        winner1,
        winner2,
        cs_winner,
        "Championship Series",
        "prob_reach_ws",
    )


    return {
        "seeded": seeded_df,
        "wild_card": [],

        "division_series": [
            ds1,
            ds2,
        ],

        "championship": cs,
        "champion": cs_winner,
    }


# ============================================================
# 2020 FORMAT
# ============================================================

def simulate_eight_team_league(
    seeded_df,
    actual=False
):

    seeds = {
        seed:
            seed_row(
                seeded_df,
                seed
            )
        for seed
        in range(
            1,
            9
        )
    }


    wild_card = []


    for high_seed, low_seed in [
        (1, 8),
        (2, 7),
        (3, 6),
        (4, 5),
    ]:

        team_a = seeds[
            high_seed
        ]

        team_b = seeds[
            low_seed
        ]


        if actual:

            winner = (
                actual_series_winner(
                    team_a,
                    team_b,
                )
            )

        else:

            winner = (
                model_series_winner(
                    team_a,
                    team_b,
                    "prob_reach_ds",
                )
            )


        wild_card.append(
            make_matchup(
                team_a,
                team_b,
                winner,
                "Wild Card",
                "prob_reach_ds",
            )
        )


    w18 = wild_card[0]["winner"]
    w27 = wild_card[1]["winner"]
    w36 = wild_card[2]["winner"]
    w45 = wild_card[3]["winner"]


    if actual:

        ds1_winner = (
            actual_series_winner(
                w18,
                w45,
            )
        )

        ds2_winner = (
            actual_series_winner(
                w27,
                w36,
            )
        )

    else:

        ds1_winner = (
            model_series_winner(
                w18,
                w45,
                "prob_reach_cs",
            )
        )

        ds2_winner = (
            model_series_winner(
                w27,
                w36,
                "prob_reach_cs",
            )
        )


    ds1 = make_matchup(
        w18,
        w45,
        ds1_winner,
        "Division Series",
        "prob_reach_cs",
    )


    ds2 = make_matchup(
        w27,
        w36,
        ds2_winner,
        "Division Series",
        "prob_reach_cs",
    )


    if actual:

        cs_winner = (
            actual_series_winner(
                ds1_winner,
                ds2_winner,
            )
        )

    else:

        cs_winner = (
            model_series_winner(
                ds1_winner,
                ds2_winner,
                "prob_reach_ws",
            )
        )


    cs = make_matchup(
        ds1_winner,
        ds2_winner,
        cs_winner,
        "Championship Series",
        "prob_reach_ws",
    )


    return {
        "seeded": seeded_df,

        "wild_card": wild_card,

        "division_series": [
            ds1,
            ds2,
        ],

        "championship": cs,
        "champion": cs_winner,
    }


def simulate_league_bracket(
    scored_df,
    league,
    year,
    actual=False
):

    seeded = seed_league(
        scored_df,
        league,
        year,
    )


    era = postseason_era(
        year
    )


    if era == "2022+":

        return simulate_modern_league(
            seeded,
            actual,
        )


    if era in [
        "2012",
        "2013-2019",
        "2021",
    ]:

        return simulate_five_team_league(
            seeded,
            actual,
        )


    if era == "2020":

        return simulate_eight_team_league(
            seeded,
            actual,
        )


    return simulate_four_team_league(
        seeded,
        actual,
    )


def build_postseason_bracket(
    scored_df,
    year,
    actual=False
):

    al = simulate_league_bracket(
        scored_df,
        "AL",
        year,
        actual,
    )


    nl = simulate_league_bracket(
        scored_df,
        "NL",
        year,
        actual,
    )


    if actual:

        champion = (
            actual_series_winner(
                al["champion"],
                nl["champion"],
            )
        )

    else:

        champion = (
            model_series_winner(
                al["champion"],
                nl["champion"],
                "prob_win_ws",
            )
        )


    world_series = make_matchup(
        al["champion"],
        nl["champion"],
        champion,
        "World Series",
        "prob_win_ws",
    )


    return {
        "AL": al,
        "NL": nl,
        "world_series":
            world_series,
    }


# ============================================================
# ACTUAL RESULTS CHECK
# ============================================================

def actual_results_available(
    season_df
):

    if season_df.empty:
        return False


    results = (
        season_df[
            "postseason_result"
        ]
        .dropna()
        .astype(str)
        .str.strip()
    )


    results = results[
        ~results.isin(
            [
                "",
                "nan",
                "None",
                "<NA>",
                "Missed Playoffs",
            ]
        )
    ]


    return len(results) > 0


# ============================================================
# MATCHUP REGISTRY
# ============================================================

def build_matchup_registry(
    bracket,
    prefix
):

    registry = {}


    for league in [
        "AL",
        "NL",
    ]:


        for index, matchup in enumerate(
            bracket[
                league
            ][
                "wild_card"
            ]
        ):

            registry[
                f"{prefix}|{league}|WC|{index}"
            ] = matchup


        for index, matchup in enumerate(
            bracket[
                league
            ][
                "division_series"
            ]
        ):

            registry[
                f"{prefix}|{league}|DS|{index}"
            ] = matchup


        registry[
            f"{prefix}|{league}|CS|0"
        ] = bracket[
            league
        ][
            "championship"
        ]


    registry[
        f"{prefix}|WS|WS|0"
    ] = bracket[
        "world_series"
    ]


    return registry


# ============================================================
# MODEL CONTRIBUTIONS
# ============================================================

PROBABILITY_THRESHOLD_MAP = {
    "prob_make_playoffs": 1,
    "prob_reach_ds": 2,
    "prob_reach_cs": 3,
    "prob_reach_ws": 4,
    "prob_win_ws": 5,
}


PROBABILITY_LABEL_MAP = {
    "prob_make_playoffs":
        "Make Playoffs",

    "prob_reach_ds":
        "Reach Division Series",

    "prob_reach_cs":
        "Reach Championship Series",

    "prob_reach_ws":
        "Reach World Series",

    "prob_win_ws":
        "Win World Series",
}


POSTSEASON_FEATURE_LABELS = {
    "runs_per_game":
        "Runs / Game",

    "AVG":
        "AVG",

    "OBP":
        "OBP",

    "SLG":
        "SLG",

    "ISO":
        "ISO",

    "HR_rate":
        "HR Rate",

    "BB_rate":
        "BB Rate",

    "SO_rate":
        "SO Rate",

    "ERA":
        "ERA",

    "WHIP":
        "WHIP",

    "K_per_9":
        "K / 9",

    "BB_per_9":
        "BB / 9",

    "HR_per_9":
        "HR / 9",

    "run_diff_per_game":
        "Run Diff / Game",

    "win_pct":
        "Win %",
}


# ============================================================
# POSTSEASON METRIC DIRECTION
# ============================================================
# These directions are used ONLY for the matchup comparison table.
# They do not change the trained model or any postseason probability.
#
# True  = higher raw value is the favorable baseball edge.
# False = lower raw value is the favorable baseball edge.
POSTSEASON_HIGHER_IS_BETTER = {
    "runs_per_game": True,
    "AVG": True,
    "OBP": True,
    "SLG": True,
    "ISO": True,
    "HR_rate": True,
    "BB_rate": True,
    "SO_rate": False,
    "ERA": False,
    "WHIP": False,
    "K_per_9": True,
    "BB_per_9": False,
    "HR_per_9": False,
    "run_diff_per_game": True,
    "win_pct": True,
}


def postseason_metric_edge(
    feature,
    team_a,
    team_b
):
    """
    Return the team with the favorable raw-stat edge for a metric.

    This is intentionally separate from model contribution. The model
    contribution can have a counterintuitive sign because of learned
    coefficients and feature correlations, while this helper answers the
    simpler question shown in the table: which team has the better value
    for this baseball metric?
    """

    value_a = pd.to_numeric(
        team_a.get(feature, np.nan),
        errors="coerce",
    )

    value_b = pd.to_numeric(
        team_b.get(feature, np.nan),
        errors="coerce",
    )

    if pd.isna(value_a) or pd.isna(value_b):
        return "N/A"

    # Avoid declaring an edge for effectively identical displayed values.
    if np.isclose(
        float(value_a),
        float(value_b),
        rtol=1e-9,
        atol=1e-12,
    ):
        return "EVEN"

    higher_is_better = POSTSEASON_HIGHER_IS_BETTER.get(
        feature,
        True,
    )

    if higher_is_better:
        winner = team_a if value_a > value_b else team_b
    else:
        winner = team_a if value_a < value_b else team_b

    return winner.get(
        "teamID",
        "A" if winner is team_a else "B",
    )


def matchup_feature_contributions(
    team_a,
    team_b,
    probability_column
):

    threshold = (
        PROBABILITY_THRESHOLD_MAP.get(
            probability_column
        )
    )


    if threshold is None:

        return pd.DataFrame()


    pipeline = (
        postseason_threshold_models[
            threshold
        ]
    )


    if not hasattr(
        pipeline,
        "named_steps"
    ):

        return pd.DataFrame()


    scaler = pipeline.named_steps.get(
        "scaler"
    )

    logistic = pipeline.named_steps.get(
        "logistic"
    )


    if (
        scaler is None
        or logistic is None
    ):

        return pd.DataFrame()


    X_a = pd.DataFrame(
        [
            {
                feature:
                    float(
                        team_a.get(
                            feature,
                            0
                        )
                    )
                for feature
                in postseason_features
            }
        ],
        columns=postseason_features,
    )


    X_b = pd.DataFrame(
        [
            {
                feature:
                    float(
                        team_b.get(
                            feature,
                            0
                        )
                    )
                for feature
                in postseason_features
            }
        ],
        columns=postseason_features,
    )


    z_a = scaler.transform(
        X_a
    )[0]

    z_b = scaler.transform(
        X_b
    )[0]


    coefficients = (
        logistic.coef_[0]
    )


    difference = (
        z_a * coefficients
        -
        z_b * coefficients
    )


    results = pd.DataFrame(
        {
            "feature":
                postseason_features,

            "difference":
                difference,
        }
    )


    results[
        "abs_difference"
    ] = (
        results["difference"]
        .abs()
    )


    return results.sort_values(
        "abs_difference",
        ascending=False,
    )


def format_postseason_feature(
    feature,
    value
):

    if pd.isna(value):
        return "N/A"


    value = float(value)


    if feature in [
        "AVG",
        "OBP",
        "SLG",
        "ISO",
        "WHIP",
        "win_pct",
    ]:

        return f"{value:.3f}"


    if feature in [
        "HR_rate",
        "BB_rate",
        "SO_rate",
    ]:

        return (
            f"{value * 100:.1f}%"
        )


    return f"{value:.2f}"


# ============================================================
# BRACKET CARDS
# ============================================================

def bracket_team_card(
    team,
    probability_column=None,
    actual=False,
    winner=False
):

    if team is None:

        return ui.div(
            "—",
            class_="bracket-empty"
        )


    seed = team.get(
        "seed",
        np.nan
    )


    seed_text = (
        f"#{int(seed)}"
        if pd.notna(seed)
        else ""
    )


    record = (
        f"{int(team.get('W', 0))}-"
        f"{int(team.get('L', 0))}"
    )


    if actual:

        footer = str(
            team.get(
                "postseason_result",
                ""
            )
        )

    elif probability_column:

        footer = (
            f"{float(team.get(probability_column, 0)) * 100:.1f}%"
        )

    else:

        footer = ""


    return ui.div(

        ui.div(

            ui.div(
                seed_text,
                class_="bracket-seed"
            ),

            team_logo(
                team.get(
                    "teamID"
                ),
                "32px",
            ),

            ui.div(

                ui.div(
                    team.get(
                        "name",
                        "Unknown"
                    ),
                    class_="bracket-team-name"
                ),

                ui.div(
                    (
                        f"{record} • "
                        f"{team.get('league', '')}"
                        f"{team.get('divID', '')}"
                    ),
                    class_="bracket-team-record"
                ),
            ),

            class_="bracket-team-main"
        ),

        ui.div(
            footer,
            class_=(
                "bracket-team-footer "
                +
                (
                    "actual-footer"
                    if actual
                    else
                    "projection-footer"
                )
            ),
        ),

        class_=(
            "bracket-team-card "
            +
            (
                "bracket-team-winner"
                if winner
                else ""
            )
        ),
    )


def clickable_matchup_card(
    matchup,
    matchup_key,
    click_input_id,
    actual=False
):

    team_a = matchup[
        "team_a"
    ]

    team_b = matchup[
        "team_b"
    ]

    winner = matchup[
        "winner"
    ]

    probability_column = matchup[
        "probability_column"
    ]


    winner_name = (
        winner.name
        if winner is not None
        else None
    )


    onclick = (
        "Shiny.setInputValue("
        f"{json.dumps(click_input_id)},"
        f"{json.dumps(matchup_key)},"
        "{priority:'event'}"
        ");"
    )


    return ui.div(

        bracket_team_card(
            team_a,
            probability_column,
            actual=actual,
            winner=(
                team_a is not None
                and team_a.name
                == winner_name
            ),
        ),

        ui.div(
            "VS",
            class_="bracket-vs"
        ),

        bracket_team_card(
            team_b,
            probability_column,
            actual=actual,
            winner=(
                team_b is not None
                and team_b.name
                == winner_name
            ),
        ),

        ui.div(
            "CLICK TO COMPARE",
            class_="matchup-click-hint"
        ),

        onclick=onclick,

        class_=(
            "bracket-matchup "
            "clickable-matchup"
        ),
    )


def bracket_round_column(
    title,
    matchups,
    prefix,
    league,
    round_code,
    click_input_id,
    actual=False
):

    cards = []


    for index, matchup in enumerate(
        matchups
    ):

        matchup_key = (
            f"{prefix}|"
            f"{league}|"
            f"{round_code}|"
            f"{index}"
        )


        cards.append(
            clickable_matchup_card(
                matchup,
                matchup_key,
                click_input_id,
                actual,
            )
        )


    if not cards:

        cards = [
            ui.div(
                "NO ROUND",
                class_="no-round"
            )
        ]


    return ui.div(

        ui.div(
            title,
            class_="bracket-round-title"
        ),

        ui.div(
            *cards,
            class_="bracket-round-stack"
        ),

        class_="bracket-round-column"
    )


def render_league_bracket(
    bracket,
    league,
    prefix,
    click_input_id,
    actual=False
):

    league_bracket = bracket[
        league
    ]


    seeded = league_bracket[
        "seeded"
    ]


    playoff_field = seeded[
        seeded["seed"].notna()
    ].sort_values(
        "seed"
    )


    seed_column = ui.div(

        ui.div(
            "SEEDS",
            class_="bracket-round-title"
        ),

        ui.div(
            *[
                bracket_team_card(
                    row,
                    (
                        None
                        if actual
                        else "prob_make_playoffs"
                    ),
                    actual=actual,
                )
                for _, row
                in playoff_field.iterrows()
            ],
            class_="seed-card-stack"
        ),

        class_="bracket-round-column"
    )


    wc_column = bracket_round_column(
        "WILD CARD",
        league_bracket[
            "wild_card"
        ],
        prefix,
        league,
        "WC",
        click_input_id,
        actual,
    )


    ds_column = bracket_round_column(
        "DIVISION SERIES",
        league_bracket[
            "division_series"
        ],
        prefix,
        league,
        "DS",
        click_input_id,
        actual,
    )


    cs_key = (
        f"{prefix}|"
        f"{league}|CS|0"
    )


    cs_column = ui.div(

        ui.div(
            (
                "ALCS"
                if league == "AL"
                else "NLCS"
            ),
            class_="bracket-round-title"
        ),

        clickable_matchup_card(
            league_bracket[
                "championship"
            ],
            cs_key,
            click_input_id,
            actual,
        ),

        class_="bracket-round-column"
    )


    champion = league_bracket[
        "champion"
    ]


    champion_column = ui.div(

        ui.div(
            f"{league} CHAMPION",
            class_="bracket-round-title"
        ),

        bracket_team_card(
            champion,
            (
                None
                if actual
                else "prob_reach_ws"
            ),
            actual=actual,
            winner=True,
        ),

        class_="bracket-round-column"
    )


    if league == "AL":

        columns = [
            seed_column,
            wc_column,
            ds_column,
            cs_column,
            champion_column,
        ]

    else:

        # MIRRORED NL BRACKET
        columns = [
            champion_column,
            cs_column,
            ds_column,
            wc_column,
            seed_column,
        ]


    return ui.div(

        ui.div(
            (
                "AMERICAN LEAGUE"
                if league == "AL"
                else
                "NATIONAL LEAGUE"
            ),
            class_=(
                "league-banner "
                +
                (
                    "al-banner"
                    if league == "AL"
                    else
                    "nl-banner"
                )
            ),
        ),

        ui.div(
            *columns,
            class_="league-round-grid"
        ),

        class_="league-bracket"
    )


def render_world_series(
    bracket,
    prefix,
    click_input_id,
    actual=False
):

    matchup = bracket[
        "world_series"
    ]


    team_a = matchup[
        "team_a"
    ]

    team_b = matchup[
        "team_b"
    ]

    winner = matchup[
        "winner"
    ]


    winner_name = (
        winner.name
        if winner is not None
        else None
    )


    matchup_key = (
        f"{prefix}|WS|WS|0"
    )


    onclick = (
        "Shiny.setInputValue("
        f"{json.dumps(click_input_id)},"
        f"{json.dumps(matchup_key)},"
        "{priority:'event'}"
        ");"
    )


    return ui.div(

        ui.div(
            (
                "ACTUAL WORLD SERIES"
                if actual
                else
                "PROJECTED WORLD SERIES"
            ),
            class_="world-series-title"
        ),

        ui.div(

            bracket_team_card(
                team_a,
                (
                    None
                    if actual
                    else "prob_win_ws"
                ),
                actual=actual,
                winner=(
                    team_a is not None
                    and team_a.name
                    == winner_name
                ),
            ),

            ui.div(
                "VS",
                class_="world-series-vs"
            ),

            bracket_team_card(
                team_b,
                (
                    None
                    if actual
                    else "prob_win_ws"
                ),
                actual=actual,
                winner=(
                    team_b is not None
                    and team_b.name
                    == winner_name
                ),
            ),

            ui.div(
                "CLICK TO COMPARE",
                class_="matchup-click-hint"
            ),

            onclick=onclick,

            class_=(
                "world-series-matchup "
                "clickable-matchup"
            ),
        ),

        ui.div(

            ui.div(
                (
                    "ACTUAL CHAMPION"
                    if actual
                    else
                    "PROJECTED CHAMPION"
                ),
                class_="champion-label"
            ),

            ui.div(
                (
                    winner.get(
                        "name",
                        "N/A"
                    )
                    if winner is not None
                    else "N/A"
                ),
                class_="champion-name"
            ),

            (
                ui.div(
                    (
                        f"{float(winner.get('prob_win_ws', 0)) * 100:.1f}%"
                    ),
                    class_="champion-probability"
                )
                if (
                    not actual
                    and winner is not None
                )
                else None
            ),

            class_="champion-box"
        ),

        class_="world-series-panel"
    )


# ============================================================
# IMPORTANT:
# BRACKET DOES NOT READ matchup_click
# ============================================================

def render_complete_bracket(
    bracket,
    prefix,
    click_input_id,
    actual=False
):

    return ui.div(

        render_league_bracket(
            bracket,
            "AL",
            prefix,
            click_input_id,
            actual,
        ),

        render_world_series(
            bracket,
            prefix,
            click_input_id,
            actual,
        ),

        render_league_bracket(
            bracket,
            "NL",
            prefix,
            click_input_id,
            actual,
        ),

        class_="complete-bracket-grid"
    )


# ============================================================
# MATCHUP ANALYSIS
# ============================================================

def matchup_team_header(
    team,
    probability_column
):

    probability = float(
        team.get(
            probability_column,
            0
        )
    )


    return ui.div(

        team_logo(
            team.get(
                "teamID"
            ),
            "68px",
        ),

        ui.div(
            team.get(
                "name",
                "Unknown"
            ),
            class_="analysis-team-name"
        ),

        ui.div(
            (
                f"{int(team.get('W', 0))}-"
                f"{int(team.get('L', 0))}"
            ),
            class_="analysis-team-record"
        ),

        ui.div(
            f"{probability * 100:.1f}%",
            class_="analysis-team-prob"
        ),

        ui.div(
            PROBABILITY_LABEL_MAP.get(
                probability_column,
                probability_column
            ),
            class_="analysis-prob-label"
        ),

        class_="analysis-team-card"
    )


def make_matchup_analysis(
    matchup,
    actual=False
):

    team_a = matchup[
        "team_a"
    ]

    team_b = matchup[
        "team_b"
    ]


    if (
        team_a is None
        or team_b is None
    ):

        return ui.div(
            "Matchup unavailable.",
            class_="matchup-empty-panel"
        )


    probability_column = matchup[
        "probability_column"
    ]


    model_winner = (
        model_series_winner(
            team_a,
            team_b,
            probability_column,
        )
    )


    contributions = (
        matchup_feature_contributions(
            team_a,
            team_b,
            probability_column,
        )
    )


    contribution_lookup = {}


    if not contributions.empty:

        contribution_lookup = (
            contributions
            .set_index("feature")
            .to_dict("index")
        )


    feature_rows = []


    for feature in postseason_features:

        # The table shows the favorable RAW-STAT edge, not the
        # signed logistic-regression contribution. Model contribution
        # differences remain available in the bar section below.
        edge = postseason_metric_edge(
            feature,
            team_a,
            team_b,
        )


        feature_rows.append(

            ui.div(

                ui.div(
                    POSTSEASON_FEATURE_LABELS.get(
                        feature,
                        feature
                    )
                ),

                ui.div(
                    format_postseason_feature(
                        feature,
                        team_a.get(
                            feature,
                            np.nan
                        )
                    )
                ),

                ui.div(
                    format_postseason_feature(
                        feature,
                        team_b.get(
                            feature,
                            np.nan
                        )
                    )
                ),

                ui.div(
                    edge,
                    class_="model-edge"
                ),

                class_="analysis-feature-row"
            )
        )


    bars = []


    if not contributions.empty:

        top = contributions.head(6)


        max_abs = max(
            float(
                top[
                    "abs_difference"
                ].max()
            ),
            1e-9,
        )


        for _, row in top.iterrows():

            difference = float(
                row["difference"]
            )


            edge_team = (
                team_a.get(
                    "teamID",
                    "A"
                )
                if difference >= 0
                else
                team_b.get(
                    "teamID",
                    "B"
                )
            )


            width = (
                abs(difference)
                /
                max_abs
                *
                100
            )


            bars.append(

                ui.div(

                    ui.div(
                        POSTSEASON_FEATURE_LABELS.get(
                            row["feature"],
                            row["feature"]
                        ),
                        class_="edge-bar-label"
                    ),

                    ui.div(

                        ui.div(
                            class_="edge-bar-fill",
                            style=(
                                f"width:{width:.1f}%;"
                            ),
                        ),

                        class_="edge-bar-track"
                    ),

                    ui.div(
                        edge_team,
                        class_="edge-bar-team"
                    ),

                    class_="edge-bar-row"
                )
            )


    if actual:

        actual_winner = matchup[
            "winner"
        ]


        result_panel = ui.div(

            ui.div(

                ui.div(
                    "MODEL PICK",
                    class_="result-label"
                ),

                ui.div(
                    model_winner.get(
                        "name",
                        "N/A"
                    ),
                    class_="model-pick"
                ),

                class_="result-card"
            ),

            ui.div(

                ui.div(
                    "ACTUAL WINNER",
                    class_="result-label"
                ),

                ui.div(
                    actual_winner.get(
                        "name",
                        "N/A"
                    ),
                    class_="actual-pick"
                ),

                class_="result-card"
            ),

            class_="result-grid"
        )

    else:

        result_panel = ui.div(

            ui.div(
                "MODEL PICK",
                class_="result-label"
            ),

            ui.div(
                model_winner.get(
                    "name",
                    "N/A"
                ),
                class_="model-pick"
            ),

            class_="projected-pick-box"
        )


    return ui.div(

        ui.div(
            "SELECTED MATCHUP",
            class_="analysis-kicker"
        ),

        ui.div(
            matchup[
                "round"
            ].upper(),
            class_="analysis-title"
        ),

        ui.div(
            (
                "Model metric: "
                f"{PROBABILITY_LABEL_MAP.get(probability_column)}"
            ),
            class_="analysis-subtitle"
        ),

        ui.div(
            (
                "These percentages represent season-level "
                "advancement probabilities rather than "
                "direct head-to-head series probabilities."
            ),
            class_="analysis-note"
        ),

        ui.div(

            matchup_team_header(
                team_a,
                probability_column,
            ),

            ui.div(
                "VS",
                class_="analysis-vs"
            ),

            matchup_team_header(
                team_b,
                probability_column,
            ),

            class_="analysis-team-grid"
        ),

        result_panel,

        ui.div(
            "REGULAR-SEASON MODEL FEATURES",
            class_="analysis-section-title"
        ),

        ui.div(

            ui.div(
                "METRIC"
            ),

            ui.div(
                team_a.get(
                    "teamID",
                    "A"
                )
            ),

            ui.div(
                team_b.get(
                    "teamID",
                    "B"
                )
            ),

            ui.div(
                "METRIC EDGE"
            ),

            class_="analysis-feature-header"
        ),

        *feature_rows,

        ui.div(
            "BIGGEST MODEL CONTRIBUTION DIFFERENCES",
            class_="analysis-section-title"
        ),

        *bars,

        class_="matchup-analysis-panel"
    )


# ============================================================
# POSTSEASON RANKINGS
# ============================================================

def postseason_ranking_row(
    row,
    rank,
    metric
):

    probability = float(
        row.get(
            metric,
            0
        )
    )


    return ui.div(

        ui.div(
            str(rank),
            class_="ranking-number"
        ),

        ui.div(

            team_logo(
                row.get(
                    "teamID"
                ),
                "38px",
            ),

            ui.div(

                ui.div(
                    row.get(
                        "name",
                        "Unknown"
                    ),
                    class_="ranking-team-name"
                ),

                ui.div(
                    (
                        f"{row.get('league', '')}"
                        f"{row.get('divID', '')} • "
                        f"{int(row.get('W', 0))}-"
                        f"{int(row.get('L', 0))}"
                    ),
                    class_="ranking-team-sub"
                ),
            ),

            class_="ranking-team-block"
        ),

        ui.div(

            ui.div(
                class_="ranking-bar-fill",
                style=(
                    f"width:{probability * 100:.1f}%;"
                ),
            ),

            class_="ranking-bar-track"
        ),

        ui.div(
            f"{probability * 100:.1f}%",
            class_="ranking-probability"
        ),

        class_="ranking-row"
    )


def make_postseason_ranking(
    dataframe,
    metric,
    title
):

    ranked = (
        dataframe
        .sort_values(
            metric,
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    return ui.div(

        ui.div(
            title,
            class_="ranking-title"
        ),

        *[
            postseason_ranking_row(
                row,
                index + 1,
                metric,
            )
            for index, row
            in ranked.iterrows()
        ],

        class_="ranking-panel"
    )


# ============================================================
# CUSTOM POSTSEASON TEAM
# ============================================================

def make_custom_postseason_scoreboard(
    team
):

    scored = score_postseason_dataframe(
        team
    )

    row = scored.iloc[0]


    probabilities = [
        (
            "MAKE PLAYOFFS",
            row[
                "prob_make_playoffs"
            ]
        ),
        (
            "REACH DIVISION SERIES",
            row[
                "prob_reach_ds"
            ]
        ),
        (
            "REACH CHAMPIONSHIP SERIES",
            row[
                "prob_reach_cs"
            ]
        ),
        (
            "REACH WORLD SERIES",
            row[
                "prob_reach_ws"
            ]
        ),
        (
            "WIN WORLD SERIES",
            row[
                "prob_win_ws"
            ]
        ),
    ]


    return ui.div(

        ui.div(

            ui.div(
                "CUSTOM TEAM POSTSEASON MODEL",
                class_="marquee-small"
            ),

            ui.div(
                row.get(
                    "name",
                    "Custom Team"
                ),
                class_="marquee-name"
            ),

            class_="custom-postseason-header"
        ),

        ui.div(
            *[
                ui.div(

                    ui.div(
                        label,
                        class_="postseason-step-label"
                    ),

                    ui.div(
                        f"{probability * 100:.1f}%",
                        class_="postseason-step-prob"
                    ),

                    ui.div(

                        ui.div(
                            class_="postseason-progress-fill",
                            style=(
                                f"width:{probability * 100:.1f}%;"
                            ),
                        ),

                        class_="postseason-progress-track"
                    ),

                    class_="postseason-probability-card"
                )

                for label, probability
                in probabilities
            ],

            class_="postseason-ladder"
        ),

        class_="scoreboard"
    )


# ============================================================
# POSTSEASON UI
# ============================================================

@module.ui
def postseason_ui():

    years = sorted(
        postseason_dataset[
            "yearID"
        ]
        .dropna()
        .astype(int)
        .unique(),
        reverse=True,
    )


    choices = [
        str(year)
        for year in years
    ]


    return ui.div(

        # Hidden click input.
        #
        # It ONLY controls the analysis section.
        # Bracket rendering does not read this value.
        ui.div(

            ui.input_text(
                "matchup_click",
                "",
                value="",
            ),

            style="display:none;",
        ),

        ui.div(

            ui.h1(
                "POSTSEASON ADVANCEMENT PREDICTOR",
                class_="dashboard-title"
            ),

            ui.p(
                (
                    "Real MLB playoff seeding • "
                    "Interactive matchups • "
                    "Actual results • "
                    "Model explanation"
                ),
                class_="dashboard-subtitle"
            ),

            class_="dashboard-header"
        ),

        ui.div(

            ui.layout_columns(

                ui.div(

                    ui.h3(
                        "SELECT SEASON"
                    ),

                    ui.input_select(
                        "postseason_year",
                        "MLB Season",
                        choices=choices,
                        selected=str(
                            POSTSEASON_DATA_YEAR
                        ),
                    ),

                    class_="season-control-box"
                ),

                ui.div(

                    ui.h3(
                        "POSTSEASON FORMAT"
                    ),

                    ui.output_ui(
                        "postseason_format_summary"
                    ),

                    class_="season-control-box"
                ),

                col_widths=[
                    6,
                    6,
                ],
            ),

            class_="selection-panel"
        ),

        ui.navset_tab(

            ui.nav_panel(

                "Postseason Bracket",

                ui.output_ui(
                    "postseason_bracket"
                ),

                ui.output_ui(
                    "matchup_analysis"
                ),
            ),

            ui.nav_panel(

                "Team Probability Rankings",

                ui.navset_tab(

                    ui.nav_panel(
                        "Make Playoffs",
                        ui.output_ui(
                            "ranking_make_playoffs"
                        ),
                    ),

                    ui.nav_panel(
                        "Reach Division Series",
                        ui.output_ui(
                            "ranking_reach_ds"
                        ),
                    ),

                    ui.nav_panel(
                        "Reach Championship Series",
                        ui.output_ui(
                            "ranking_reach_cs"
                        ),
                    ),

                    ui.nav_panel(
                        "Reach World Series",
                        ui.output_ui(
                            "ranking_reach_ws"
                        ),
                    ),

                    ui.nav_panel(
                        "Win World Series",
                        ui.output_ui(
                            "ranking_win_ws"
                        ),
                    ),
                ),
            ),

            ui.nav_panel(

                "Custom Team",

                ui.div(

                    ui.layout_columns(

                        ui.input_text(
                            "custom_team_name",
                            "Team Name",
                            value="Custom Team",
                        ),

                        ui.input_numeric(
                            "custom_team_games",
                            "Games",
                            value=162,
                            min=1,
                        ),

                        ui.input_numeric(
                            "custom_team_wins",
                            "Wins",
                            value=95,
                            min=0,
                        ),

                        col_widths=[
                            4,
                            4,
                            4,
                        ],
                    ),

                    ui.div(
                        "OFFENSE",
                        class_="score-section-title"
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_team_runs",
                            "Runs",
                            value=790,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_ab",
                            "At-Bats",
                            value=5500,
                            min=1,
                        ),

                        ui.input_numeric(
                            "custom_team_hits",
                            "Hits",
                            value=1400,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_doubles",
                            "Doubles",
                            value=280,
                            min=0,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_team_triples",
                            "Triples",
                            value=25,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_hr",
                            "Home Runs",
                            value=220,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_bb",
                            "Walks",
                            value=550,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_so",
                            "Strikeouts",
                            value=1350,
                            min=0,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_team_hbp",
                            "HBP",
                            value=70,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_sf",
                            "Sacrifice Flies",
                            value=45,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_ra",
                            "Runs Allowed",
                            value=690,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_era",
                            "ERA",
                            value=3.75,
                            step=.01,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.div(
                        "PITCHING",
                        class_="score-section-title"
                    ),

                    ui.layout_columns(

                        ui.input_numeric(
                            "custom_team_ipouts",
                            "Pitching Outs",
                            value=4374,
                            min=1,
                        ),

                        ui.input_numeric(
                            "custom_team_ha",
                            "Hits Allowed",
                            value=1325,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_hra",
                            "Home Runs Allowed",
                            value=180,
                            min=0,
                        ),

                        ui.input_numeric(
                            "custom_team_bba",
                            "Walks Allowed",
                            value=490,
                            min=0,
                        ),

                        col_widths=[
                            3,
                            3,
                            3,
                            3,
                        ],
                    ),

                    ui.input_numeric(
                        "custom_team_soa",
                        "Pitching Strikeouts",
                        value=1450,
                        min=0,
                    ),

                    ui.input_action_button(
                        "predict_custom_postseason",
                        "RUN POSTSEASON MODEL",
                        class_="predict-button",
                    ),

                    class_="custom-panel"
                ),

                ui.output_ui(
                    "custom_postseason_scoreboard"
                ),
            ),
        ),
    )


# ============================================================
# POSTSEASON SERVER
# ============================================================

@module.server
def postseason_server(
    input,
    output,
    session
):

    @reactive.calc
    def year():

        return int(
            input.postseason_year()
            or POSTSEASON_DATA_YEAR
        )


    @reactive.calc
    def season():

        return postseason_dataset[
            postseason_dataset[
                "yearID"
            ]
            == year()
        ].copy()


    @reactive.calc
    def scored():

        return score_postseason_dataframe(
            season()
        )


    @reactive.calc
    def projected_bracket():

        return build_postseason_bracket(
            scored(),
            year(),
            actual=False,
        )


    @reactive.calc
    def actual_bracket():

        if not actual_results_available(
            season()
        ):

            return None


        return build_postseason_bracket(
            scored(),
            year(),
            actual=True,
        )


    @reactive.calc
    def registry():

        result = (
            build_matchup_registry(
                projected_bracket(),
                "proj",
            )
        )


        actual = actual_bracket()


        if actual is not None:

            result.update(
                build_matchup_registry(
                    actual,
                    "actual",
                )
            )


        return result


    @output
    @render.ui
    def postseason_format_summary():

        return ui.div(

            ui.div(
                postseason_format_label(
                    year()
                ),
                class_="selected-position-chip"
            ),

            ui.div(
                (
                    f"Era: "
                    f"{postseason_era(year())}"
                ),
                class_="postseason-format-detail"
            ),
        )


    # ========================================================
    # BRACKET RENDERING
    #
    # IMPORTANT:
    # NO matchup_click IS READ HERE.
    # ========================================================

    @output
    @render.ui
    def postseason_bracket():

        projected = projected_bracket()

        actual = actual_bracket()


        blocks = [

            ui.div(

                ui.div(
                    f"{year()} MODEL PROJECTION",
                    class_="bracket-kicker"
                ),

                ui.div(
                    "PROJECTED MLB POSTSEASON",
                    class_="bracket-main-title"
                ),

                

                class_="bracket-header"
            ),

            render_complete_bracket(
                projected,
                "proj",
                session.ns(
                    "matchup_click"
                ),
                actual=False,
            ),
        ]


        if actual is not None:

            blocks.extend(
                [

                    ui.div(
                        class_="bracket-divider"
                    ),

                    ui.div(

                        ui.div(
                            "ACTUAL POSTSEASON RESULTS",
                            class_="bracket-main-title"
                        ),

                        ui.div(
                            (
                                f"{year()} actual postseason path"
                            ),
                            class_="bracket-subtitle"
                        ),

                        class_="bracket-header"
                    ),

                    render_complete_bracket(
                        actual,
                        "actual",
                        session.ns(
                            "matchup_click"
                        ),
                        actual=True,
                    ),
                ]
            )


        return ui.div(
            *blocks,
            class_="postseason-bracket-panel"
        )


    # ========================================================
    # CLICK ANALYSIS
    # ========================================================

    @output
    @render.ui
    def matchup_analysis():

        matchups = registry()


        clicked = (
            input.matchup_click()
            or ""
        )


        if clicked in matchups:

            key = clicked

        elif (
            "proj|AL|WC|0"
            in matchups
        ):

            key = (
                "proj|AL|WC|0"
            )

        elif matchups:

            key = next(
                iter(matchups)
            )

        else:

            return ui.div(
                "No matchup available.",
                class_="matchup-empty-panel"
            )


        return make_matchup_analysis(
            matchups[key],
            actual=key.startswith(
                "actual|"
            ),
        )


    # ========================================================
    # RANKINGS
    # ========================================================

    @output
    @render.ui
    def ranking_make_playoffs():

        return make_postseason_ranking(
            scored(),
            "prob_make_playoffs",
            "MAKE PLAYOFFS",
        )


    @output
    @render.ui
    def ranking_reach_ds():

        return make_postseason_ranking(
            scored(),
            "prob_reach_ds",
            "REACH DIVISION SERIES",
        )


    @output
    @render.ui
    def ranking_reach_cs():

        return make_postseason_ranking(
            scored(),
            "prob_reach_cs",
            "REACH CHAMPIONSHIP SERIES",
        )


    @output
    @render.ui
    def ranking_reach_ws():

        return make_postseason_ranking(
            scored(),
            "prob_reach_ws",
            "REACH WORLD SERIES",
        )


    @output
    @render.ui
    def ranking_win_ws():

        return make_postseason_ranking(
            scored(),
            "prob_win_ws",
            "WIN WORLD SERIES",
        )


    # ========================================================
    # CUSTOM POSTSEASON
    # ========================================================

    @reactive.calc
    @reactive.event(
        input.predict_custom_postseason
    )
    def custom_postseason_team():

        G = max(
            float(
                input.custom_team_games()
                or 1
            ),
            1,
        )


        W = min(
            max(
                float(
                    input.custom_team_wins()
                    or 0
                ),
                0,
            ),
            G,
        )


        L = G - W


        R = float(
            input.custom_team_runs()
            or 0
        )


        RA = float(
            input.custom_team_ra()
            or 0
        )


        AB = max(
            float(
                input.custom_team_ab()
                or 1
            ),
            1,
        )


        H = float(
            input.custom_team_hits()
            or 0
        )


        doubles = float(
            input.custom_team_doubles()
            or 0
        )


        triples = float(
            input.custom_team_triples()
            or 0
        )


        HR = float(
            input.custom_team_hr()
            or 0
        )


        BB = float(
            input.custom_team_bb()
            or 0
        )


        SO = float(
            input.custom_team_so()
            or 0
        )


        HBP = float(
            input.custom_team_hbp()
            or 0
        )


        SF = float(
            input.custom_team_sf()
            or 0
        )


        ERA = float(
            input.custom_team_era()
            or 0
        )


        IPouts = max(
            float(
                input.custom_team_ipouts()
                or 1
            ),
            1,
        )


        HA = float(
            input.custom_team_ha()
            or 0
        )


        HRA = float(
            input.custom_team_hra()
            or 0
        )


        BBA = float(
            input.custom_team_bba()
            or 0
        )


        SOA = float(
            input.custom_team_soa()
            or 0
        )


        PA = (
            AB
            + BB
            + HBP
            + SF
        )


        singles = max(
            H
            - doubles
            - triples
            - HR,
            0,
        )


        TB = (
            singles
            + 2 * doubles
            + 3 * triples
            + 4 * HR
        )


        AVG = H / AB


        OBP = (
            (
                H
                + BB
                + HBP
            )
            /
            PA
            if PA > 0
            else 0
        )


        SLG = TB / AB

        ISO = SLG - AVG


        IP = IPouts / 3


        WHIP = (
            (HA + BBA)
            /
            IP
            if IP > 0
            else 0
        )


        return pd.DataFrame(
            {
                "name": [
                    input.custom_team_name()
                    or "Custom Team"
                ],

                "G": [G],
                "W": [W],
                "L": [L],

                "runs_per_game": [
                    R / G
                ],

                "AVG": [AVG],
                "OBP": [OBP],
                "SLG": [SLG],
                "ISO": [ISO],

                "HR_rate": [
                    HR / PA
                    if PA > 0
                    else 0
                ],

                "BB_rate": [
                    BB / PA
                    if PA > 0
                    else 0
                ],

                "SO_rate": [
                    SO / PA
                    if PA > 0
                    else 0
                ],

                "ERA": [ERA],
                "WHIP": [WHIP],

                "K_per_9": [
                    SOA / IP * 9
                    if IP > 0
                    else 0
                ],

                "BB_per_9": [
                    BBA / IP * 9
                    if IP > 0
                    else 0
                ],

                "HR_per_9": [
                    HRA / IP * 9
                    if IP > 0
                    else 0
                ],

                "run_diff_per_game": [
                    (R - RA) / G
                ],

                "win_pct": [
                    W / G
                ],
            }
        )


    @output
    @render.ui
    def custom_postseason_scoreboard():

        return make_custom_postseason_scoreboard(
            custom_postseason_team()
        )


# ============================================================
# CSS
# ============================================================

scoreboard_css = r"""

body {
    margin: 0;
    background:
        radial-gradient(
            circle at center top,
            #244d31 0%,
            #102c1d 45%,
            #07140d 100%
        );
    color: #f3f3f3;
    font-family: Arial, Helvetica, sans-serif;
}

.container-fluid {
    max-width: 1800px;
    padding: 20px;
}


/* =========================================================
   HEADER
   ========================================================= */

.app-shell-header,
.dashboard-header {
    background:
        linear-gradient(
            180deg,
            #111923,
            #081018
        );
    border: 4px solid #c6a95c;
    border-radius: 12px;
    padding: 20px 26px;
    margin-bottom: 18px;
    text-align: center;
}

.app-shell-title,
.dashboard-title {
    margin: 0;
    color: #f6e9a8;
    font-weight: 900;
    letter-spacing: 4px;
}

.app-shell-title {
    font-size: 34px;
}

.dashboard-title {
    font-size: 30px;
}

.app-shell-subtitle,
.dashboard-subtitle {
    color: #b9c7d2;
    margin-top: 6px;
}


/* =========================================================
   NAV
   ========================================================= */

.nav-tabs {
    border-bottom: 3px solid #c6a95c;
    margin-bottom: 18px;
}

.nav-tabs .nav-link {
    background: #10261a;
    color: #dbe3d7;
    border: 1px solid #31523b;
    font-weight: 800;
}

.nav-tabs .nav-link.active {
    background: #8f1d2c;
    color: white;
    border-color: #c6a95c;
}


/* =========================================================
   INPUT PANELS
   ========================================================= */

.selection-panel,
.custom-panel {
    background:
        linear-gradient(
            135deg,
            #10291c,
            #0a1b12
        );
    border: 2px solid #526b57;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 18px;
}

.season-control-box {
    background: #07140d;
    border: 2px solid #45564a;
    border-radius: 9px;
    padding: 13px;
}

.form-control,
.form-select,
.selectize-input {
    background: #f3f4e9 !important;
    color: #111 !important;
    border: 2px solid #c6a95c !important;
}

.predict-button {
    background: #8f1d2c !important;
    border: 2px solid #c6a95c !important;
    color: white !important;
    font-weight: 900 !important;
    margin-top: 12px;
}

.selected-position-chip {
    display: inline-block;
    background: #8f1d2c;
    border: 2px solid #c6a95c;
    color: white;
    font-size: 20px;
    font-weight: 900;
    padding: 8px 16px;
}


/* =========================================================
   HOF
   ========================================================= */

.instruction-panel {
    background: #071725;
    border: 2px solid #58728c;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
}

.instruction-title {
    color: #f3e5a3;
    font-weight: 900;
}

.instruction-grid {
    display: grid;
    grid-template-columns:
        repeat(2,1fr);
    gap: 18px;
}

.instruction-grid > div {
    border-left: 4px solid #c6a95c;
    padding: 10px;
}

.scoreboard {
    background:
        linear-gradient(
            180deg,
            #173621,
            #0d2517
        );
    border: 6px solid #b69a55;
    border-radius: 16px;
    padding: 22px;
    margin: 22px 0;
}

.player-marquee {
    display: grid;
    grid-template-columns:
        210px 1fr;
    gap: 22px;
    align-items: center;
    background: #080c0a;
    border: 4px solid #465b4b;
    padding: 18px;
    border-radius: 10px;
}

.marquee-photo-column {
    display: flex;
    justify-content: center;
}

.marquee-info-column {
    text-align: center;
}

.headshot-frame,
.headshot-placeholder {
    width: 180px;
    height: 180px;
    border-radius: 12px;
    border: 4px solid #c6a95c;
}

.headshot-frame {
    overflow: hidden;
}

.player-headshot {
    width: 100%;
    height: 100%;
    object-fit: contain;
}

.headshot-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
}

.marquee-small {
    color: #e5cc75;
    font-weight: 900;
}

.marquee-name {
    color: #ffd94d;
    font-size: 38px;
    font-weight: 900;
}

.marquee-subtitle {
    color: #d4dfd4;
}

.score-section-title {
    color: #efe3a2;
    font-weight: 900;
    margin: 18px 0 10px;
}

.score-grid {
    display: grid;
    grid-template-columns:
        repeat(
            5,
            minmax(110px,1fr)
        );
    gap: 10px;
}

.score-cell {
    background: #080d09;
    border: 2px solid #45564a;
    border-radius: 8px;
    padding: 10px;
    text-align: center;
}

.score-cell-label {
    color: #c5cec6;
    font-size: 11px;
    font-weight: 800;
}

.score-cell-value {
    color: #ffd84c;
    font-size: 24px;
    font-weight: 900;
}

.big-score-grid {
    display: grid;
    grid-template-columns:
        1fr 1fr;
    gap: 15px;
}

.big-score-panel {
    background: #070c09;
    border: 3px solid #4f6355;
    border-radius: 10px;
    padding: 20px;
    text-align: center;
}

.big-panel-label {
    font-weight: 900;
}

.big-panel-number {
    color: #ffd84c;
    font-size: 50px;
    font-weight: 900;
}

.designation-on {
    color: #68ff88;
    font-size: 28px;
    font-weight: 900;
}

.designation-off {
    color: #ff6868;
    font-size: 28px;
    font-weight: 900;
}

.threshold-board,
.scoreboard-note {
    text-align: center;
    margin-top: 12px;
}


/* =========================================================
   2027 HOF BALLOT
   ========================================================= */

.hof-ballot-intro {
    background: linear-gradient(180deg, #0c1d2b, #07131d);
    border: 3px solid #58728c;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 20px;
}

.hof-ballot-section-title,
.hof-ballot-detail-title,
.hof-ballot-board-title {
    color: #f6e9a8;
    font-weight: 900;
    letter-spacing: 1px;
}

.hof-ballot-section-description,
.hof-ballot-board-note {
    color: #c5d0d8;
}

.hof-ballot-board {
    margin-top: 18px;
}

.hof-ballot-board-title {
    font-size: 20px;
    margin-bottom: 4px;
}

.hof-ballot-board-note {
    font-size: 12px;
    margin-bottom: 14px;
}

.hof-ballot-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
}

.hof-ballot-card {
    background: linear-gradient(180deg, #102a1b, #08170f);
    border: 2px solid #4e6554;
    border-radius: 10px;
    padding: 14px;
    min-height: 180px;
    display: flex;
    flex-direction: column;
}

.hof-ballot-card-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.hof-ballot-rank {
    color: #f6d768;
    font-weight: 900;
    font-size: 17px;
}

.hof-ballot-status {
    font-size: 10px;
    font-weight: 900;
    letter-spacing: .7px;
    border-radius: 999px;
    padding: 5px 8px;
}

.hof-ballot-first {
    background: #143e5a;
    border: 1px solid #63a9d4;
    color: #cfeeff;
}

.hof-ballot-returning {
    background: #4d3a13;
    border: 1px solid #d6b34e;
    color: #fff0af;
}

.hof-ballot-player {
    color: #ffffff;
    font-size: 19px;
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: 12px;
}

.hof-ballot-prob-wrap {
    margin-top: auto;
}

.hof-ballot-prob-label {
    color: #aebcb2;
    font-size: 9px;
    font-weight: 800;
    letter-spacing: .8px;
}

.hof-ballot-prob {
    color: #ffd84c;
    font-size: 28px;
    font-weight: 900;
}

.hof-ballot-designation {
    margin-top: 7px;
    padding-top: 7px;
    border-top: 1px solid #304237;
    font-size: 11px;
    font-weight: 900;
}

.hof-ballot-positive {
    color: #68ff88;
}

.hof-ballot-negative {
    color: #ff9a9a;
}

.hof-ballot-unavailable {
    color: #aeb8bf;
}

.hof-ballot-filter-panel {
    margin: 16px 0 18px 0;
    padding: 14px;
    border: 1px solid #355168;
    border-radius: 10px;
    background: #081722;
}

.hof-ballot-filter-summary {
    margin-top: 8px;
    color: #f6e9a8;
    font-size: 12px;
    font-weight: 900;
    letter-spacing: .6px;
}

.hof-ballot-player-header {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 12px;
}

.hof-ballot-card-info {
    flex: 1;
    min-width: 0;
}

.hof-ballot-headshot {
    width: 76px;
    height: 76px;
    border-radius: 10px;
    overflow: hidden;
    border: 2px solid #536c5a;
    background: #07110b;
    flex: 0 0 76px;
}

.hof-ballot-headshot-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: top center;
}

.hof-ballot-headshot-placeholder {
    width: 76px;
    height: 76px;
    border-radius: 10px;
    border: 2px solid #536c5a;
    background: #07110b;
    color: #8fa49a;
    font-size: 9px;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    flex: 0 0 76px;
}

.hof-ballot-position {
    background: #223128;
    border: 1px solid #64796a;
    color: #dfe8e1;
    border-radius: 999px;
    padding: 5px 8px;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: .5px;
}

@media (max-width: 1100px) {
    .hof-ballot-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 700px) {
    .hof-ballot-grid {
        grid-template-columns: 1fr;
    }
}


/* =========================================================
   ALL MLB
   ========================================================= */

.allmlb-position-help {
    color: #8fa49a;
    font-size: 11px;
    margin-top: 8px;
    margin-bottom: 0;
}


/* =========================================================
   DIAMOND PANEL
   ========================================================= */

.diamond-panel {
    background:
        linear-gradient(
            180deg,
            #102719 0%,
            #08180f 100%
        );

    border: 5px solid #b69a55;
    border-radius: 16px;

    padding: 20px;
    margin-bottom: 18px;

    box-shadow:
        inset 0 0 24px
        rgba(0, 0, 0, .25);
}


.diamond-heading {
    text-align: center;
    margin-bottom: 12px;
}


.diamond-title {
    color: #f6e9a8;

    font-size: 24px;
    font-weight: 900;

    letter-spacing: 1px;

    margin-bottom: 5px;
}


.diamond-description {
    color: #aebdb4;

    max-width: 760px;

    margin: 0 auto;

    font-size: 12px;
    line-height: 1.45;
}


/* =========================================================
   BASEBALL FIELD
   ========================================================= */

.baseball-field {
    position: relative;

    width: min(
        100%,
        760px
    );

    aspect-ratio: 1.22 / 1;

    margin:
        20px auto
        16px auto;

    overflow: hidden;

    background:
        repeating-linear-gradient(
            90deg,
            rgba(255, 255, 255, .025) 0,
            rgba(255, 255, 255, .025) 45px,
            rgba(0, 0, 0, .025) 45px,
            rgba(0, 0, 0, .025) 90px
        ),
        radial-gradient(
            ellipse at center 70%,
            #31834b 0%,
            #256e3d 40%,
            #19572f 72%,
            #103c23 100%
        );

    border: 5px solid #d5bd76;

    border-radius:
        52%
        52%
        12%
        12%;

    box-shadow:
        inset 0 0 40px
        rgba(0, 0, 0, .35),

        0 8px 20px
        rgba(0, 0, 0, .25);
}


/* =========================================================
   INFIELD DIRT
   ========================================================= */

.baseball-field::before {
    content: "";

    position: absolute;

    width: 45%;
    aspect-ratio: 1 / 1;

    left: 50%;
    top: 61%;

    transform:
        translate(
            -50%,
            -50%
        )
        rotate(45deg);

    background:
        linear-gradient(
            135deg,
            #bc985f,
            #947142
        );

    border:
        4px solid
        #d5bd91;

    box-shadow:
        inset 0 0 15px
        rgba(65, 42, 19, .30);

    z-index: 1;
}


/* =========================================================
   INNER INFIELD GRASS
   ========================================================= */

.baseball-field::after {
    content: "";

    position: absolute;

    width: 23%;
    aspect-ratio: 1 / 1;

    left: 50%;
    top: 61%;

    transform:
        translate(
            -50%,
            -50%
        )
        rotate(45deg);

    background: #287443;

    border:
        2px solid
        rgba(255, 255, 255, .12);

    z-index: 2;
}


/* =========================================================
   FIELD LABELS
   ========================================================= */

.field-zone-label {
    position: absolute;

    z-index: 4;

    pointer-events: none;

    font-size: 10px;
    font-weight: 900;

    letter-spacing: 4px;

    color:
        rgba(255, 255, 255, .28);
}


.field-outfield-label {
    top: 8%;
    left: 50%;

    transform:
        translateX(-50%);
}


.field-infield-label {
    top: 68%;
    left: 50%;

    transform:
        translateX(-50%);

    color:
        rgba(255, 255, 255, .20);
}


/* =========================================================
   BASES
   ========================================================= */

.base-marker {
    position: absolute;

    width: 17px;
    height: 17px;

    background: #ffffff;

    border:
        2px solid
        #d9d9d9;

    transform:
        rotate(45deg);

    z-index: 8;

    box-shadow:
        0 2px 4px
        rgba(0, 0, 0, .45);
}


.second-base-marker {
    top: 39%;
    left: 50%;

    transform:
        translateX(-50%)
        rotate(45deg);
}


.third-base-marker {
    top: 59%;
    left: 31%;
}


.first-base-marker {
    top: 59%;
    right: 31%;
}


/* =========================================================
   HOME PLATE
   ========================================================= */

.home-plate-marker {
    position: absolute;

    left: 50%;
    bottom: 7%;

    width: 25px;
    height: 21px;

    transform:
        translateX(-50%);

    background: white;

    clip-path:
        polygon(
            0 0,
            100% 0,
            100% 58%,
            50% 100%,
            0 58%
        );

    z-index: 8;

    filter:
        drop-shadow(
            0 2px 3px
            rgba(0, 0, 0, .50)
        );
}


/* =========================================================
   PITCHER MOUND
   ========================================================= */

.pitcher-mound {
    position: absolute;

    left: 50%;
    top: 61%;

    width: 48px;
    height: 23px;

    transform:
        translate(
            -50%,
            -50%
        );

    background: #c5a16d;

    border:
        2px solid
        #dac096;

    border-radius: 50%;

    z-index: 6;

    box-shadow:
        0 2px 5px
        rgba(0, 0, 0, .35);
}


/* =========================================================
   POSITION BUTTONS
   ========================================================= */

.field-position-button {
    position: absolute !important;

    width: 64px;
    height: 64px;

    min-width: 64px;

    border-radius: 50% !important;

    padding: 0 !important;

    display: flex !important;

    align-items: center !important;
    justify-content: center !important;

    background:
        linear-gradient(
            180deg,
            #102a3a 0%,
            #07141e 100%
        ) !important;

    border:
        3px solid
        #e1c35b !important;

    color:
        #ffe477 !important;

    font-size:
        16px !important;

    font-weight:
        900 !important;

    z-index: 20;

    box-shadow:
        0 4px 10px
        rgba(0, 0, 0, .50);

    transition:
        transform .15s ease,
        background .15s ease,
        border-color .15s ease,
        box-shadow .15s ease;
}


.field-position-button:hover {
    background:
        linear-gradient(
            180deg,
            #a82438,
            #6f1624
        ) !important;

    border-color:
        #ffe57a !important;

    color:
        #ffffff !important;

    transform:
        scale(1.09);

    box-shadow:
        0 0 17px
        rgba(255, 221, 78, .46);
}


/* =========================================================
   OF
   ========================================================= */

.field-of {
    top: 19%;
    left: 50%;

    transform:
        translateX(-50%);
}


.field-of:hover {
    transform:
        translateX(-50%)
        scale(1.09);
}


/* =========================================================
   SS
   ========================================================= */

.field-ss {
    top: 43%;
    left: 34%;
}


/* =========================================================
   2B
   ========================================================= */

.field-2b {
    top: 43%;
    right: 34%;
}


/* =========================================================
   3B
   ========================================================= */

.field-3b {
    top: 59%;
    left: 17%;
}


/* =========================================================
   1B
   ========================================================= */

.field-1b {
    top: 59%;
    right: 17%;
}


/* =========================================================
   C
   ========================================================= */

.field-c {
    bottom: 2%;
    left: 50%;

    transform:
        translateX(-50%);
}


.field-c:hover {
    transform:
        translateX(-50%)
        scale(1.09);
}


/* =========================================================
   DH
   ========================================================= */

.field-dh {
    right: 4%;
    bottom: 4%;

    background:
        linear-gradient(
            180deg,
            #692532,
            #351117
        ) !important;

    border-color:
        #d9a84e !important;
}


/* =========================================================
   FIELD LEGEND
   ========================================================= */

.field-legend {
    display: grid;

    grid-template-columns:
        repeat(
            4,
            minmax(
                120px,
                1fr
            )
        );

    gap: 7px;

    background: #06110b;

    border:
        1px solid
        #3b5843;

    border-radius: 9px;

    padding: 10px;
}


.field-legend-item {
    display: flex;

    gap: 7px;

    align-items: center;

    color: #9eafa4;

    font-size: 9px;
}


.field-legend-code {
    color: #ffe06b;

    font-weight: 900;

    min-width: 22px;
}


/* =========================================================
   POSITION COMPARISON
   ========================================================= */

.position-comparison-panel {
    background: #071725;

    border:
        3px solid
        #58728c;

    border-radius: 14px;

    padding: 20px;
}


.selected-position-header {
    text-align: center;
}


.selected-position-title {
    color: #ffd84c;

    font-size: 34px;
    font-weight: 900;
}


.selected-position-subtitle {
    color: #9fb0ba;

    font-size: 11px;

    margin-bottom: 15px;
}


.selection-vs-snub-grid {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 18px;
}


.comparison-section-title {
    color: #f3e6a5;

    border-left:
        5px solid
        #c6a95c;

    padding: 10px;

    font-weight: 900;
}


.allmlb-comparison-card {
    background: #08120c;

    border-radius: 10px;

    padding: 13px;

    margin-bottom: 10px;
}


.selected-card {
    border:
        3px solid
        #42c96a;
}


.snub-card {
    border:
        3px solid
        #d0a849;
}


.comparison-card-header {
    display: grid;

    grid-template-columns:
        1fr auto;

    gap: 12px;
}


.comparison-badge {
    color: #e9d77f;

    font-size: 9px;
    font-weight: 900;
}


.comparison-player-name {
    color: #f6efc8;

    font-size: 18px;
    font-weight: 900;
}


.comparison-player-position {
    color: #91a197;

    font-size: 9px;

    margin-top: 2px;
}


.comparison-prob-label {
    color: #9fafb5;

    font-size: 8px;
    font-weight: 900;

    text-align: right;
}


.comparison-prob-value {
    color: #ffd84c;

    font-size: 23px;
    font-weight: 900;

    text-align: right;
}


.comparison-stat-grid {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 6px;

    margin-top: 8px;
}


.comparison-stat {
    background: #050906;

    border:
        1px solid
        #26362b;

    border-radius: 5px;

    padding: 6px;

    text-align: center;
}


.comparison-stat-label {
    color: #87988e;

    font-size: 8px;
    font-weight: 800;
}


.comparison-stat-value {
    color: #f3e7a8;

    font-size: 14px;
    font-weight: 900;
}


.empty-comparison-message {
    color: #91a2aa;

    background: #08131a;

    border:
        1px solid
        #31444d;

    border-radius: 8px;

    padding: 12px;
}

/* =========================================================
   ALL-MLB PLAYER COMPARISON CONTROLS
   ========================================================= */

.allmlb-comparison-controls {
    margin-top: 20px;
    margin-bottom: 8px;

    padding: 16px;

    background:
        linear-gradient(
            135deg,
            #102719,
            #08170f
        );

    border:
        2px solid
        #b69a55;

    border-radius: 11px;
}


.allmlb-comparison-title {
    color: #f5e7a4;

    font-size: 15px;
    font-weight: 900;

    letter-spacing: .5px;
}


.allmlb-comparison-description {
    color: #9caf9f;

    font-size: 10px;
    line-height: 1.4;

    margin-top: 3px;
    margin-bottom: 12px;
}


.allmlb-comparison-controls .form-select {
    background: #f3f4e9 !important;
    color: #111 !important;

    border:
        2px solid
        #c6a95c !important;
}


/* =========================================================
   ALL-MLB MODEL EXPLANATION
   ========================================================= */

.allmlb-model-explanation {
    margin-top: 24px;

    padding: 20px;

    background:
        linear-gradient(
            180deg,
            #0a1923 0%,
            #061018 100%
        );

    border:
        2px solid
        #526f80;

    border-radius:
        13px;

    box-shadow:
        inset 0 0 20px
        rgba(0, 0, 0, .20);
}


/* ---------------------------------------------------------
   HEADER
   --------------------------------------------------------- */

.allmlb-explanation-header {
    text-align: center;

    margin-bottom: 18px;
}


.allmlb-explanation-kicker {
    color: #d0b252;

    font-size: 9px;
    font-weight: 900;

    letter-spacing: 3px;

    margin-bottom: 5px;
}


.allmlb-explanation-title {
    color: #f4e6a5;

    font-size: 22px;
    font-weight: 900;

    letter-spacing: .5px;
}


.allmlb-explanation-subtitle {
    color: #92a5b0;

    font-size: 10px;

    margin-top: 5px;
}


/* ---------------------------------------------------------
   PLAYER COMPARISON
   --------------------------------------------------------- */

.allmlb-explain-player-grid {
    display: grid;

    grid-template-columns:
        1fr
        55px
        1fr;

    gap: 12px;

    align-items: center;

    max-width: 900px;

    margin:
        0 auto
        15px auto;
}


.allmlb-explain-player-card {
    padding: 13px;

    border-radius: 9px;

    text-align: center;

    background: #07110c;
}


.actual-explain-card {
    border:
        2px solid
        #38c96a;
}


.snub-explain-card {
    border:
        2px solid
        #d4aa38;
}


.allmlb-explain-player-label {
    color: #9daeb4;

    font-size: 8px;
    font-weight: 900;

    letter-spacing: 1px;
}


.allmlb-explain-player-name {
    color: #f4f1db;

    font-size: 17px;
    font-weight: 900;

    margin-top: 3px;
}


.allmlb-explain-prob {
    font-size: 28px;
    font-weight: 900;

    margin-top: 2px;
}


.selected-explain-prob {
    color: #59e37c;
}


.snub-explain-prob {
    color: #ffd33f;
}


.allmlb-explain-vs {
    text-align: center;

    color: #c9ae56;

    font-size: 14px;
    font-weight: 900;
}


/* ---------------------------------------------------------
   SUMMARY
   --------------------------------------------------------- */

.allmlb-explanation-summary {
    max-width: 900px;

    margin:
        0 auto
        18px auto;

    padding: 10px 12px;

    background: #101f28;

    border-left:
        4px solid
        #c6a95c;

    color: #c8d3d8;

    font-size: 11px;
    line-height: 1.45;
}


/* ---------------------------------------------------------
   SECTION TITLES
   --------------------------------------------------------- */

.allmlb-explanation-section-title {
    color: #efe0a1;

    font-size: 13px;
    font-weight: 900;

    border-left:
        4px solid
        #c6a95c;

    padding-left: 9px;

    margin:
        18px 0
        7px 0;
}


.allmlb-feature-description {
    color: #899ca6;

    font-size: 9px;

    margin-bottom: 9px;
}


/* ---------------------------------------------------------
   FEATURE CHIPS
   --------------------------------------------------------- */

.allmlb-feature-chip-grid {
    display: flex;

    flex-wrap: wrap;

    gap: 6px;
}


.allmlb-feature-chip {
    display: inline-block;

    padding:
        5px 8px;

    background: #0c1b23;

    border:
        1px solid
        #435c68;

    border-radius: 20px;

    color: #c8d4d9;

    font-size: 8px;
    font-weight: 800;
}


/* ---------------------------------------------------------
   DRIVER TABLE
   --------------------------------------------------------- */

.allmlb-driver-header,
.allmlb-driver-row {
    display: grid;

    grid-template-columns:
        1.1fr
        .8fr
        .8fr
        1.5fr;

    gap: 7px;

    align-items: center;
}


.allmlb-driver-header {
    background: #142832;

    border:
        1px solid
        #334e5b;

    border-radius:
        6px;

    padding: 8px;

    color: #aab9c0;

    font-size: 8px;
    font-weight: 900;

    text-align: center;
}


.allmlb-driver-row {
    background: #07141b;

    border:
        1px solid
        #253f4a;

    border-radius:
        6px;

    padding: 8px;

    margin-top: 5px;
}


.allmlb-driver-feature {
    color: #f0e5ae;

    font-size: 10px;
    font-weight: 900;
}


.allmlb-driver-value {
    color: #d9e1e4;

    font-size: 10px;
    font-weight: 800;

    text-align: center;
}


/* ---------------------------------------------------------
   DRIVER EFFECT
   --------------------------------------------------------- */

.allmlb-driver-effect {
    min-width: 0;
}


.allmlb-driver-bar-track {
    width: 100%;

    height: 8px;

    background: #1c2d35;

    border-radius: 10px;

    overflow: hidden;
}


.allmlb-driver-bar-fill {
    height: 100%;

    border-radius: 10px;
}


.allmlb-driver-selected {
    color: #54df7a;
}


.allmlb-driver-bar-fill.allmlb-driver-selected {
    background:
        linear-gradient(
            90deg,
            #207743,
            #55df7b
        );
}


.allmlb-driver-snub {
    color: #ffd045;
}


.allmlb-driver-bar-fill.allmlb-driver-snub {
    background:
        linear-gradient(
            90deg,
            #96701a,
            #e7b934
        );
}


.allmlb-driver-even {
    color: #a4b0b6;
}


.allmlb-driver-bar-fill.allmlb-driver-even {
    background: #71808a;
}


.allmlb-driver-edge {
    font-size: 8px;
    font-weight: 900;

    margin-top: 3px;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}


/* ---------------------------------------------------------
   EXPLANATION FOOTNOTE
   --------------------------------------------------------- */

.allmlb-explanation-note {
    margin-top: 16px;

    padding: 9px 11px;

    background: #09161d;

    border:
        1px solid
        #344d58;

    border-radius: 7px;

    color: #83959e;

    font-size: 8.5px;

    line-height: 1.5;
}


.allmlb-explanation-unavailable {
    padding: 12px;

    color: #93a4ac;

    background: #09151c;

    border-radius: 7px;
}


/* ---------------------------------------------------------
   RESPONSIVE EXPLANATION
   --------------------------------------------------------- */

@media (max-width: 850px) {

    .allmlb-explain-player-grid {
        grid-template-columns: 1fr;
    }


    .allmlb-explain-vs {
        padding: 2px 0;
    }


    .allmlb-driver-header,
    .allmlb-driver-row {
        grid-template-columns:
            1fr
            .8fr
            .8fr
            1.3fr;
    }


    .allmlb-explanation-title {
        font-size: 18px;
    }
}

/* =========================================================
   ACTUAL ALL-MLB ROSTER
   ========================================================= */

.actual-roster-panel {
    background: #071725;

    border:
        2px solid
        #58728c;

    border-radius: 12px;

    padding: 18px;

    position: sticky;

    top: 15px;
}


.actual-roster-title {
    color: #f6e9a8;

    font-weight: 900;
}


.actual-roster-description {
    color: #9eafb8;

    font-size: 11px;

    line-height: 1.4;

    margin-bottom: 14px;
}


.actual-roster-card {
    background: #08120c;

    border:
        2px solid
        #45564a;

    border-radius: 8px;

    padding: 10px;

    margin-bottom: 10px;
}


.actual-roster-position {
    color: #ffe170;

    font-size: 20px;
    font-weight: 900;

    margin-bottom: 5px;
}


.actual-roster-player-row {
    display: grid;

    grid-template-columns:
        1fr auto;

    gap: 8px;

    padding: 5px 0;

    border-top:
        1px solid
        #33483a;
}


.roster-player-name {
    color: #edf2ed;

    font-weight: 800;
}


.roster-player-prob {
    color: #e7ce72;

    font-size: 10px;
    font-weight: 800;
}


/* =========================================================
   CUSTOM ALL MLB
   ========================================================= */

.custom-allmlb-header,
.custom-postseason-header {
    text-align: center;
}


/* =========================================================
   RESPONSIVE ALL MLB
   ========================================================= */

@media (max-width: 1200px) {

    .actual-roster-panel {
        position: static;
    }
}


@media (max-width: 850px) {

    .baseball-field {
        width: 100%;
    }


    .field-position-button {
        width: 52px;
        height: 52px;

        min-width: 52px;

        font-size:
            13px !important;
    }


    .field-3b {
        left: 11%;
    }


    .field-1b {
        right: 11%;
    }


    .field-legend {
        grid-template-columns:
            repeat(2, 1fr);
    }


    .selection-vs-snub-grid {
        grid-template-columns: 1fr;
    }
}

/* =========================================================
   POSTSEASON
   ========================================================= */

.postseason-format-detail {
    margin-top: 8px;
    color: #cbd5cf;
    font-size: 12px;
}

.postseason-bracket-panel {
    background:
        linear-gradient(
            180deg,
            #0a2130,
            #05131d
        );
    border: 4px solid #b69a55;
    border-radius: 15px;
    padding: 14px;
    margin-top: 12px;
}

.bracket-header {
    text-align: center;
    margin-bottom: 12px;
}

.bracket-kicker {
    color: #d7bd70;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
}

.bracket-main-title {
    color: #ffdc5a;
    font-size: 27px;
    font-weight: 900;
}

.bracket-subtitle {
    color: #b9c8d1;
    font-size: 11px;
}

.bracket-divider {
    height: 2px;
    background:
        linear-gradient(
            90deg,
            transparent,
            #b69a55,
            transparent
        );
    margin: 24px 0;
}



/* =========================================================
   2027 PROJECTED ALL-MLB DIAMOND
   ========================================================= */

.projection-diamond-panel {
    margin-top: 18px;
}

.projection-diamond-kicker {
    color: #f0d464;
    font-size: 11px;
    font-weight: 900;
    letter-spacing: 2px;
    margin-bottom: 5px;
}

.projection-baseball-field {
    width: min(100%, 980px);
    aspect-ratio: 1.30 / 1;
    min-height: 690px;
    overflow: visible;
    margin-top: 22px;
    margin-bottom: 28px;
}

.projection-position-card {
    position: absolute !important;
    z-index: 30 !important;
    width: 210px;
    min-height: 116px;
    padding: 8px 10px !important;
    border-radius: 12px !important;
    border: 2px solid #657783 !important;
    background: linear-gradient(180deg, #0d2230 0%, #06131b 100%) !important;
    color: #f5f7f8 !important;
    box-shadow: 0 6px 18px rgba(0,0,0,.48);
    white-space: normal !important;
    text-align: left !important;
    transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}

.projection-position-card:hover {
    transform: translateY(-3px) scale(1.02);
    border-color: #f0d464 !important;
    box-shadow: 0 0 18px rgba(240,212,100,.42);
}

.projection-position-card-active {
    border-color: #f0d464 !important;
    box-shadow: 0 0 20px rgba(240,212,100,.55);
    background: linear-gradient(180deg, #173044 0%, #091a25 100%) !important;
}

.projection-diamond-button-content {
    width: 100%;
}

.projection-diamond-position-label {
    color: #ffe477;
    font-size: 17px;
    line-height: 1;
    font-weight: 950;
    text-align: center;
    margin-bottom: 6px;
}

.projection-diamond-team-label {
    font-size: 7px;
    font-weight: 950;
    letter-spacing: 1.5px;
    margin-top: 4px;
    margin-bottom: 2px;
}

.projection-first-label {
    color: #f0d464;
}

.projection-second-label {
    color: #64d8a0;
    border-top: 1px solid rgba(100,216,160,.25);
    padding-top: 4px;
}

.projection-diamond-player-list {
    display: grid;
    gap: 1px;
}

.projection-diamond-player-name {
    color: #ffffff;
    font-size: 10px;
    font-weight: 850;
    line-height: 1.12;
    overflow: hidden;
    text-overflow: ellipsis;
}

.projection-card-of {
    width: 420px;
    min-height: 148px;
    top: 11%;
    left: 50%;
    transform: translateX(-50%);
}

.projection-card-of:hover {
    transform: translateX(-50%) translateY(-3px) scale(1.02);
}

.projection-card-ss {
    top: 39%;
    left: 13%;
}

.projection-card-2b {
    top: 39%;
    right: 13%;
}

.projection-card-3b {
    top: 59%;
    left: 1.5%;
}

.projection-card-1b {
    top: 59%;
    right: 1.5%;
}

.projection-card-c {
    width: 245px;
    bottom: -2%;
    left: 50%;
    transform: translateX(-50%);
}

.projection-card-c:hover {
    transform: translateX(-50%) translateY(-3px) scale(1.02);
}

.projection-card-dh {
    width: 190px;
    top: 23%;
    right: 1.5%;
    border-color: #b87758 !important;
}

.projection-diamond-legend {
    display: grid;
    grid-template-columns: auto 1fr auto 1fr;
    gap: 8px 12px;
    align-items: center;
    margin: 0 auto 10px auto;
    max-width: 900px;
    padding: 10px 12px;
    border: 1px solid #36596e;
    background: #07151e;
    border-radius: 9px;
    color: #aab9c1;
    font-size: 9px;
}

.projection-diamond-legend-first,
.projection-diamond-legend-second {
    font-size: 8px;
    font-weight: 950;
    letter-spacing: 1px;
}

.projection-diamond-legend-first {
    color: #f0d464;
}

.projection-diamond-legend-second {
    color: #64d8a0;
}

.projection-selected-position-header {
    display: flex;
    gap: 12px;
    align-items: center;
    flex-wrap: wrap;
    margin: 24px 0 12px 0;
    padding: 12px 14px;
    border: 1px solid #36596e;
    border-radius: 10px;
    background: #07151e;
}

.projection-selected-kicker {
    color: #f0d464;
    font-size: 9px;
    font-weight: 950;
    letter-spacing: 1.5px;
}

.projection-selected-description {
    flex: 1;
    min-width: 250px;
    color: #9dafb8;
    font-size: 9px;
}

@media (max-width: 900px) {
    .projection-baseball-field {
        min-height: 760px;
        aspect-ratio: auto;
        border-radius: 30px;
    }

    .projection-position-card {
        width: 180px;
        min-height: 108px;
    }

    .projection-card-of {
        width: 360px;
        top: 9%;
    }

    .projection-card-ss {
        top: 36%;
        left: 4%;
    }

    .projection-card-2b {
        top: 36%;
        right: 4%;
    }

    .projection-card-3b {
        top: 55%;
        left: 1%;
    }

    .projection-card-1b {
        top: 55%;
        right: 1%;
    }

    .projection-card-dh {
        top: 22%;
        right: 1%;
        width: 170px;
    }

    .projection-card-c {
        bottom: 1%;
        width: 220px;
    }
}

/* =========================================================
   MASTER BRACKET
   ========================================================= */

.complete-bracket-grid {
    display: grid;
    grid-template-columns:
        minmax(0,1fr)
        235px
        minmax(0,1fr);
    gap: 10px;
    align-items: center;
}


/* =========================================================
   LEAGUE BRACKET
   ========================================================= */

.league-bracket {
    background: #04121b;
    border: 2px solid #36596e;
    border-radius: 12px;
    padding: 8px;
    min-width: 0;
}

.league-banner {
    text-align: center;
    font-size: 17px;
    font-weight: 900;
    padding: 7px;
    border-radius: 8px;
    margin-bottom: 7px;
}

.al-banner {
    background: #1d4f78;
    border: 2px solid #5396c7;
}

.nl-banner {
    background: #70252f;
    border: 2px solid #c85f6d;
}

.league-round-grid {
    display: grid;
    grid-template-columns:
        1.05fr
        1.12fr
        1.12fr
        1.05fr
        .9fr;
    gap: 5px;
    align-items: center;
}

.bracket-round-column {
    min-width: 0;
}

.bracket-round-title {
    color: #f0d464;
    font-size: 8px;
    font-weight: 900;
    min-height: 25px;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    border-bottom: 1px solid #304b5a;
    margin-bottom: 5px;
}

.bracket-round-stack {
    display: grid;
    gap: 8px;
}

.seed-card-stack {
    display: grid;
    gap: 4px;
}

.no-round {
    color: #718490;
    text-align: center;
    font-size: 8px;
}


/* =========================================================
   BRACKET TEAM
   ========================================================= */

.bracket-team-card {
    background: #0b1b25;
    border: 1px solid #426276;
    border-radius: 7px;
    padding: 4px;
    min-width: 0;
}

.bracket-team-winner {
    border: 2px solid #d4b246;
    background: #292309;
}

.bracket-team-main {
    display: grid;
    grid-template-columns:
        20px
        32px
        minmax(0,1fr);
    gap: 3px;
    align-items: center;
}

.bracket-seed {
    color: #f2d45c;
    font-size: 9px;
    font-weight: 900;
    text-align: center;
}

.bracket-team-name {
    color: #f2f4f4;
    font-size: 8px;
    font-weight: 900;
    line-height: 1.05;
}

.bracket-team-record {
    color: #8fa2ad;
    font-size: 6.5px;
}

.bracket-team-footer {
    border-top: 1px solid #294655;
    margin-top: 3px;
    padding-top: 2px;
    text-align: right;
    font-size: 8px;
    font-weight: 900;
}

.projection-footer {
    color: #ffdc4e;
}

.actual-footer {
    color: #b9cad3;
    font-size: 6.5px;
}

.bracket-matchup {
    background: #06151e;
    border: 1px solid #294957;
    border-radius: 8px;
    padding: 4px;
}

.clickable-matchup {
    cursor: pointer;
    transition: .12s ease;
}

.clickable-matchup:hover {
    transform: translateY(-2px);
    border-color: #e0c75d;
    box-shadow:
        0 0 8px
        rgba(224,199,93,.35);
}

.bracket-vs {
    color: #d2b754;
    text-align: center;
    font-size: 7px;
    font-weight: 900;
}

.matchup-click-hint {
    color: #748c99;
    font-size: 6px;
    text-align: center;
}

.team-logo {
    object-fit: contain;
}

.team-logo-fallback {
    display: flex;
    align-items: center;
    justify-content: center;
    background: #142734;
    border-radius: 50%;
    font-size: 7px;
}

.bracket-team-card .team-logo {
    width: 32px !important;
    height: 32px !important;
}


/* =========================================================
   WORLD SERIES
   ========================================================= */

.world-series-panel {
    background: #060d12;
    border: 2px solid #c3a143;
    border-radius: 12px;
    padding: 8px;
}

.world-series-title {
    color: #ffda59;
    font-size: 14px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 7px;
}

.world-series-matchup {
    border: 1px solid #405765;
    border-radius: 8px;
    padding: 4px;
}

.world-series-vs {
    color: #f5d55a;
    font-size: 17px;
    font-weight: 900;
    text-align: center;
}

.champion-box {
    background:
        linear-gradient(
            135deg,
            #423604,
            #181305
        );
    border: 2px solid #ddbd4c;
    border-radius: 8px;
    margin-top: 8px;
    padding: 8px;
    text-align: center;
}

.champion-label {
    color: #eed574;
    font-size: 7px;
    font-weight: 900;
}

.champion-name {
    color: #fff0a1;
    font-size: 13px;
    font-weight: 900;
}

.champion-probability {
    color: #ffdc4e;
    font-size: 10px;
    font-weight: 900;
}


/* =========================================================
   MATCHUP ANALYSIS
   ========================================================= */

.matchup-analysis-panel {
    background:
        linear-gradient(
            180deg,
            #081823,
            #050e14
        );
    border: 4px solid #b69a55;
    border-radius: 15px;
    padding: 20px;
    margin-top: 18px;
}

.analysis-kicker {
    color: #c5ab58;
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 2px;
    text-align: center;
}

.analysis-title {
    color: #ffe16d;
    font-size: 25px;
    font-weight: 900;
    text-align: center;
}

.analysis-subtitle {
    text-align: center;
    color: #cdd9df;
}

.analysis-note {
    background: #101d25;
    color: #aebec6;
    border-left: 4px solid #708d9e;
    max-width: 900px;
    margin: 10px auto;
    padding: 8px;
    font-size: 11px;
}

.analysis-team-grid {
    display: grid;
    grid-template-columns:
        1fr
        80px
        1fr;
    gap: 12px;
    align-items: center;
}

.analysis-team-card {
    background: #081119;
    border: 2px solid #405c6c;
    border-radius: 12px;
    text-align: center;
    padding: 14px;
}

.analysis-team-card .team-logo {
    width: 68px !important;
    height: 68px !important;
}

.analysis-team-name {
    color: #f1f5f5;
    font-size: 20px;
    font-weight: 900;
}

.analysis-team-record {
    color: #93a8b3;
}

.analysis-team-prob {
    color: #ffe05e;
    font-size: 34px;
    font-weight: 900;
}

.analysis-prob-label {
    color: #a6b7bf;
    font-size: 9px;
}

.analysis-vs {
    text-align: center;
    color: #d7bc5a;
    font-size: 26px;
    font-weight: 900;
}

.projected-pick-box,
.result-card {
    background: #102618;
    border: 2px solid #3d8955;
    border-radius: 9px;
    padding: 12px;
    margin-top: 14px;
    text-align: center;
}

.result-grid {
    display: grid;
    grid-template-columns:
        1fr 1fr;
    gap: 12px;
}

.result-label {
    color: #a7b7bf;
    font-size: 9px;
    font-weight: 900;
}

.model-pick {
    color: #ffdb59;
    font-size: 19px;
    font-weight: 900;
}

.actual-pick {
    color: #67e98b;
    font-size: 19px;
    font-weight: 900;
}

.analysis-section-title {
    color: #f0df99;
    font-weight: 900;
    border-left: 5px solid #c6a95c;
    padding-left: 9px;
    margin: 18px 0 8px;
}

.analysis-feature-header,
.analysis-feature-row {
    display: grid;
    grid-template-columns:
        1.4fr
        .8fr
        .8fr
        .8fr;
    gap: 5px;
    padding: 7px;
}

.analysis-feature-header {
    background: #142630;
    font-size: 9px;
    font-weight: 900;
    text-align: center;
}

.analysis-feature-row {
    background: #07131a;
    border: 1px solid #273f4b;
    font-size: 11px;
}

.analysis-feature-row > div:not(:first-child) {
    text-align: center;
}

.model-edge {
    color: #87efa3;
    font-weight: 900;
}

.edge-bar-row {
    display: grid;
    grid-template-columns:
        170px
        1fr
        60px;
    gap: 10px;
    align-items: center;
    margin-bottom: 7px;
}

.edge-bar-label {
    font-size: 11px;
}

.edge-bar-track {
    height: 14px;
    background: #192933;
    border-radius: 8px;
    overflow: hidden;
}

.edge-bar-fill {
    height: 100%;
    background:
        linear-gradient(
            90deg,
            #37764e,
            #d0b04e
        );
}

.edge-bar-team {
    color: #ffe06b;
    font-weight: 900;
    text-align: center;
}


/* =========================================================
   RANKINGS
   ========================================================= */

.ranking-panel {
    background: #07151f;
    border: 4px solid #b69a55;
    border-radius: 13px;
    padding: 20px;
}

.ranking-title {
    color: #ffe170;
    font-size: 25px;
    font-weight: 900;
    margin-bottom: 15px;
}

.ranking-row {
    display: grid;
    grid-template-columns:
        50px
        minmax(260px,1.4fr)
        minmax(180px,2fr)
        90px;
    gap: 12px;
    align-items: center;
    background: #071117;
    border: 1px solid #3f5661;
    border-radius: 8px;
    padding: 9px;
    margin-bottom: 6px;
}

.ranking-number {
    color: #d7bd70;
    font-size: 20px;
    font-weight: 900;
    text-align: center;
}

.ranking-team-block {
    display: flex;
    align-items: center;
    gap: 10px;
}

.ranking-team-name {
    font-weight: 900;
}

.ranking-team-sub {
    color: #8799a1;
    font-size: 10px;
}

.ranking-bar-track {
    height: 14px;
    background: #19272e;
    border-radius: 10px;
    overflow: hidden;
}

.ranking-bar-fill {
    height: 100%;
    background:
        linear-gradient(
            90deg,
            #37764e,
            #d2ad4c
        );
}

.ranking-probability {
    color: #ffd84c;
    font-size: 18px;
    font-weight: 900;
    text-align: right;
}


/* =========================================================
   CUSTOM POSTSEASON
   ========================================================= */

.postseason-ladder {
    display: grid;
    gap: 10px;
}

.postseason-probability-card {
    background: #060c11;
    border: 2px solid #41586a;
    border-radius: 10px;
    padding: 12px;
}

.postseason-step-label {
    font-weight: 900;
}

.postseason-step-prob {
    color: #ffd84c;
    font-size: 22px;
    font-weight: 900;
}

.postseason-progress-track {
    height: 16px;
    background: #16232b;
    border-radius: 20px;
    overflow: hidden;
    margin-top: 5px;
}

.postseason-progress-fill {
    height: 100%;
    background:
        linear-gradient(
            90deg,
            #2c7747,
            #d0ad49
        );
}


/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 1250px) {

    .complete-bracket-grid {
        grid-template-columns: 1fr;
    }

    .world-series-panel {
        width: min(100%,430px);
        margin: 10px auto;
    }
}


@media (max-width: 850px) {

    .score-grid {
        grid-template-columns:
            repeat(2,1fr);
    }

    .big-score-grid {
        grid-template-columns: 1fr;
    }

    .instruction-grid {
        grid-template-columns: 1fr;
    }

    .selection-vs-snub-grid {
        grid-template-columns: 1fr;
    }

    .analysis-team-grid {
        grid-template-columns: 1fr;
    }

    .result-grid {
        grid-template-columns: 1fr;
    }
}

"""


# ============================================================
# APP UI
# ============================================================

app_ui = ui.page_fluid(

    ui.tags.style(
        scoreboard_css
    ),

    ui.div(

        ui.h1(
            "MLB PREDICTION LAB",
            class_="app-shell-title"
        ),

        ui.p(
            (
                "Hall of Fame modeling • "
                "All-MLB modeling • "
                "Postseason advancement modeling"
            ),
            class_="app-shell-subtitle"
        ),

        class_="app-shell-header"
    ),

    ui.navset_tab(

        ui.nav_panel(

            "Hall of Fame Predictor",

            hof_ui(
                "hof"
            ),
        ),

        ui.nav_panel(

            "All-MLB Predictor",

            allmlb_ui(
                "allmlb"
            ),
        ),

        ui.nav_panel(

            "Postseason Advancement",

            postseason_ui(
                "postseason"
            ),
        ),

        id="main_module_tab",
    ),
)


# ============================================================
# MAIN SERVER
# ============================================================

def server(
    input,
    output,
    session
):

    hof_server(
        "hof"
    )

    allmlb_server(
        "allmlb"
    )

    postseason_server(
        "postseason"
    )

# ============================================================
# APP
# ============================================================

app = App(
    app_ui,
    server
)

# /usr/local/bin/python3.12 -m shiny run --reload /Users/brettgreenberg/Desktop/STA3100_Shiny.py