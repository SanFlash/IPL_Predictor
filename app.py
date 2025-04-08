from flask import Flask, render_template, request, jsonify
from model import IPLPredictor
import pandas as pd

# Load player and delivery data
player_df = pd.read_csv("ipl_2025_team_players.csv")
deliveries_df = pd.read_csv("ipl__deliveries.csv")
predictor = IPLPredictor(deliveries_df, player_df)



app = Flask(__name__)

@app.route("/")
def index():
    return render_template("idex1.html")

@app.route("/get_players")
def get_players():
    team1 = request.args.get("team1")
    team2 = request.args.get("team2")

    team1_players = player_df[player_df["Team Name"] == team1]["Player Name"].tolist()
    team2_players = player_df[player_df["Team Name"] == team2]["Player Name"].tolist()

    return jsonify({team1: team1_players, team2: team2_players})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    selected_players = data.get("players", [])
    venue = data.get("venue", "")

    selected_players = [p.strip() for p in selected_players]
    df_selected = player_df[player_df["Player Name"].isin(selected_players)]

    stats = []

    for player in selected_players:
        category = df_selected[df_selected["Player Name"] == player]["Category"].values[0]

        # Batting stats
        batting_df = deliveries_df[deliveries_df['striker'] == player]
        venue_batting_df = batting_df[batting_df['venue'] == venue]
        batting_df = venue_batting_df if not venue_batting_df.empty else batting_df

        runs = batting_df['runs_of_bat'].sum()
        balls_faced = batting_df.shape[0]
        dismissals = deliveries_df[(deliveries_df['striker'] == player) & (deliveries_df['player_dismissed'] == player)].shape[0]
        fours = batting_df[batting_df['runs_of_bat'] == 4].shape[0]
        sixes = batting_df[batting_df['runs_of_bat'] == 6].shape[0]
        batting_avg = (runs / dismissals) if dismissals > 0 else runs
        strike_rate = (runs / balls_faced * 100) if balls_faced > 0 else 0

        # Bowling stats
        bowling_df = deliveries_df[deliveries_df['bowler'] == player]
        venue_bowling_df = bowling_df[bowling_df['venue'] == venue]
        bowling_df = venue_bowling_df if not venue_bowling_df.empty else bowling_df

        wickets = bowling_df['player_dismissed'].notnull().sum()
        balls_bowled = bowling_df.shape[0]
        runs_conceded = bowling_df['runs_of_bat'].sum()
        overs_bowled = balls_bowled / 6 if balls_bowled > 0 else 0
        bowling_avg = (runs_conceded / wickets) if wickets > 0 else runs_conceded
        economy = (runs_conceded / overs_bowled) if overs_bowled > 0 else 0

        # Scoring system based on averages
        if category == "Batsman" or category == "Wicket-Keeper":
            score = (batting_avg * 0.7) + (strike_rate * 0.3)
        elif category == "Bowler":
            score = 100 - (bowling_avg * 0.6 + economy * 0.4)
        elif category == "All-Rounder":
            bat_score = (batting_avg * 0.5) + (strike_rate * 0.2)
            bowl_score = 100 - (bowling_avg * 0.5 + economy * 0.3)
            score = bat_score + bowl_score
        else:
            score = 0

        stats.append({
            "Player": player,
            "Category": category,
            "Runs": runs,
            "Balls Faced": balls_faced,
            "Strike Rate": round(strike_rate, 2),
            "Batting Avg": round(batting_avg, 2),
            "Fours": fours,
            "Sixes": sixes,
            "Wickets": wickets,
            "Balls Bowled": balls_bowled,
            "Runs Conceded": runs_conceded,
            "Economy": round(economy, 2),
            "Bowling Avg": round(bowling_avg, 2) if wickets > 0 else "-",
            "Score": round(score, 2)
        })

    stats_df = pd.DataFrame(stats)

    # Role-based limits
    role_limits = {
        "Wicket-Keeper": (1, 3),
        "Batsman": (3, 5),
        "All-Rounder": (1, 3),
        "Bowler": (3, 5)
    }

    final_team = pd.DataFrame()
    remaining_df = stats_df.copy()

    for role, (min_limit, _) in role_limits.items():
        role_df = stats_df[stats_df["Category"] == role].sort_values("Score", ascending=False).head(min_limit)
        final_team = pd.concat([final_team, role_df])
        remaining_df = remaining_df[~remaining_df["Player"].isin(role_df["Player"])]

    remaining_slots = 12 - final_team.shape[0]
    remaining_best = remaining_df.sort_values("Score", ascending=False).head(remaining_slots)
    final_team = pd.concat([final_team, remaining_best])

    remaining_pool = stats_df[~stats_df["Player"].isin(final_team["Player"])]
    backup_df = remaining_pool.sort_values("Score", ascending=False).head(5)

    # Compute average total runs at the venue
    total_runs_venue = deliveries_df[deliveries_df["venue"] == venue].groupby("match_id")["runs_of_bat"].sum()
    target_score = int(total_runs_venue.mean()) if not total_runs_venue.empty else 170

    return jsonify({
        "players": final_team["Player"].tolist(),
        "scores": final_team["Score"].tolist(),
        "target_score": target_score,
        "backup": backup_df["Player"].tolist()
    })



@app.route("/visual", methods=["POST", "GET"])
def visual():
    if request.method == "POST":
        data = request.json
        selected_players = data.get("selected_players", [])
        stats = predictor.get_player_stats_for_visual(selected_players)
        return jsonify({"stats": stats})
    return render_template("visual.html")




@app.route('/predict_winner', methods=['POST'])
def predict_winner():
    data = request.get_json()
    team1 = data.get('team1')
    team2 = data.get('team2')
    venue = data.get('venue')
    selected_players = data.get('selected_players')

    if not team1 or not team2 or not venue or not selected_players:
        return jsonify({"error": "Missing team or player selection."})

    # Load CSVs
    deliveries_df = pd.read_csv('ipl__deliveries.csv')
    players_df = pd.read_csv('ipl_2025_team_players.csv')

    predictor = IPLPredictor(deliveries_df, players_df)
    stats_df = predictor.calculate_player_stats(selected_players)

    team1_score = stats_df[stats_df['Player'].isin(players_df[players_df['Team Name'] == team1]['Player Name'])]['Score'].sum()
    team2_score = stats_df[stats_df['Player'].isin(players_df[players_df['Team Name'] == team2]['Player Name'])]['Score'].sum()

    if team1_score > team2_score:
        winner = team1
    elif team2_score > team1_score:
        winner = team2
    else:
        winner = "Tie"

    return jsonify({"predicted_winner": winner})

@app.route('/predict_target_score', methods=['POST'])
def predict_target_score():
    data = request.get_json()
    selected_players = data.get('selected_players')
    venue = data.get('venue')

    if not selected_players or not venue:
        return jsonify({"error": "Missing player or venue selection."})

    deliveries_df = pd.read_csv("ipl__deliveries.csv")
    players_df = pd.read_csv("ipl_2025_team_players.csv")

    predictor = IPLPredictor(deliveries_df, players_df)
    stats_df = predictor.calculate_player_stats(selected_players)

    # Venue-based player data
    venue_data = deliveries_df[(deliveries_df['venue'] == venue) & (deliveries_df['striker'].isin(selected_players))]

    if venue_data.empty:
        return jsonify({"target_score_range": "170–185"})  # fallback default

    # Basic venue run stats
    total_runs = venue_data['runs_of_bat'].sum()
    total_balls = venue_data.shape[0]
    runs_per_ball = total_runs / total_balls if total_balls > 0 else 1.2

    # Predict performance from player stats
    total_score = stats_df['Score'].sum()
    batting_players = stats_df[stats_df['Category'].isin(['Batsman', 'All-Rounder', 'Wicket-Keeper'])]

    total_batting_score = batting_players['Score'].sum()
    avg_batting_score = total_batting_score / len(batting_players) if not batting_players.empty else 30

    # Historical range from deliveries
    venue_scores = deliveries_df[deliveries_df['venue'] == venue].groupby('match_id')['runs_of_bat'].sum()
    historical_avg = venue_scores.mean()
    historical_std = venue_scores.std()

    # Predict target score using combined intelligence
    base_target = (runs_per_ball * 120) + (avg_batting_score * 0.5)
    lower = max(130, int(base_target - (historical_std * 0.25)))
    upper = min(240, int(base_target + (historical_std * 0.25)))

    return jsonify({
        "target_score_range": f"{lower}–{upper}"
    })


if __name__ == "__main__":
    app.run(debug=True)
