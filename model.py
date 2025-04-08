import pandas as pd

class IPLPredictor:
    def __init__(self, deliveries_df, player_df):
        self.deliveries_df = deliveries_df
        self.player_df = player_df

    def calculate_player_stats(self, selected_players):
        df_selected = self.player_df[self.player_df["Player Name"].isin(selected_players)]
        stats = []

        for player in selected_players:
            player_row = df_selected[df_selected["Player Name"] == player]
            if player_row.empty:
                continue

            category = player_row["Category"].values[0]
            player_stats = {
                "Player": player,
                "Category": category,
            }

            batting_df = self.deliveries_df[self.deliveries_df['striker'] == player]
            player_stats["Runs"] = batting_df['runs_of_bat'].sum()
            player_stats["Balls Faced"] = batting_df.shape[0]
            player_stats["Strike Rate"] = (player_stats["Runs"] / player_stats["Balls Faced"] * 100) if player_stats["Balls Faced"] > 0 else 0
            player_stats["Fours"] = batting_df[batting_df['runs_of_bat'] == 4].shape[0]
            player_stats["Sixes"] = batting_df[batting_df['runs_of_bat'] == 6].shape[0]

            bowling_df = self.deliveries_df[self.deliveries_df['bowler'] == player]
            player_stats["Wickets"] = bowling_df[bowling_df['player_dismissed'].notnull()].shape[0]
            player_stats["Balls Bowled"] = bowling_df.shape[0]
            player_stats["Runs Conceded"] = bowling_df['runs_of_bat'].sum()
            player_stats["Economy"] = (player_stats["Runs Conceded"] / (player_stats["Balls Bowled"] / 6)) if player_stats["Balls Bowled"] > 0 else 0

            score = 0
            if category == "Batsman":
                score = player_stats["Runs"] + 2 * player_stats["Fours"] + 3 * player_stats["Sixes"]
            elif category == "Bowler":
                score = 25 * player_stats["Wickets"] - 2 * player_stats["Economy"]
            elif category == "All-Rounder":
                score = player_stats["Runs"] + 20 * player_stats["Wickets"]
            elif category == "Wicket-Keeper":
                score = player_stats["Runs"] + 5 * player_stats["Fours"]

            player_stats["Score"] = score
            stats.append(player_stats)

        return pd.DataFrame(stats)

    def get_player_stats_for_visual(self, selected_players):
        return self.calculate_player_stats(selected_players).to_dict(orient="records")
    
    def predict_winner_based_on_selected_players(self, team1_name, team2_name, selected_players_team1, selected_players_team2, venue):
        # Step 1: Calculate stats for selected players
        team1_stats = self.calculate_player_stats(selected_players_team1)
        team2_stats = self.calculate_player_stats(selected_players_team2)

        # Step 2: Sum scores of all selected players from both teams
        team1_score = team1_stats["Score"].sum()
        team2_score = team2_stats["Score"].sum()

        # Step 3: Adjust for venue impact
        venue_df = self.deliveries_df[self.deliveries_df['venue'] == venue]
        venue_boost = venue_df.groupby('batting_team')['runs_of_bat'].mean().reset_index()

        team1_venue_bonus = venue_boost[venue_boost['batting_team'] == team1_name]['runs_of_bat'].mean() if not venue_boost[venue_boost['batting_team'] == team1_name].empty else 0
        team2_venue_bonus = venue_boost[venue_boost['batting_team'] == team2_name]['runs_of_bat'].mean() if not venue_boost[venue_boost['batting_team'] == team2_name].empty else 0

        team1_score += team1_venue_bonus
        team2_score += team2_venue_bonus

        # Step 4: Determine winner
        winner = team1_name if team1_score > team2_score else team2_name
        margin = abs(team1_score - team2_score)

        return {
            "winner": winner,
            "team1_score": round(team1_score, 2),
            "team2_score": round(team2_score, 2),
            "margin": round(margin, 2)
        }
    
