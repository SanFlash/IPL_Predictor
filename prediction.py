import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and prepare the data
deliveries_data = pd.read_csv('ipl__deliveries.csv')
team_players_data = pd.read_csv('ipl_2025_team_players.csv')

# Function to calculate player performance at a specific venue
def calculate_performance(team, venue):
    # Filter deliveries data for the team and venue
    team_data = deliveries_data[(deliveries_data['batting_team'] == team) | (deliveries_data['bowling_team'] == team)]
    team_data = team_data[team_data['venue'] == venue]
    
    # Calculate performance metrics (e.g., runs scored by batsmen, wickets taken by bowlers)
    batsmen_performance = team_data.groupby('striker')['runs_of_bat'].sum().reset_index(name='runs_scored')
    bowlers_performance = team_data.groupby('bowler')['player_dismissed'].count().reset_index(name='wickets_taken')
    
    # Merge the performance data for batsmen and bowlers
    performance_data = pd.merge(batsmen_performance, bowlers_performance, left_on='striker', right_on='bowler', how='outer')
    performance_data = performance_data.fillna(0)  # Handle missing values (no dismissals or runs)

    # Calculate aggregate performance metrics
    total_runs = performance_data['runs_scored'].sum()
    total_wickets = performance_data['wickets_taken'].sum()

    return total_runs, total_wickets

# Function to prepare features for ML model
def prepare_features_for_winner_prediction():
    teams = team_players_data['Team Name'].unique()
    features = []
    labels = []

    # For each team, calculate their performance at each venue
    for team in teams:
        for venue in deliveries_data['venue'].unique():
            runs, wickets = calculate_performance(team, venue)
            features.append([runs, wickets])
            # Assume "team1" wins when they bat first, label as 1 for team1 win
            labels.append(1 if team == 'team1' else 0)

    features_df = pd.DataFrame(features, columns=['total_runs', 'total_wickets'])
    return features_df, labels

# Train the winning team prediction model
def train_winner_model():
    X, y = prepare_features_for_winner_prediction()
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# Load the trained model for prediction
winner_model = train_winner_model()

# Function to predict the winner
def predict_winner(team1, team2, venue):
    team1_runs, team1_wickets = calculate_performance(team1, venue)
    team2_runs, team2_wickets = calculate_performance(team2, venue)

    # Prepare the input for prediction
    input_data = pd.DataFrame([[team1_runs, team1_wickets], [team2_runs, team2_wickets]], columns=['total_runs', 'total_wickets'])
    
    # Predict the winner
    predictions = winner_model.predict(input_data)
    
    # Return the team with the higher probability of winning
    if predictions[0] == 1:
        return team1
    else:
        return team2
