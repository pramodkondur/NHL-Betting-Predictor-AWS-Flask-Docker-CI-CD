import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def clean_data(self, df):
        """Clean the data by handling missing values, duplicates, etc."""

        # Select required columns
        df = df[['awayTeamCode','game_id','homeTeamCode','homeTeamWon','isPlayoffGame','season']]
        
        # Dropping duplicate game_id and season as it has mulitple rows with different events
        df = df.drop_duplicates(subset=['game_id', 'season']).reset_index(drop=True)

        duplicate_mapping = {
        'S.J': 'SJS',   # San Jose Sharks
        'N.J': 'NJD',   # New Jersey Devils
        'T.B': 'TBL',   # Tampa Bay Lightning
        'L.A': 'LAK'    # Los Angeles Kings
        }

        # Replace duplicates in 'awayTeamCode' and 'homeTeamCode' columns
        df['awayTeamCode'] = df['awayTeamCode'].replace(duplicate_mapping)
        df['homeTeamCode'] = df['homeTeamCode'].replace(duplicate_mapping)
        # Replace ATL
        df['homeTeamCode'] = df['homeTeamCode'].replace('ATL', 'WPG')
        df['awayTeamCode'] = df['awayTeamCode'].replace('ATL', 'WPG')
        
        return df

    def add_firstfeatures(self, df):
        """Add additional features to the dataframe."""
        # Sort the DataFrame by season and game_id
        df = df.sort_values(by=['season', 'game_id']).reset_index(drop=True)

        # Initialize columns for total games, wins, and losses
        df['total_games_played_by_home'] = 0
        df['total_games_played_by_away'] = 0
        df['total_wins_home'] = 0
        df['total_losses_home'] = 0
        df['total_wins_away'] = 0
        df['total_losses_away'] = 0

        # Use dictionaries to store cumulative counts
        team_games_count = {}
        team_wins_count = {}

        # Variable to keep track of the last season processed
        last_season = None

        # Iterate through each row to calculate totals
        for index, row in df.iterrows():
            # Get team codes and season
            home_team = row['homeTeamCode']
            away_team = row['awayTeamCode']
            season = row['season']

            # Reset the dictionaries if the season has changed
            if last_season != season:
                team_games_count = {}
                team_wins_count = {}
                last_season = season

            # Initialize if not present for home team
            if home_team not in team_games_count:
                team_games_count[home_team] = {'total': 0, 'wins': 0}
            if away_team not in team_games_count:
                team_games_count[away_team] = {'total': 0, 'wins': 0}

            # Update current row with previous totals
            df.at[index, 'total_games_played_by_home'] = team_games_count[home_team]['total']
            df.at[index, 'total_games_played_by_away'] = team_games_count[away_team]['total']
            df.at[index, 'total_wins_home'] = team_wins_count.get(home_team, 0)
            df.at[index, 'total_losses_home'] = df.at[index, 'total_games_played_by_home'] - df.at[index, 'total_wins_home']
            df.at[index, 'total_wins_away'] = team_wins_count.get(away_team, 0)
            df.at[index, 'total_losses_away'] = df.at[index, 'total_games_played_by_away'] - df.at[index, 'total_wins_away']

            # Update counts for the home team
            team_games_count[home_team]['total'] += 1
            if row['homeTeamWon'] == 1:
                team_wins_count[home_team] = team_wins_count.get(home_team, 0) + 1

            # Update counts for the away team
            team_games_count[away_team]['total'] += 1
            if row['homeTeamWon'] == 0:
                team_wins_count[away_team] = team_wins_count.get(away_team, 0) + 1
            
        return df

    def calculate_last_10_games_wins(self, df, team_code, game_id, season, max_n=10):
        # Filter for all previous games in the current season
        previous_games = df[(df['season'] == season) & (df['game_id'] < game_id)]
        
        # Filter games where the team is either home or away
        team_games = previous_games[(previous_games['homeTeamCode'] == team_code) | 
                                    (previous_games['awayTeamCode'] == team_code)]
        
        # Sort by game_id to ensure proper game order
        team_games = team_games.sort_values(by='game_id')
        
        # Select the last n games (maximum max_n)
        last_n_games = team_games.tail(max_n)
        
        # If there are no games available, return 0
        if last_n_games.empty:
            return 0
        
        # Calculate the total wins in the last n games
        total_wins = 0
        for _, game in last_n_games.iterrows():
            if game['homeTeamCode'] == team_code:
                total_wins += game['homeTeamWon']  # Home win is 1 if they won
            else:
                total_wins += (1 - game['homeTeamWon'])  # Away win is 1 if home team lost
        
        return total_wins

    # Apply this function for every row to calculate cumulative total wins for both home and away teams
    def add_last_10_games_win_columns(self, df, max_n=10):
        df['last_10_games_win_home'] = 0
        df['last_10_games_win_away'] = 0
        
        for idx, row in df.iterrows():
            game_id = row['game_id']
            season = row['season']
            home_team = row['homeTeamCode']
            away_team = row['awayTeamCode']
            
            # Calculate cumulative wins for home team
            cumulative_wins_home = self.calculate_last_10_games_wins(df, home_team, game_id, season, max_n)
            
            # Calculate cumulative wins for away team
            cumulative_wins_away = self.calculate_last_10_games_wins(df, away_team, game_id, season, max_n)
            
            # Assign cumulative wins to the new columns
            df.at[idx, 'last_10_games_win_home'] = cumulative_wins_home
            df.at[idx, 'last_10_games_win_away'] = cumulative_wins_away
        
        return df

    # Helper function to get last meeting result till the current row (with season check)
    def get_last_meeting_result(self, team1, team2, current_game_id, current_season):
        last_meeting = self.df[((self.df['homeTeamCode'] == team1) & (self.df['awayTeamCode'] == team2)) |
                        ((self.df['homeTeamCode'] == team2) & (self.df['awayTeamCode'] == team1)) &
                        (self.df['game_id'] < current_game_id) &
                        (self.df['season'] == current_season)]  # Ensure same season and before the current game
        last_meeting = last_meeting.tail(1)
        
        return last_meeting['homeTeamWon'].values[0] if not last_meeting.empty else None

    # Helper function to get last game result (home or away) till the current row (with season check)
    def get_last_game_result(self, team_code, current_game_id, current_season):
        last_game = self.df[((self.df['homeTeamCode'] == team_code) | (df['awayTeamCode'] == team_code)) &
                    (self.df['game_id'] < current_game_id) &
                    (self.df['season'] == current_season)]  # Ensure same season and before the current game
        last_game = last_game.tail(1)
        
        if last_game.empty:
            return None
        return last_game['homeTeamWon'].values[0] if last_game['homeTeamCode'].values[0] == team_code else 1 - last_game['homeTeamWon'].values[0]

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook/data/shots_2007-2023.csv')
            logging.info('Read the dataset as dataframe')

            # Clean the data
            df = self.clean_data(df)
            logging.info('Data cleaning completed')

            # Add additional features
            df = self.add_firstfeatures(df)
            logging.info('First Feature engineering completed')

            df = self.add_last_10_games_win_columns(df)
            logging.info('Last 10 games wins column added')

            df['last_meeting_result'] = df.apply(lambda row: self.get_last_meeting_result(row['homeTeamCode'], row['awayTeamCode'], row['game_id'], row['season']), axis=1)
            df['last_game_result_home'] = df.apply(lambda row: self.get_last_game_result(row['homeTeamCode'], row['game_id'], row['season']), axis=1)
            df['last_game_result_away'] = df.apply(lambda row: self.get_last_game_result(row['awayTeamCode'], row['game_id'], row['season']), axis=1)
            logging.info('Last meeting and games result added')


            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return(
                self.ingestion_config.raw_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    #obj=DataIngestion()
    #raw_data=obj.initiate_data_ingestion()

    raw_data = ('artifacts/data.csv')
    
    data_transformation=DataTransformation()
    
    #data_transformation.initiate_data_transformation(train_data,test_data)
    
    X_train_transformed, X_test_transformed, y_train, y_test =data_transformation.initiate_data_transformation(raw_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train_transformed, X_test_transformed, y_train, y_test))



