import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifacts/model.pkl'
            preprocessor_path='artifacts/proprocessor.pkl'
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_transformed=preprocessor.transform(features)
            preds=model.predict(data_transformed)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                homeTeamCode: str,
                 awayTeamCode: str,
                 season: int,
                 isPlayoffGame: bool,
                 total_games_played_by_home: int,
                 total_games_played_by_away: int,
                 total_wins_home: int,
                 total_losses_home: int,
                 total_wins_away: int,
                 total_losses_away: int,
                 last_10_games_win_home: int,
                 last_10_games_win_away: int,
                 last_meeting_result: int,
                 last_game_result_home: float,
                 last_game_result_away: float):
        
                self.homeTeamCode = homeTeamCode
                self.awayTeamCode = awayTeamCode
                self.season = season
                self.isPlayoffGame = isPlayoffGame
                self.total_games_played_by_home = total_games_played_by_home
                self.total_games_played_by_away = total_games_played_by_away
                self.total_wins_home = total_wins_home
                self.total_losses_home = total_losses_home
                self.total_wins_away = total_wins_away
                self.total_losses_away = total_losses_away
                self.last_10_games_win_home = last_10_games_win_home
                self.last_10_games_win_away = last_10_games_win_away
                self.last_meeting_result = last_meeting_result
                self.last_game_result_home = last_game_result_home
                self.last_game_result_away = last_game_result_away

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "homeTeamCode": [self.homeTeamCode],
                "awayTeamCode": [self.awayTeamCode],
                "season": [self.season],
                "isPlayoffGame": [self.isPlayoffGame],
                "total_games_played_by_home": [self.total_games_played_by_home],
                "total_games_played_by_away": [self.total_games_played_by_away],
                "total_wins_home": [self.total_wins_home],
                "total_losses_home": [self.total_losses_home],
                "total_wins_away": [self.total_wins_away],
                "total_losses_away": [self.total_losses_away],
                "last_10_games_win_home": [self.last_10_games_win_home],
                "last_10_games_win_away": [self.last_10_games_win_away],
                "last_meeting_result": [self.last_meeting_result],
                "last_game_result_home": [self.last_game_result_home],
                "last_game_result_away": [self.last_game_result_away]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

