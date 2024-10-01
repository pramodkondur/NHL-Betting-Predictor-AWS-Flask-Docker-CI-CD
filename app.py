from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            awayTeamCode=request.form.get('awayTeamCode'),
            homeTeamCode=request.form.get('homeTeamCode'),
            season=int(request.form.get('season')),
            isPlayoffGame=int(request.form.get('isPlayoffGame')), 
            total_games_played_by_home=int(request.form.get('total_games_played_by_home')),
            total_games_played_by_away=int(request.form.get('total_games_played_by_away')),
            total_wins_home=int(request.form.get('total_wins_home')),
            total_losses_home=int(request.form.get('total_losses_home')),
            total_wins_away=int(request.form.get('total_wins_away')),
            total_losses_away=int(request.form.get('total_losses_away')),
            last_10_games_win_home=int(request.form.get('last_10_games_win_home')),
            last_10_games_win_away=int(request.form.get('last_10_games_win_away')),
            last_meeting_result=int(request.form.get('last_meeting_result')),  # 1 for Yes, 0 for No, 2 for Haven't played
            last_game_result_home=float(request.form.get('last_game_result_home')),
            last_game_result_away=float(request.form.get('last_game_result_away'))
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8080)


 