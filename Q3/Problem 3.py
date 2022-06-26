import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def predict_innings(innings_file, prediction_file, model_file = ''):
    test_df = pd.read_csv(innings_file)

    if model_file != '':
        model = pd.read_pickle(model_file)
        test_df['runs_now'] = test_df.groupby('match_id').cumsum()['total_runs']
        test_df['player_dismissed'] = test_df['player_dismissed'].apply(lambda x: 1 if type(x) == str else 0)
        test_df['wickets_now'] = test_df.groupby('match_id').cumsum()['player_dismissed']
        test_df['ball_done'] = (test_df['over'] - 1)*6 + test_df['ball'] 
        test_df['run_rate'] = test_df['runs_now'] / (test_df['ball_done']/6)
        test_df['striker_runs'] = np.zeros(len(test_df))
        test_df['non_striker_runs'] = np.zeros(len(test_df))
        runs = {}
        runs[test_df['batsman'][0]] = test_df['batsman_runs'][0]
        runs[test_df['non_striker'][0]] = 0 
        test_df['striker_runs'][0] = test_df['batsman_runs'][0]
        for i in range(1, len(test_df)):
            if(test_df['match_id'][i] != test_df['match_id'][i-1]):
                runs = {}
            if test_df['batsman'][i] not in runs:
                runs[test_df['batsman'][i]] = test_df['batsman_runs'][i]
            else:
                runs[test_df['batsman'][i]] += test_df['batsman_runs'][i]
            if test_df['non_striker'][i] not in runs:
                runs[test_df['non_striker'][i]] = 0
            test_df['striker_runs'][i] = runs[test_df['batsman'][i]]
            test_df['non_striker_runs'][i] = runs[test_df['non_striker'][i]]
            if test_df['player_dismissed'][i] == 1:
                runs.pop(test_df['batsman'][i])
        X_test = test_df.filter(['runs_now', 'wickets_now', 'ball_done', 'run_rate', 'striker_runs', 'non_striker_runs'], axis = 1)
        preds = model.predict(X_test)
        match_scores = {}
        for i in range(len(preds)-1):
            if(test_df['match_id'][i] != test_df['match_id'][i+1]):
                match_scores[test_df['match_id'][i]] = preds[i]
        match_scores[test_df['match_id'][len(preds)-1]] = preds[len(preds)-1]
        res_df = pd.DataFrame({'match_id': match_scores.keys(), 'prediction': match_scores.values()})
        res_df = res_df.sort_values('match_id')
        res_df['prediction'] = res_df['prediction'].astype(int)
        res_df.to_csv(prediction_file, index=False)

    
    else:
        predicted_runs = test_df.groupby("match_id").apply(lambda x:np.round((120.0 * x["total_runs"].sum())/x["balls_done"].max())).reset_index(name="prediction")
        predicted_runs.to_csv(prediction_file, index=False)
    
# predict_innings('IPL_test.csv', 'IPL_prediction.csv', 'regressor.pkl')