print('Welcome to IPL Win Predictor using Machine Learning!!')
print('I will help you predict the winner of a IPL match with best possible accuracy.')
print('Let me get things loaded...')

# Data Preprocessing

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Importing Dataset
data = pd.read_csv('Datasets/IPL_team_prediction_data.csv')

# Lets see how many teams we have in the dataset
all_teams = data['current_team'].unique().tolist()

"""So some teams indeed played less, this is the with respect to data of IPL 2024, till **22st May 2024.**

As the data is created by me manually, it doesn't have any outlier, or missing values. So lets move on to next steps of data preprocessing

## Feature engineering

### Aggregating single match results in single row

Creating a additional dataframe to show aggregate info of both teams of a match in single line.
Deleting 'choose' column here as we won't require it from now
"""

rows = []
for i in range(0, len(data), 2):
    current_row = data.iloc[i]
    next_row = data.iloc[i + 1] if i + 1 < len(data) else None
    if next_row is not None:
        rows.append({
            'current_team': current_row['current_team'],
            'opp_team': next_row['current_team'],
            'venue': current_row['venue'],
            'scored': current_row['scored'],
            'conceived': current_row['conceived'],
            'result': current_row['result']
        })

data_dub = pd.DataFrame(rows)

"""This would naturally halve the length of our original dataframe"""

"""### Interchanging the team names

Now, lets logically interchange the teams so that we create a more robust ML model. This would give us reliable prediction in case of user interchanges the names of teams in input
"""

# Interchanging values of teams, venue and runs in scored and conceived also the results
rows = []
for i, row in data_dub.iterrows():
  # first row with same values that of original frame
  first_row = [row['current_team'], row['opp_team'], row['venue'],
                row['scored'], row['conceived'], row['result']]

  # second row with values interchanged
  second_row = [row['opp_team'], row['current_team'],
                   'away' if row['venue'] == 'home' else 'home',
                   row['conceived'], row['scored'],
                   'loss' if row['result'] == 'won' else 'won']
  rows.append(first_row)
  rows.append(second_row)

data_robust = pd.DataFrame(rows, columns=['current_team', 'opp_team', 'venue', 'scored', 'conceived', 'result'])

"""Now this new robust dataframe must have same no. of rows as original one."""

"""### Getting avg NRR for each team as per venue

This dataset would be used in the last step where we would modify our predicted output which is winning percent with this NRR value.
"""

team_total_runs = data_robust.groupby(['current_team'])['scored'].sum().reset_index()
team_total_conceded = data_robust.groupby(['current_team'])['conceived'].sum().reset_index()
team_overs_bowled = data_robust.groupby(['current_team']).size().reset_index(name='matches_played')
team_overs_bowled['overs_bowled'] = team_overs_bowled['matches_played'] * 20
team_overs_bowled['overs_faced'] = team_overs_bowled['matches_played'] * 20

team_nrr_stats = team_total_runs.merge(team_total_conceded, on='current_team')
team_nrr_stats = team_nrr_stats.merge(team_overs_bowled[['current_team', 'matches_played', 'overs_bowled', 'overs_faced']], on='current_team')
team_nrr_stats['nrr'] = (team_nrr_stats['scored']/team_nrr_stats['overs_faced']) - (team_nrr_stats['conceived'] / team_nrr_stats['overs_bowled'])

"""## Encoding Categorical features

Label Encoding: **Venue**

OneHotEncoding: **Current Team**, **Opp Team**

Regular value replacement method: **Result**
"""

le = LabelEncoder()
data_robust['venue'] = le.fit_transform(data_robust['venue'])
data_robust['result'] = data_robust['result'].replace({'won':1, 'loss':0})
encoded_columns = pd.get_dummies(data=data_robust, columns=['current_team','opp_team'],dtype=int)

# Removing some unwanted columns, Rearranging the columns
encoded_columns.drop(columns=['scored', 'conceived'], inplace=True)
encoded_columns= encoded_columns[['current_team_CSK',
       'current_team_DC', 'current_team_GT', 'current_team_KKR',
       'current_team_LSG', 'current_team_MI', 'current_team_PBKS',
       'current_team_RCB', 'current_team_RR', 'current_team_SRH',
       'opp_team_CSK', 'opp_team_DC', 'opp_team_GT', 'opp_team_KKR',
       'opp_team_LSG', 'opp_team_MI', 'opp_team_PBKS', 'opp_team_RCB',
       'opp_team_RR', 'opp_team_SRH','venue', 'result']]

"""Now this is our Final dataframe which is ready for fitting into our ML model

# Training the model

We won't split our dataset here because this won't be the final step of the model preparation.
We would be coupling this with a NRR formula
"""

X = encoded_columns.iloc[:, :-1].values
y = encoded_columns.iloc[:, -1].values

# Scalling the input values
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_scaled, y)

# Predicting single results
print('Ok done!!')
def get_team_name(team_no,all_teams):
  while True:
    team_name = input('Enter Team{} name (in format: MI, PBKS, etc.) press Exit to leave: '.format(team_no))
    team_name = team_name.upper()
    if team_name == 'EXIT':
      print("Exiting the program.")
      return None
    if team_name in all_teams:
      return team_name
    else:
      print("Sorry, didn't recognised this team..\nPlease try Again..")

team1 = get_team_name(1,all_teams)
if team1:
  # Only prompt for the second team if the first team was found and not None
  team2 = get_team_name(2,all_teams)

while True:
  in_venue = input('Enter Venue as of Team1 (home/away): ')
  if in_venue not in data['venue'].unique().tolist():
    print('Venue unrecognised!!')
  else:
    break
  
# sample_input = ['SRH','RR','home']
venue_encoded = le.transform([in_venue])[0]

dummy_columns = [col for col in encoded_columns.columns if col not in ['venue','result']]
encoded_input_df = pd.DataFrame(0, index=[0], columns=dummy_columns)
encoded_input_df[f'current_team_{team1}'] = 1
encoded_input_df[f'opp_team_{team2}'] = 1
encoded_input_df['venue'] = venue_encoded
dummy_columns.append('venue')
encoded_input_df = encoded_input_df.reindex(columns=dummy_columns, fill_value=0)

input_data = encoded_input_df.values
input_data = sc.transform(input_data)

win_probability = classifier.predict_proba(input_data)[:, 1]
team1_nrr = team_nrr_stats.loc[team_nrr_stats['current_team'] == team1,"nrr"].values[0]
team2_nrr = team_nrr_stats.loc[team_nrr_stats['current_team'] == team2, "nrr"].values[0]
diff_percent = (team2_nrr - team1_nrr)/team1_nrr
adjusted_pred_prob = win_probability + (win_probability * (diff_percent / 100))

if adjusted_pred_prob >= 0.5:
  print('{} Wins this match with {}% chance!!'.format(team1,round(adjusted_pred_prob[0],4)*100))
else:
  print('{} Wins this match with {}% chance!!'.format(team1,round(1-adjusted_pred_prob[0],4)*100))
