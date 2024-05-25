
# IPL Win Predictor using Machine Learning

To build a sophisticated AI bot that could predict the winner of the match even before it commences. It mimics human instincts by making use of the current year’s records & also considers comparative recent performances to make its prediction (Makes use of a formula based on NRR).
## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Features](#features)
- [Packages needed](#packages-needed)
- [Usage](#usage)
- [Technical Details](#technical-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Predicting the outcome of an IPL match is an intriguing exercise often influenced by human psychology. As fans, we rely on recent performances and venue-specific advantages to make our predictions. Inspired by this natural approach, my project aims to replicate this process through a statistical model.

This project leverages historical performance data and venue statistics to predict match outcomes. By mimicking the human tendency to focus on recent performances, the model primarily considers the current season's data. Additionally, it accounts for the home-ground advantage, demonstrating statistically how teams tend to perform better at familiar venues.

To refine these predictions, the model incorporates a formula that adjusts the predicted values using the Net Run Rate (NRR). NRR offers an excellent insight into a team's performance in the current scenario, making our predictions even more robust.

The result is a highly accurate model, achieving impressive accuracy in predicting match outcomes during testing. Whether you're a data enthusiast or an IPL fan, this project provides a fascinating blend of sports and data science, offering reliable predictions based on well-founded statistical principles.
## Dataset
The Dataset includes historical IPL match info team-wise. Their names, toss result, venue, runs scored, runs conceived, and match result. The Dataset is manually created and hence doesn’t have any missing data and can be directly proceeded to analysis.

Refer to filename ``IPL_team_prediction_data.csv``.
For each match, the dataset would have two rows, for two teams each. So if team1 bats first, then the data would look like: team1 name, bat (under choose column), venue (now this would contain values: home/ away depending on what it is for this team1), runs it scored when batted, runs it conceived during fielding and what was the result for team1.

If we’re now entering details for team2: team2 name, bowl, and all other would be as per the performance.

## Features
This IPL win predictor has some unique features that make it different from other scripts:

- Highly inspired by how we humans predict the outcome by using our instincts and past performances.

- It mimics our past performance analysis by applying a logistic regression prediction of an initial probability of win value, then it uses those teams’s current NRR (its calculated on the basis of data provided to date) as NRR or Net Run Rate is an excellent indicator of a team’s performance.

- The project utilizes logistic regression, leveraging its probabilistic features, and employs a formula based on the Net Run Rate (NRR) to predict a team's winning chances.

- It utilizes only the current year’s data because IPL is a tournament where a team changes its players every year, so utilizing the previous year’s data can be misleading.

- Achieves impressive accuracy in predicting win probabilities for recent matches, based on the provided data. In testing, the model demonstrated a reliable 75% accuracy rate, ensuring dependable predictions.


## Packages needed
This project is implemented entirely in Python, leveraging its robust ecosystem of libraries and tools for data analysis and machine learning. To get started, ensure you have Python 3 installed on your system. Additionally, the project requires the following packages:

- **pandas**: For efficient data manipulation and analysis, pandas provide flexible data structures to work with structured data seamlessly.
- **sklearn** (scikit-learn): This library is essential for machine learning tasks, offering simple and efficient tools for data mining and data analysis, including the logistic regression model used in this project.
- **numpy**: As a fundamental package for scientific computing in Python, numpy supports large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.

These packages can be simply installed by using:
```
pip install pandas scikit-learn numpy
```
## Usage
To run the project, simply execute the ``ipl_win_predictor.py`` file in your preferred Python environment or terminal. Make sure you have all the required packages installed as mentioned in the previous section.

Once you have installed the necessary packages, navigate to the directory containing the project files and run the main Python script:
```
cd {directory_where_project_is_downloaded}
python3 ipl_win_predictor.py
```

It would then prompt you to enter the names of the teams participating in the match for which you want to predict the outcome. Enter venue info as *home* or *away* as per the entered name of Team 1. In case of a Neutral venue, consider the nearest team’s state as home, and other, away.

This will execute the project and perform the intended analysis or predictions based on the provided dataset.
## Technical Details
This project aims to replicate human decision-making skills in predicting the outcomes of IPL matches by leveraging machine learning techniques and statistical analysis. It consists of two main stages, one complementing the other to enhance the accuracy of predictions.

### Stage 1: Team Matchup Analysis using Machine Learning
In the first stage, the model mimics human decision-making by analyzing past data and team track records. Using machine learning, specifically logistic regression classification, the model is trained on historical match records from the current year. This includes details such as team matchups, venue information, and match outcomes. By learning from this data, the model identifies trends in team matchups at specific venues, allowing it to make informed predictions.

**Data Preprocessing**:

During the data preprocessing phase, a step was implemented to enhance the robustness of the training dataset. This involved interchanging the positions of team1 and team2 in match records, ensuring that the model comprehensively understands the dynamics of team rivalries.

### Stage 2: Strength Assessment using Net Run Rate (NRR)
In the second stage, the model evaluates the strengths of the teams participating in the match. This is done mathematically by calculating the **Net Run Rate (NRR)** for each team based on match data supplied in the training dataset. Venue information is not considered in these calculations to avoid over-dependence on venue-specific factors, although it is acknowledged that venue may have some significance.

To adjust the predicted win probability obtained from the first stage based on NRR differences between the competing teams, a formula is applied:
```
Adjusted_winning_percent = predicted_win_probability + (predicted_win_probability x (percentage_difference_nrr/ 100))
```

Where:

- ``predicted_win_probability`` is the probability score obtained from the previous machine learning prediction.
- ``Percentage_difference_nrr`` = (Opposition team NRR - Current team NRR) / Current team NRR

This two-step approach enhances the model's robustness and accuracy, resulting in highly reliable predictions.
## Results
The performance of this model was evaluated using a test dataset comprising 20 matches from the recent second half of the tournament. The model achieved an impressive accuracy rate of **75%**, which is exceptionally good for sports predictions.

### Why Recent Matches Were Used for Testing
You might wonder why the test data was drawn from the second half of the tournament rather than an aggregated dataset including both recent and older matches. The reason lies in the model's reliance on past data and comparative Net Run Rate (NRR) scores. NRR provides insights into the current performance of the teams. Testing on older matches would yield contradictory and often incorrect predictions, as the NRR and team dynamics can change significantly over time. Hence, using recent matches ensures the model's predictions are based on the most relevant and up-to-date information, reflecting the current performance levels of the teams.
## Contributing
Thank you for your interest in contributing to **IPL Win Predictor using Machine Learning**! We welcome any contributions that can help improve the model's accuracy and functionality. Whether you have suggestions, bug fixes, or new features in mind, your input is invaluable. Please feel free to submit issues, provide feedback, or make pull requests. Together, we can enhance the predictive capabilities of this project and make it even better.


## License
This project is licensed under the MIT License.
