# MLB Prediction Lab

MLB Prediction Lab is an interactive Python Shiny application that uses machine learning models to analyze three baseball questions:

- Hall of Fame career probability
- All-MLB selection probability
- Postseason advancement probability

## Live App

[Open MLB Prediction Lab](https://greenbergb23-mlbpredictionlab.share.connect.posit.cloud/)

## Models

### Hall of Fame Predictor

The Hall of Fame model evaluates a player's career statistics and awards to estimate how closely the player's profile resembles historical Hall of Fame inductees.

The model uses regularized logistic regression and includes variables such as:

- Hits
- Home runs
- Walks
- Batting average
- On-base percentage
- Slugging percentage
- Service years
- MVP awards
- Gold Gloves
- Silver Sluggers
- Position
- 3,000-hit milestone

The model uses a cross-validated classification threshold rather than the default 0.50 cutoff.

### All-MLB Predictor

The All-MLB model evaluates an individual player season and estimates the probability that the season resembles a First Team or Second Team All-MLB selection.

Features include:

- Plate appearances
- Hits
- Home runs
- RBI
- Walks
- Stolen bases
- AVG
- OBP
- SLG
- OPS
- ISO
- Home run rate
- Walk rate
- Strikeout rate
- Primary position

The model uses chronological cross-validation so that earlier seasons are used to predict later seasons.

### Postseason Advancement Predictor

The postseason model evaluates team-level regular-season performance and estimates probabilities for:

- Making the playoffs
- Reaching the Division Series
- Reaching the Championship Series
- Reaching the World Series
- Winning the World Series

The postseason model uses a cumulative-threshold logistic regression framework because postseason outcomes are ordered stages.

Features include:

- Runs per game
- AVG
- OBP
- SLG
- ISO
- Home run rate
- Walk rate
- Strikeout rate
- ERA
- WHIP
- K/9
- BB/9
- HR/9
- Run differential per game
- Winning percentage

## Model Evaluation

The models were evaluated using multiple metrics, including:

- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- PR-AUC
- Brier score
- Log loss

These metrics were used together because the positive classes in the Hall of Fame and All-MLB models are relatively rare.

## Technology

The application was built using:

- Python
- Shiny for Python
- pandas
- NumPy
- scikit-learn
- joblib
- GitHub
- Posit Connect Cloud

## Deployment

The project is stored on GitHub and deployed through Posit Connect Cloud.

Live application:

https://greenbergb23-mlbpredictionlab.share.connect.posit.cloud/