# Linear Regression from Scratch

This project implements a simple linear regression model using gradient descent — from scratch — to predict housing prices based on the average number of rooms (`RM`) using the Boston Housing dataset.

## Files
- `data/boston.csv` – the dataset
- `model.py` – functions for training and prediction
- `predict.py` – use the trained model to predict prices for new inputs
- `train.py` – script to load data and train the model

## How to Use

1. Run `train.py` to train the model.
2. Note or save the model parameters.
3. Use `predict.py` to make predictions with a given number of rooms.

## Example
```bash
python train.py
python predict.py
