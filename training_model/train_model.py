import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os


os.makedirs('model', exist_ok=True)

# 1. Load datasets
train_df = pd.read_csv('../train_set_almaty_apartments.csv')
test_df = pd.read_csv('../test_set_almaty_apartments.csv', sep=';')

# 2. Drop ID column
train_df.drop(columns=['Id'], inplace=True, errors='ignore')
test_ids = test_df.get('Id', pd.Series(range(len(test_df))))  
test_df.drop(columns=['Id'], inplace=True, errors='ignore')

# 3. Separate features and target
X = train_df.drop(columns=['price'])
y = train_df['price']

# 4. Train/Test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

# 6. Evaluate on validation set
y_pred = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_val, y_pred)))
print("R2 Score:", r2_score(y_val, y_pred))

# 7. Predict on test set
test_predictions = model.predict(test_df)

# 8. Save predictions
submission_df = pd.DataFrame({'Id': test_ids, 'predicted_price': test_predictions})
submission_df.to_csv('test_predictions.csv', index=False)

# 9. Save model for deployment
joblib.dump(model, 'model/catboost_apartment_price_model.pkl')
