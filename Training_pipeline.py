# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from SAMKNNRegressor_model import SAMKNNRegressor
from tqdm import tqdm
import pickle
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_data():
    # Load your data with concept drift
    # Preprocess your data as required (e.g., handling missing values, encoding categorical variables)
    df=pd.read_excel("/Users/lokendrakumar.meena/Downloads/file_hzl copy/AirQualityUCI.xlsx",sheet_name="AirQualityUCI")[:-100]
    df.index=pd.to_datetime(df.Date.astype(str)+' '+df.Time.astype(str))
    df=df.drop(columns=['Date','Time','NOx(GT)'])
    df = df[df["NO2(GT)"]>-150]
    df["NO2(GT)"][-900:]=df["NO2(GT)"][-900:]+80

    return df # Your preprocessed DataFrame with concept drift
def series_to_df(series):
    df=pd.DataFrame(columns = series.index)
    df.loc[0] = series.values
    return df

# Function to train SAMKNNRegressor
def train_samknn(df):
    target_var = 'NO2(GT)'  # Replace 'target' with your target variable name
    var = df.columns.tolist()  # Assuming all columns are features except the target variable
    base = df[:-1700].reset_index(drop=True).copy()
    iter_ = df[-1700:].copy() # Splitting data into base and iterative parts
    
    x_base = base.drop(target_var, axis=1).values
    y_base = base[target_var].values
    
    sam = SAMKNNRegressor()
    sam.fit(x_base, y_base)
    pred_iterr = []
    for i in tqdm(range(iter_.shape[0])):
        data = iter_.iloc[i]
        data = series_to_df(data)

        x_iter = data.drop(target_var, axis=1).values
        y_iter = data[target_var].values
        pred_iter = sam.predict(x_iter)
        pred_iterr.append(pred_iter[0])
        sam.partial_fit(x_iter, y_iter)
        #data = iter_.iloc[i]
        #x_iter = data.drop(target_var).values.reshape(1, -1)
        #pred = sam.predict(x_iter)
        #pred_iter.append(pred[0])
        #sam.partial_fit(x_iter, data[target_var].reshape(1, -1))
    
    iter_['PRED'] = pred_iterr
    return sam, iter_

# Function to evaluate MAPE and MAE
def evaluate(predictions, target_var,model_name):
    # Compute and return MAPE and MAE
    absolute_errors = np.abs(predictions['PRED'] - predictions[target_var])
    mape = np.mean(np.abs(absolute_errors / predictions[target_var])) * 100
    mae = np.mean(absolute_errors)
    R2 = r2_score(predictions[target_var], predictions['PRED'] )
    print('R2 : ', R2)
    # Plot true vs. predicted values
    plt.figure(figsize=(10, 6))
    plt.plot(predictions.index, predictions[target_var], label='True Values')
    plt.plot(predictions.index, predictions['PRED'], label='Predicted Values')
    # Highlighting concept drift
    highlight_start_index = max(len(predictions) - 800, 0)
    highlight_end_index = len(predictions)
    highlight_start_x = predictions.index[highlight_start_index]
    highlight_end_x = predictions.index[highlight_end_index - 1]
    highlight_height = max(predictions[target_var].max(), predictions['PRED'].max())  # Adjust height as needed
    highlight_alpha = 0.2  # Adjust transparency as needed
    highlight_color = 'red'

    plt.axvspan(highlight_start_x, highlight_end_x, color=highlight_color, alpha=highlight_alpha)
    plt.text(predictions.index[-400], highlight_height * 0.9, 'Concept Drift', ha='center', va='center', fontsize=12, color='black')

    plt.xlabel('Index')
    plt.ylabel(target_var)
    plt.title(model_name+': True vs. Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()
    return mape, mae

# Main function to run the experiment
def main():
    # Load data
    data = load_data()
    
    # Train SAMKNNRegressor
    sam_model, iter_data = train_samknn(data)
    
    # Save SAMKNN model
    with open("samknn_model.pickle", "wb") as f:
        pickle.dump(sam_model, f)
    
    # Load SAMKNN model
    with open("samknn_model.pickle", "rb") as f:
        sam_model = pickle.load(f)
    
    # Test SAMKNNRegressor
    mape, mae = evaluate(iter_data, 'NO2(GT)',"SAMKNNRegressor")  # Replace 'target' with your target variable name
    print("SAMKNNRegressor MAPE:", mape)
    print("SAMKNNRegressor MAE:", mae)

    data=data.reindex()
    X_train = data.drop(columns=['NO2(GT)'])[:-1700]
    X_test = data.drop(columns=['NO2(GT)'])[-1700:]
    y_train = data['NO2(GT)'][:-1700]
    y_test = data['NO2(GT)'][-1700:]
    #X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['NO2(GT)']),data['NO2(GT)'], test_size=0.6, random_state=42)
    #Optionally, compare with KNNRegressor
    knn_model = KNeighborsRegressor()
    
    knn_model.fit(X_train, y_train)
    X_test["PRED"] = knn_model.predict(X_test)
    X_test["NO2(GT)"]=y_test
    print(X_test["PRED"] , y_test)
    knn_mape, knn_mae = evaluate(X_test, "NO2(GT)","KNNRegressor")
    print("KNNRegressor MAPE:", knn_mape)
    print("KNNRegressor MAE:", knn_mae)

if __name__ == "__main__":
    main()
