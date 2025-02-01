# Import necessary libraries
import numpy as np  # Library for numerical operations
import streamlit as st  # Library for creating web apps
import pandas as pd  # Library for data manipulation and analysis
from sklearn.model_selection import train_test_split  # Function to split data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Class to standardize features by removing the mean and scaling to unit variance
from sklearn.linear_model import SGDClassifier  # Stochastic Gradient Descent classifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier  # Ensemble methods for classification
import matplotlib.pyplot as plt  # Library for creating static, animated, and interactive visualizations

# Title
st.title('Heart Disease Prediction Using Machine Learning')  # Set the title of the Streamlit app

@st.cache_data
def load_data():
    df = pd.read_csv('processed_data.csv')  # Load dataset from a CSV file
    df = df.drop('id', axis=1)  # Drop the 'id' column from the dataset as it's not needed for prediction
    return df

df = load_data()

@st.cache_data
def preprocess_data(df):
    X = df.iloc[:, :-1].values  # Extract feature columns (all columns except the last one)
    Y = df.iloc[:, -1].values  # Extract target column (the last column)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1, stratify=Y)  
    # Split the data into training and testing sets. 75% for training and 25% for testing. Stratify ensures balanced class distribution
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = preprocess_data(df)

# Feature scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)  # Fit the scaler on the training data and transform it
X_test = sc.transform(X_test)  # Transform the testing data using the same scaler

@st.cache_resource
def train_models():
    logistic_model = SGDClassifier(random_state=1, loss='log_loss', max_iter=1000, penalty='l1', alpha=0.0001, warm_start=True)
    logistic_model.fit(X_train, Y_train)

    forest_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=1)
    forest_model.fit(X_train, Y_train)

    gbm_model = GradientBoostingClassifier(random_state=1)
    gbm_model.fit(X_train, Y_train)
    
    return logistic_model, forest_model, gbm_model

logistic_model, forest_model, gbm_model = train_models()

def predict_cardiovascular_disease(input_data, model):
    input_data_as_numpy_array = np.asarray(input_data)  # Convert input data to numpy array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)  # Reshape the input data for the model
    input_data_scaled = sc.transform(input_data_reshaped)  # Scale the input data using the same scaler
    prediction_proba = model.predict_proba(input_data_scaled)  # Predict the probabilities
    return prediction_proba

def get_stage(probability):
    if probability < 0.25:
        return 'Healthy'  # Return 'Healthy' if probability is less than 0.25
    elif probability < 0.5:
        return 'Stage 1'  # Return 'Stage 1' if probability is less than 0.5
    elif probability < 0.75:
        return 'Stage 2'  # Return 'Stage 2' if probability is less than 0.75
    else:
        return 'Stage 3'  # Return 'Stage 3' if probability is greater than or equal to 0.75

def get_prediction_details(input_data, model):
    prediction_proba = predict_cardiovascular_disease(input_data, model)  # Get prediction probabilities
    probability_healthy = prediction_proba[0][0]  # Extract probability of being healthy
    probability_disease = prediction_proba[0][1]  # Extract probability of having the disease
    stage = get_stage(probability_disease)  # Determine the disease stage
    return {
        'probability_healthy': probability_healthy * 100,  # Convert probability to percentage
        'probability_disease': probability_disease * 100,  # Convert probability to percentage
        'stage': stage  # Include the disease stage in the result
    }

def plot_prediction_graph(predictions):
    models = list(predictions.keys())  # Get the names of the models
    healthy_probs = [predictions[model]['probability_healthy'] for model in models]  # Get healthy probabilities for each model
    disease_probs = [predictions[model]['probability_disease'] for model in models]  # Get disease probabilities for each model

    x = np.arange(len(models))  # Create an array of model indices
    width = 0.35  # Width of the bars

    fig, ax = plt.subplots()  # Create a new figure and axes for the plot
    bars1 = ax.bar(x - width/2, healthy_probs, width, label='Healthy')  # Plot bars for healthy probabilities
    bars2 = ax.bar(x + width/2, disease_probs, width, label='Heart Disease')  # Plot bars for disease probabilities

    # Set plot labels and title
    ax.set_xlabel('Models')
    ax.set_ylabel('Probability (%)')
    ax.set_title('Prediction Probabilities by Model')
    ax.set_xticks(x)  # Set x-axis ticks to model indices
    ax.set_xticklabels(models)  # Set x-axis labels to model names
    ax.legend()  # Add a legend to the plot

    # Annotate bars with their heights
    for bar in bars1 + bars2:
        height = bar.get_height()  # Get the height of the bar
        ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')

    st.pyplot(fig)  # Display the plot in the Streamlit app

# Initialize session state to store predictions
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}  # Create an empty dictionary in session state to store predictions

# Input fields
st.write('## Input Features')  # Section header for input fields

# User input fields for model features
username = st.text_input('Enter your name')  # Text input for username
age = st.number_input('Age (in years)', min_value=18, max_value=100, value=50)  # Input for age
age_in_days = age * 365  # Convert age to days
height = st.number_input('Height (in cm)', min_value=100, max_value=250, value=170)  # Input for height
weight = st.number_input('Weight (in kg)', min_value=30.0, max_value=200.0, value=70.0)  # Input for weight
gender = st.radio('Gender', ('Male', 'Female'))  # Radio button for gender
gender = 1 if gender == 'Male' else 2  # Encode gender as 1 for Male and 2 for Female
ap_hi = st.number_input('Systolic Blood Pressure - Higher number (mmHg)', min_value=80, max_value=250, value=120)  # Input for systolic blood pressure
ap_lo = st.number_input('Diastolic Blood Pressure - Lower number (mmHg)', min_value=40, max_value=200, value=80)  # Input for diastolic blood pressure

# Cholesterol level input
cholesterol_options = {'Normal': 1, 'Borderline High': 2, 'High': 3}  # Cholesterol options with corresponding values
cholesterol = st.selectbox('Cholesterol Level (in mg/dL): Normal < 200, Borderline High 200-239, High >= 240', options=list(cholesterol_options.keys()))  
cholesterol = cholesterol_options[cholesterol]  # Get the corresponding value for the selected option

# Glucose level input
glucose_options = {'Normal': 1, 'Prediabetes': 2, 'Diabetes': 3}  # Glucose options with corresponding values
glucose = st.selectbox('Glucose Level (in mg/dL): Normal < 100, Prediabetes 100-125, Diabetes >= 126', options=list(glucose_options.keys()))  
glucose = glucose_options[glucose]  # Get the corresponding value for the selected option

# Smoking status input
smoking = st.radio('Smoking', ('No', 'Yes'))  # Radio button for smoking status
smoking = 1 if smoking == 'Yes' else 0  # Encode smoking status as 1 for Yes and 0 for No

# Alcohol intake input
alcohol = st.radio('Alcohol Intake', ('No', 'Yes'))  # Radio button for alcohol intake
alcohol = 1 if alcohol == 'Yes' else 0  # Encode alcohol intake as 1 for Yes and 0 for No

# Physical activity input
physical_activity = st.radio('Physical Activity: Do you engage in regular physical activity (e.g., 150 minutes of moderate exercise per week)?', ('No', 'Yes'))  
physical_activity = 1 if physical_activity == 'Yes' else 0  # Encode physical activity as 1 for Yes and 0 for No

# Prediction buttons
if st.button('Heart Disease Test Result using Logistic Regression Model'):
    user_input = [age_in_days, height, weight, gender, ap_hi, ap_lo, cholesterol, glucose, smoking, alcohol, physical_activity]  # Collect user input
    user_input = [float(x) for x in user_input]  # Convert input to float
    prediction_details = get_prediction_details(user_input, logistic_model)  # Get prediction details from logistic model
    st.session_state.predictions['Logistic Regression'] = prediction_details  # Store prediction details in session state
    st.write(f"Probability of being healthy: {prediction_details['probability_healthy']:.2f}%")  # Display probability of being healthy
    st.write(f"Probability of having heart disease: {prediction_details['probability_disease']:.2f}%")  # Display probability of having heart disease
    st.write(f"Stage: {prediction_details['stage']}")  # Display stage of heart disease

if st.button('Heart Disease Test Result using Random Forest Model'):
    user_input = [age_in_days, height, weight, gender, ap_hi, ap_lo, cholesterol, glucose, smoking, alcohol, physical_activity]  # Collect user input
    user_input = [float(x) for x in user_input]  # Convert input to float
    prediction_details = get_prediction_details(user_input, forest_model)  # Get prediction details from random forest model
    st.session_state.predictions['Random Forest'] = prediction_details  # Store prediction details in session state
    st.write(f"Probability of being healthy: {prediction_details['probability_healthy']:.2f}%")  # Display probability of being healthy
    st.write(f"Probability of having heart disease: {prediction_details['probability_disease']:.2f}%")  # Display probability of having heart disease
    st.write(f"Stage: {prediction_details['stage']}")  # Display stage of heart disease

if st.button('Heart Disease Test Result using Gradient Boosting Model'):
    user_input = [age_in_days, height, weight, gender, ap_hi, ap_lo, cholesterol, glucose, smoking, alcohol, physical_activity]  # Collect user input
    user_input = [float(x) for x in user_input]  # Convert input to float
    prediction_details = get_prediction_details(user_input, gbm_model)  # Get prediction details from gradient boosting model
    st.session_state.predictions['Gradient Boosting'] = prediction_details  # Store prediction details in session state
    st.write(f"Probability of being healthy: {prediction_details['probability_healthy']:.2f}%")  # Display probability of being healthy
    st.write(f"Probability of having heart disease: {prediction_details['probability_disease']:.2f}%")  # Display probability of having heart disease
    st.write(f"Stage: {prediction_details['stage']}")  # Display stage of heart disease

if st.button('Show Prediction Comparison Graph'):
    if st.session_state.predictions:
        plot_prediction_graph(st.session_state.predictions)  # Plot prediction comparison graph if predictions are available
    else:
        st.write("Please make predictions using the models first.")  # Prompt user to make predictions first
