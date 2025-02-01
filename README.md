# Heart Disease Prediction Using Machine Learning

## Project Overview
This project is a web application built using Streamlit that predicts the likelihood of heart disease in individuals based on various health metrics. The application utilizes machine learning models, including Logistic Regression, Random Forest, and Gradient Boosting, to provide predictions and insights.

## Features
- **Data Loading**: Loads processed heart disease data from a CSV file.
- **User Input**: Collects user information such as age, height, weight, blood pressure, cholesterol levels, and lifestyle choices.
- **Model Training**: Trains multiple machine learning models to predict heart disease.
- **Prediction Results**: Displays the probability of being healthy or having heart disease along with the disease stage.
- **Visualization**: Provides a comparison graph of predictions from different models.

## Technologies Used
- Python
- Streamlit
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Installation Instructions
1. Ensure you have Python installed on your machine.
2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
3. Install the required libraries:
   ```bash
   pip install streamlit numpy pandas scikit-learn matplotlib
   ```

## Usage Instructions
1. Run the application:
   ```bash
   streamlit run app.py
   ```
2. Open the provided local URL in your web browser to access the application.
3. Fill in the input fields with your health metrics and click the prediction buttons to see the results.

## Input Features
- **Age**: The age of the individual in years.
- **Height**: The height of the individual in centimeters.
- **Weight**: The weight of the individual in kilograms.
- **Systolic Blood Pressure**: The higher number in a blood pressure reading (mmHg).
- **Diastolic Blood Pressure**: The lower number in a blood pressure reading (mmHg).
- **Cholesterol Level**: The cholesterol level categorized as Normal, Borderline High, or High.
- **Glucose Level**: The glucose level categorized as Normal, Prediabetes, or Diabetes.
- **Smoking Status**: Indicates whether the individual is a smoker.
- **Alcohol Intake**: Indicates whether the individual consumes alcohol.
- **Physical Activity**: Indicates whether the individual engages in regular physical activity.

## Output Explanation
- **Probability of Being Healthy**: The likelihood that the individual does not have heart disease.
- **Probability of Having Heart Disease**: The likelihood that the individual has heart disease.
- **Stage**: The classification of heart disease severity based on the probability.

## Real-World Applications
This project can be utilized in healthcare settings to assist medical professionals in assessing the risk of heart disease in patients, enabling early intervention and personalized treatment plans.

## Data Sources
The dataset used in this project is sourced from [insert source here], which contains health metrics and corresponding labels for heart disease.

## Limitations
- The accuracy of the predictions is dependent on the quality and representativeness of the dataset.
- The models may not account for all factors influencing heart disease risk.

## Future Improvements
- Implement additional machine learning models for comparison.
- Enhance the user interface for better user experience.
- Add functionality for users to upload their own datasets for predictions.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## Contact Information
For any questions or support, please reach out to [gopalsharmacse2025@gmail.com].