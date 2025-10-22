# app.py

import gradio as gr
import joblib
import numpy as np

#Load the trained health insurance model
xgb_model = joblib.load("Health_insurance_model.pk1")

# Premium calculation function
def calculate_premium(predicted_claim, expense=0.10, risk=0.05, profit=0.08, inflation=0.03):
    loadings = 1 + expense + risk + profit + inflation
    return predicted_claim * loadings

# Prediction function for Gradio
def predict_premium(age, sex, bmi, children, smoker, region,
                    expense=0.1, risk=0.05, profit=0.08, inflation=0.03):
    
    # --- Encode categorical variables exactly as in your dataset ---
    sex_encoded = 0 if sex.lower() == 'male' else 1
    smoker_encoded = 0 if smoker.lower() == 'no' else 1
    region_mapping = {'southeast':0,'southwest':1,'northeast':3,'northwest':4}
    region_encoded = region_mapping.get(region.lower(), 0)  # default to southeast
    
    # Prepare input
    X_input = np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])
    
    # Predict claim and remove logarithm normalisation
    predicted_claim = np.expm1(xgb_model.predict(X_input)[0])
    
    # Calculate final premium
    final_premium = calculate_premium(predicted_claim, expense, risk, profit, inflation)
    
    # Return as tuple for Gradio
    return f"£{predicted_claim:,.2f}", f"£{final_premium:,.2f}"

#Define Gradio inputs and outputs
inputs = [
    gr.Number(label="Age"),
    gr.Dropdown(['male', 'female'], label="Sex"),
    gr.Number(label="BMI"),
    gr.Number(label="Children"),
    gr.Dropdown(['yes', 'no'], label="Smoker"),
    gr.Dropdown(['southeast', 'southwest', 'northeast', 'northwest'], label="Region"),
    gr.Number(label="Expense Loading", value=0.10),
    gr.Number(label="Risk Margin", value=0.05),
    gr.Number(label="Profit Margin", value=0.08),
    gr.Number(label="Inflation", value=0.03)
]

outputs = [
    gr.Label(label="Predicted Claim Cost"),
    gr.Label(label="Final Premium")
]

#Gradio interface
interface = gr.Interface(
    fn=predict_premium,
    inputs=inputs,
    outputs=outputs,
    title="Health Insurance Premium Calculator",
    description="Predict annual health insurance claim costs and generate final premiums based on user inputs and loadings."
)

#Launch the app
if __name__ == "__main__":
    interface.launch(share=False)
