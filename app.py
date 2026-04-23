import pandas as pd
import gradio as gr
from joblib import load
from google import genai

# Gemini client
client = genai.Client(api_key="AIzaSyAChHGyVWVuPJzr7C4JauRaGXgXJDsqvSE")

# Load dataset
df = pd.read_csv("./dataset/training_data.csv")
symptoms = list(df.columns[:-2])

# Load trained model
model = load("./saved_model/random_forest.joblib")


def predict_disease(selected_symptoms):

    if not selected_symptoms:
        return "Please select at least one symptom."

    input_vector = [0] * len(symptoms)

    for symptom in selected_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1

    prediction = model.predict([input_vector])[0]

    return f"""
🧠 **My AMIGO Insight**

Possible condition:
### {prediction}

⚠️ This is not a medical diagnosis.
Please consult a doctor for proper evaluation.
"""


def chat_with_amigo(message, history):

    prompt = f"""
You are My AMIGO, a supportive AI health companion.

Be empathetic and helpful. Do not give medical diagnosis.
Encourage users to seek professional help if needed.

User message: {message}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text


with gr.Blocks() as app:

    gr.Markdown("# 🧠 My AMIGO")
    gr.Markdown("AI Companion for Health Awareness")

    with gr.Tab("Symptom Checker"):

        symptom_selector = gr.Dropdown(
            choices=symptoms,
            multiselect=True,
            label="Select Symptoms"
        )

        predict_btn = gr.Button("Analyze Symptoms")

        output = gr.Markdown()

        predict_btn.click(
            fn=predict_disease,
            inputs=symptom_selector,
            outputs=output
        )

    with gr.Tab("Chat with My AMIGO"):

        chatbot = gr.ChatInterface(
            fn=chat_with_amigo,
            title="Talk with My AMIGO"
        )

app.launch()