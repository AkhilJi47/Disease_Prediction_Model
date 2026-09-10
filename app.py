import pandas as pd
import gradio as gr
from joblib import load
from google import genai

# ==========================================
# GEMINI CLIENT
# ==========================================
client = genai.Client(api_key="AIzaSyARJ5v9N99NwtkDtZZ0iPmEO5jJgjkgU-o")

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv("./dataset/training_data.csv")
symptoms = list(df.columns[:-2])

# ==========================================
# LOAD MODEL
# ==========================================
model = load("./saved_model/random_forest.joblib")

# ==========================================
# PREDICT FUNCTION
# ==========================================
# ==========================================
# REPLACE ONLY predict_disease() FUNCTION
# ==========================================

def predict_disease(selected_symptoms):
    try:
        if not selected_symptoms:
            return """
            <div class="result-card">
                <div class="result-disclaimer">
                    ⚠️ Please select at least one symptom.
                </div>
            </div>
            """

        # ----------------------------
        # ML Prediction
        # ----------------------------
        input_vector = [0] * len(symptoms)

        for symptom in selected_symptoms:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                input_vector[index] = 1

        prediction = model.predict([input_vector])[0]

        # ----------------------------
        # GEMINI EXPLANATION
        # ----------------------------
        prompt = f"""
Disease: {prediction}

Explain in clean SIMPLE HTML only.

Use EXACT format:

<h3>Possible Causes</h3>
<ul><li>...</li></ul>

<h3>How to Recover</h3>
<ul><li>...</li></ul>

<h3>When to Consult Doctor</h3>
<ul><li>...</li></ul>

<h3>Prevention Tips</h3>
<ul><li>...</li></ul>

Rules:
No markdown
No ```html
No body tag
No style tag
Only direct HTML
Short concise answer
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        ai_html = response.text if response.text else "No extra insights available."

        # remove markdown code blocks if Gemini gives them
        ai_html = ai_html.replace("```html", "").replace("```", "").strip()

        # ----------------------------
        # FINAL HTML OUTPUT
        # ----------------------------
        final_html = f"""
        <div class="result-card">

            <div class="result-icon">🔬</div>

            <div class="result-label">
                Possible Condition Detected
            </div>

            <div class="result-disease">
                {prediction}
            </div>

            <div class="result-disclaimer">
                ⚠️ This is AI-assisted insight only,
                not a medical diagnosis.
            </div>

            <div style='margin-top:25px;text-align:left;'>
                {ai_html}
            </div>

        </div>
        """

        return final_html

    except Exception as e:
        return f'''
        <div class="result-card">
            <div class="result-disclaimer">
                ⚠️ Error: {str(e)}
            </div>
        </div>
        '''

# ==========================================
# CHAT FUNCTION (FIXED)
# ==========================================
def chat_with_amigo(message, history):
    try:
        prompt = f"""
You are My AMIGO, a warm, empathetic AI health companion.

Rules:
- Be caring
- Be calm
- Use simple language
- Never diagnose disease
- Suggest doctor when needed
- Keep answers short

User message: {message}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        if response.text:
            return response.text
        else:
            return "I'm here with you. Please try asking again."

    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

# ==========================================
# CSS
# ==========================================
css = """
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

:root{
 --cream:#fdf8f2;
 --warm:#f7f0e8;
 --card:#ffffff;
 --teal:#2a7f7f;
 --teal2:#3aa8a8;
 --amber:#e8924a;
 --amberlt:#fde8d4;
 --text:#1e2d2d;
 --muted:#6b8080;
 --border:#d9ece9;
}

.gradio-container{
 background:var(--cream)!important;
 font-family:'DM Sans',sans-serif!important;
}

/* HEADER */
.app-hero{
 background:linear-gradient(135deg,#1a5f5f,#2a7f7f,#1e4a6b);
 padding:55px;
 text-align:center;
 border-radius:0 0 30px 30px;
}

.hero-title{
 font-size:52px;
 color:white;
 font-family:'DM Serif Display',serif;
}

.hero-sub{
 color:rgba(255,255,255,.8);
 font-size:18px;
 margin-top:8px;
}

/* CARD */
.section-card{
 background:white;
 border-radius:18px;
 padding:28px;
 box-shadow:0 10px 35px rgba(0,0,0,.08);
 margin-top:20px;
}

/* BUTTON */
.analyze-btn{
 background:linear-gradient(135deg,var(--teal),var(--teal2))!important;
 color:white!important;
 border:none!important;
 border-radius:12px!important;
 font-weight:700!important;
}

/* RESULT */
.result-card{
 background:#f0fafa;
 border-radius:18px;
 padding:30px;
 text-align:center;
 margin-top:20px;
 border:1px solid #d8f0f0;
}

.result-icon{
 font-size:40px;
}

.result-label{
 font-size:13px;
 color:var(--teal);
 font-weight:700;
 margin-top:10px;
}

.result-disease{
 font-size:34px;
 color:#111;
 margin-top:10px;
 font-family:'DM Serif Display',serif;
}

.result-disclaimer{
 margin-top:18px;
 background:var(--amberlt);
 padding:12px;
 border-radius:10px;
 color:#000;
}

/* TIP */
.tip-box{
 display:flex;
 gap:12px;
 background:var(--amberlt);
 border-left:4px solid var(--amber);
 padding:15px;
 border-radius:10px;
 margin-top:20px;
 color:#000!important;
}

.tip-box span,
.tip-box strong{
 color:#000!important;
}

.result-card h3{
 color:#111;
 margin-top:20px;
 margin-bottom:8px;
 font-size:20px;
 text-align:left;
}

.result-card ul{
 text-align:left;
 padding-left:20px;
 color:#222;
}

.result-card h3{
    color:#111 !important;
    font-size:28px;
    margin-top:24px;
    margin-bottom:12px;
    text-align:left;
    font-weight:700;
}

.result-card p,
.result-card li,
.result-card ul,
.result-card span,
.result-card div{
    color:#222 !important;
    text-align:left;
    font-size:18px;
    line-height:1.8;
}

.result-card ul{
    padding-left:24px;
    margin-bottom:15px;
}

.result-card li{
    margin-bottom:8px;
}

.result-card li{
 margin-bottom:8px;
}

/* CHAT TEXTBOX */
.chat-input-row textarea{
 background:white!important;
 color:#000!important;
 caret-color:#000!important;
 border-radius:12px!important;
 border:1px solid #ccc!important;
}

.chat-input-row textarea::placeholder{
 color:#000!important;
 opacity:1!important;
}
"""

# ==========================================
# UI
# ==========================================
with gr.Blocks(css=css, title="My AMIGO") as app:

    gr.HTML("""
    <div class="app-hero">
        <div class="hero-title">🩺 My AMIGO</div>
        <div class="hero-sub">AI Medical Insight & Guidance Optimized</div>
    </div>
    """)

    with gr.Tabs():

        # ==================================
        # TAB 1
        # ==================================
        with gr.Tab("🔬 Symptom Checker"):

            gr.HTML("""
            <div class="section-card">
                <h2 style="color:#111;">What's bothering you today?</h2>
                <p style="color:#666;margin-top:8px;">
                    Select symptoms and get AI-powered health insights.
                </p>
            """)

            symptom_selector = gr.Dropdown(
                choices=symptoms,
                multiselect=True,
                label="Your Symptoms",
                info="Search or select symptoms"
            )

            predict_btn = gr.Button(
                "✦ Analyze Symptoms",
                elem_classes="analyze-btn"
            )

            output = gr.HTML()

            gr.HTML("""
            <div class="tip-box">
                <span>💡</span>
                <span>
                    For best results, select <strong>all symptoms</strong>
                    you're experiencing.
                </span>
            </div>
            </div>
            """)

            predict_btn.click(
                fn=predict_disease,
                inputs=symptom_selector,
                outputs=output
            )

        # ==================================
        # TAB 2
        # ==================================
        with gr.Tab("💬 Chat with AMIGO"):

            gr.ChatInterface(
                fn=chat_with_amigo,
                chatbot=gr.Chatbot(height=450),
                textbox=gr.Textbox(
                    placeholder="Type your question or concern...",
                    container=False,
                    elem_classes="chat-input-row",
                    submit_btn="Send →"
                )
            )

# ==========================================
# RUN
# ==========================================
app.launch()