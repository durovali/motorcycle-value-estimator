import json
import os
import pickle

import gradio as gr
import numpy as np
from openai import OpenAI
from transformers import pipeline

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Load CV model for brand classification
cv_classifier = pipeline("image-classification", model="durovali/vit-motorcycle")

# Load ML model for price prediction
with open("motorcycle_price_model.pkl", "rb") as f:
    price_model = pickle.load(f)

with open("brand_codes.pkl", "rb") as f:
    brand_codes = pickle.load(f)

# Mapping from CV labels to dataset brand names
cv_to_brand = {
    "bmw": "BMW",
    "honda": "Honda",
    "kawasaki": "Kawasaki",
    "suzuki": "Suzuki",
    "triumph": None,
    "yamaha": "Yamaha",
}


def classify_motorcycle(image):
    """Classify the motorcycle brand from the uploaded image."""
    if image is None:
        return None, {}
    results = cv_classifier(image)
    scores = {r["label"]: round(r["score"], 4) for r in results}
    top_label = results[0]["label"]
    return top_label, scores


def predict_price(brand_name, year, km_driven, owner_num):
    """Predict the price using the trained Gradient Boosting model."""
    if brand_name not in brand_codes:
        return None, f"Marke '{brand_name}' nicht im Preismodell vorhanden."

    brand_code = brand_codes[brand_name]
    age = 2024 - year

    features = np.array([[brand_code, year, km_driven, owner_num, age]])
    prediction = price_model.predict(features)[0]
    return round(float(prediction), 2), None


def generate_explanation(brand, year, km, owner, price, cv_scores):
    """Generate a German explanation of the price estimate using OpenAI."""
    if openai_client is None:
        return "OpenAI API Key fehlt. Bitte als Secret eintragen."

    system_prompt = (
        "Du bist ein Motorrad-Experte und hilfst bei der Wertschaetzung von Gebrauchtmotorraedern. "
        "Erklaere die Preisschaetzung kurz und verstaendlich auf Deutsch. "
        "Erwahne die wichtigsten Faktoren (Marke, Alter, Kilometer, Besitzer). "
        "Gib auch eine kurze Einschaetzung ob der Preis fair ist. "
        "Fuege einen Hinweis auf Unsicherheit hinzu. "
        "Berechne KEINEN neuen Preis. "
        "Antworte NUR mit validem JSON ohne Markdown: "
        '{"answer": "<deine Erklaerung auf Deutsch>"}'
    )

    user_prompt = (
        f"Motorrad: {brand}, Baujahr {year}, {km} km gefahren, {owner}. Besitzer. "
        f"Die Bilderkennung hat folgende Konfidenz: {json.dumps(cv_scores)}. "
        f"Geschaetzter Preis: {price:.0f} INR (Indische Rupien). "
        f"Das entspricht ca. {price * 0.011:.0f} CHF."
    )

    try:
        response = openai_client.responses.create(
            model=OPENAI_MODEL,
            instructions=system_prompt,
            input=user_prompt,
        )
        raw = response.output_text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        parsed = json.loads(raw)
        return parsed.get("answer", raw)
    except Exception as e:
        return f"Fehler bei der Erklaerung: {str(e)}"


def run_pipeline(image, year, km_driven, owner):
    """Main pipeline: CV -> ML -> NLP."""
    cv_label, cv_scores = classify_motorcycle(image)

    if cv_label is None:
        return {}, "Kein Bild", 0, "Bitte lade ein Motorrad-Bild hoch."

    brand_name = cv_to_brand.get(cv_label)
    if brand_name is None:
        return (
            cv_scores,
            f"Erkannt: {cv_label}",
            0,
            f"Die Marke '{cv_label}' ist leider nicht im Preisdatensatz enthalten.",
        )

    owner_num = int(owner.replace(".", "")[0])
    price, error = predict_price(brand_name, int(year), int(km_driven), owner_num)

    if error:
        return cv_scores, f"Erkannt: {cv_label}", 0, error

    explanation = generate_explanation(
        brand_name, int(year), int(km_driven), owner_num, price, cv_scores
    )

    result_text = f"Erkannt: {brand_name} (Konfidenz: {cv_scores.get(cv_label, 0):.1%})"

    return cv_scores, result_text, price, explanation


with gr.Blocks(title="Motorcycle Value Estimator") as demo:
    gr.Markdown(
        """
        # Motorcycle Value Estimator
        Lade ein Foto deines Motorrads hoch und gib die Details ein.
        Die App erkennt die Marke, schaetzt den Preis und erklaert die Bewertung.

        **So funktioniert es:**
        1. Computer Vision erkennt die Motorrad-Marke
        2. ML-Modell berechnet den geschaetzten Preis
        3. LLM erklaert die Wertschaetzung
        """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Motorrad-Foto")
            year_input = gr.Slider(
                minimum=2000, maximum=2024, value=2018, step=1, label="Baujahr"
            )
            km_input = gr.Number(value=15000, label="Kilometer gefahren")
            owner_input = gr.Dropdown(
                choices=["1. Besitzer", "2. Besitzer", "3. Besitzer", "4. Besitzer"],
                value="1. Besitzer",
                label="Besitzer",
            )
            submit_btn = gr.Button("Wert schaetzen", variant="primary")

        with gr.Column():
            cv_output = gr.JSON(label="Bilderkennung (CV)")
            brand_output = gr.Textbox(label="Erkannte Marke")
            price_output = gr.Number(label="Geschaetzter Preis (INR)")
            explanation_output = gr.Textbox(label="Erklaerung", lines=8)

    submit_btn.click(
        fn=run_pipeline,
        inputs=[image_input, year_input, km_input, owner_input],
        outputs=[cv_output, brand_output, price_output, explanation_output],
    )

    gr.Markdown(
        """
        ---
        **Hinweis:** Die Preise basieren auf indischen Marktdaten (INR).
        Die Schaetzung dient nur als Orientierung.
        """
    )

demo.launch()
