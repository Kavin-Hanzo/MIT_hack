import os
import requests
import json
from flask import Flask, render_template, request
import speech_recognition as sr
from gtts import gTTS
from transformers import T5ForConditionalGeneration, T5Tokenizer

app = Flask(__name__)

# ✅ API Config (Google Custom Search)
API_KEY = "AIzaSyCeliXLEqhlyp5s0YSqVRY_Rxef7n2iakY"
CX_ID = "a2a632fdde7cf45c1"  # Custom Search Engine ID

# ✅ Load T5 Model for Summarization
tokenizer = T5Tokenizer.from_pretrained("t5-small")  # Try "t5-base" for better results
model = T5ForConditionalGeneration.from_pretrained("t5-small")

# ✅ Google Medical Search Function
def google_medical_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CX_ID}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return {"error": "Failed to fetch results"}

# ✅ Summarize Text Using T5 Model
def summarize_text(text, max_length=100):
    input_text = "summarize: " + text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Speech-to-Text Function
def process_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio."
    except sr.RequestError:
        return "Error with the recognition service."

# ✅ Flask Routes
@app.route("/", methods=["GET", "POST"])
def index():
    text = None
    summary = None
    audio_path = None

    if request.method == "POST":
        # Check for text input
        text_input = request.form.get("text", "").strip()
        
        # Check for audio file
        audio_file = request.files.get("audio")

        # Convert Audio to Text if Provided
        if audio_file and audio_file.filename:
            audio_path = "uploaded_audio.wav"
            audio_file.save(audio_path)
            text = process_audio(audio_path)
        else:
            text = text_input

        # ✅ Perform Google Search and Summarization
        if text:
            search_results = google_medical_search(text)
            snippets = [item["snippet"] for item in search_results.get("items", [])]

            # Merge snippets into a single text block
            combined_text = " ".join(snippets)
            chunk_size = 512
            chunks = [combined_text[i:i + chunk_size] for i in range(0, len(combined_text), chunk_size)]
            summaries = [summarize_text(chunk, max_length=70) for chunk in chunks]
            summary = " ".join(summaries)

            # ✅ Convert Summary to Speech (TTS)
            tts = gTTS(summary)
            audio_path = "static/summary_audio.mp3"
            tts.save(audio_path)

    return render_template("index.html", text=text, summary=summary, audio_path=audio_path)

if __name__ == "__main__":
    app.run(debug=True)
