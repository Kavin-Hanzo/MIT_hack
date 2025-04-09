import streamlit as st
from streamlit_mic_recorder import mic_recorder
import speech_recognition as sr
from duckduckgo_search import DDGS
from transformers import pipeline
import gtts
import io
import os
import tempfile
import soundfile as sf
import numpy as np
import traceback

# --- Configuration & Initialization ---

# Initialize Speech Recognizer
r = sr.Recognizer()

# Load Summarization Pipeline (using @st.cache_resource to load only once)
# Using a distilled version for faster loading and inference, good for demos.
# Replace with "facebook/bart-large-cnn" or others for potentially higher quality.
@st.cache_resource
def load_summarizer():
    st.info("Loading summarization model (might take a moment on first run)...")
    try:
        summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        st.success("Summarization model loaded.")
        return summarizer
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        st.error(traceback.format_exc()) # More detailed error for debugging
        st.stop() # Stop execution if model fails to load

summarizer = load_summarizer()

# --- Helper Functions ---

def transcribe_audio(audio_bytes, sample_rate=16000):
    """
    Transcribes audio bytes using SpeechRecognition library.
    Saves bytes to a temporary WAV file as SpeechRecognition works best with files.
    """
    st.info("Transcribing audio...")
    try:
        # Convert bytes to a NumPy array first if necessary (depends on mic_recorder output format)
        # Assuming mic_recorder gives raw bytes that soundfile can interpret with subtype='PCM_16'
        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            # Write the bytes to the WAV file using soundfile
            # Adjust channels/subtype if needed based on actual recorder output
            sf.write(tmp_wav.name, np.frombuffer(audio_bytes, dtype=np.int16), sample_rate, subtype='PCM_16')
            wav_filename = tmp_wav.name

        # Use the temporary WAV file with SpeechRecognition
        with sr.AudioFile(wav_filename) as source:
            audio_data = r.record(source) # Read the entire audio file

        # Recognize speech using Google Web Speech API (requires internet)
        text = r.recognize_google(audio_data)
        st.success("Transcription successful.")
        return text

    except sr.UnknownValueError:
        st.warning("Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        st.error(f"An error occurred during transcription: {e}")
        st.error(traceback.format_exc()) # Detailed error
        return None
    finally:
        # Clean up the temporary file
        if 'wav_filename' in locals() and os.path.exists(wav_filename):
            os.remove(wav_filename)


def perform_search(query, max_results=5):
    """Performs a web search using DuckDuckGo Search."""
    st.info(f"Searching for: '{query}'...")
    search_results_text = ""
    try:
        with DDGS() as ddgs:
            results = [r for r in ddgs.text(query, max_results=max_results)]
            if not results:
                st.warning("No search results found.")
                return None

            # Combine titles and bodies/snippets for summarization context
            for result in results:
                search_results_text += f"Title: {result.get('title', 'N/A')}\nSnippet: {result.get('body', 'N/A')}\nURL: {result.get('href', 'N/A')}\n\n"

        st.success(f"Found {len(results)} results.")
        return search_results_text.strip() # Return combined text

    except Exception as e:
        st.error(f"An error occurred during search: {e}")
        st.error(traceback.format_exc()) # Detailed error
        return None


def summarize_text(text_to_summarize):
    """Summarizes text using the Hugging Face pipeline."""
    st.info("Summarizing search results...")
    if not text_to_summarize or len(text_to_summarize) < 100: # Basic check
         st.warning("Not enough text content from search results to summarize effectively.")
         return text_to_summarize # Return original if too short

    try:
        # Max length for distilbart is 1024 tokens, min length helps avoid trivial summaries
        # Adjust max_length and min_length as needed
        summary = summarizer(text_to_summarize, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        st.success("Summarization complete.")
        return summary
    except Exception as e:
        st.error(f"An error occurred during summarization: {e}")
        st.error(traceback.format_exc()) # Detailed error
        return "Error during summarization."


def text_to_speech(text):
    """Converts text to speech using gTTS and returns audio bytes."""
    st.info("Converting summary to speech...")
    if not text:
        st.warning("No text to convert to speech.")
        return None
    try:
        tts = gtts.gTTS(text)
        # Save TTS audio to a BytesIO object (in-memory file)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0) # Rewind the buffer to the beginning
        st.success("Text-to-speech conversion complete.")
        return mp3_fp.read() # Return the raw bytes
    except Exception as e:
        st.error(f"An error occurred during text-to-speech conversion: {e}")
        st.error(traceback.format_exc()) # Detailed error
        return None


# --- Streamlit App UI and Workflow ---

st.set_page_config(layout="wide")
st.title("Voice Search & Summarization Assistant")
st.write("Record your query, get search results summarized, and hear the summary.")
st.markdown("---")

# Initialize session state variables
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'search_results' not in st.session_state:
    st.session_state.search_results = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'audio_summary' not in st.session_state:
    st.session_state.audio_summary = None
if 'processing_complete' not in st.session_state:
     st.session_state.processing_complete = False

# Step 1: Record Audio
st.subheader("1. Record Your Query")
st.info("Click the microphone icon below, speak your query clearly, and click it again to stop.")

# Use streamlit-mic-recorder widget
# The key provides a unique identifier. `start_prompt` and `stop_prompt` customize button text.
# `format="wav"` might simplify things but let's stick to default bytes and handle conversion.
audio_info = mic_recorder(
    start_prompt="⏺️ Start Recording",
    stop_prompt="⏹️ Stop Recording",
    just_once=True, # Record only once until manually restarted if needed
    use_container_width=False,
    # format="wav", # Optional: If you prefer WAV output directly
    callback=None, # We process after recording stops, using the return value
    args=(),
    kwargs={},
    key='mic_recorder' # Assign a key
)

# Process audio *after* recording stops (when audio_info is populated)
if audio_info and audio_info.get('bytes'):
    st.audio(audio_info['bytes'], format='audio/wav') # Play back recorded audio (optional)
    st.session_state.processing_complete = False # Reset processing flag
    st.session_state.transcribed_text = ""     # Clear previous results
    st.session_state.search_results = ""
    st.session_state.summary = ""
    st.session_state.audio_summary = None

    # --- Start Processing Workflow ---
    with st.spinner("Processing audio..."):
        # Get audio bytes and sample rate from the recorder info
        audio_bytes = audio_info['bytes']
        sample_rate = audio_info.get('sample_rate', 16000) # Use default if not provided

        # 1. Transcription
        st.session_state.transcribed_text = transcribe_audio(audio_bytes, sample_rate)

        if st.session_state.transcribed_text:
            # 2. Search
            st.session_state.search_results = perform_search(st.session_state.transcribed_text)

            if st.session_state.search_results:
                # 3. Summarization
                st.session_state.summary = summarize_text(st.session_state.search_results)

                if st.session_state.summary and "Error" not in st.session_state.summary:
                     # 4. Text-to-Speech
                     st.session_state.audio_summary = text_to_speech(st.session_state.summary)
                     st.session_state.processing_complete = True # Mark processing as done

# --- Display Results ---
st.markdown("---")
st.subheader("2. Results")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Transcribed Text")
    if st.session_state.transcribed_text:
        st.text_area("You said:", value=st.session_state.transcribed_text, height=100, disabled=True)
    else:
        st.caption("Waiting for recording...")

    st.markdown("#### Search Results (Raw Snippets)")
    if st.session_state.search_results:
        st.text_area("Found online:", value=st.session_state.search_results, height=300, disabled=True)
    elif st.session_state.transcribed_text:
        st.caption("No search performed or no results found.")
    else:
         st.caption("Waiting for transcription...")


with col2:
    st.markdown("#### Summarized Answer")
    if st.session_state.summary:
        st.text_area("Summary:", value=st.session_state.summary, height=150, disabled=True)

        st.markdown("#### Listen to Summary")
        if st.session_state.audio_summary:
            st.audio(st.session_state.audio_summary, format='audio/mp3')
        elif st.session_state.processing_complete:
             st.warning("Could not generate audio for the summary.")
        else:
             st.caption("Waiting for summary generation...")

    elif st.session_state.search_results:
         st.caption("No summary generated or required.")
    else:
         st.caption("Waiting for search results...")


st.markdown("---")
st.caption("Powered by Streamlit, SpeechRecognition, DuckDuckGo, Hugging Face Transformers, and gTTS.")

# --- Important Considerations ---
st.sidebar.title("Notes & Considerations")
st.sidebar.markdown("""
- **Internet Required:** Transcription (Google) and TTS (gTTS) need an internet connection. Search also needs it.
- **Model Loading:** The summarization model downloads on first run, which can take time and requires disk space.
- **Transcription Accuracy:** Depends on recording quality, accent, and background noise.
- **Search Quality:** Using DuckDuckGo basic search. For more robust results, consider integrating a paid API (Google Search, Bing Search, Serper API) using `st.secrets` for keys.
- **Summarization Quality:** Depends on the model used and the clarity/relevance of search results. `distilbart` is fast but might miss nuances.
- **Resource Usage:** NLP models can be memory-intensive.
- **Error Handling:** Basic error handling is included, but real-world apps might need more robust checks.
- **Dependencies:** Make sure all libraries in `requirements.txt` are installed (`pip install -r requirements.txt`). You might need system libraries like `ffmpeg` or build tools depending on your OS.
""")