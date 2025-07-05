# 🎓 LectureLens AI

**LectureLens AI** is an AI-powered Streamlit app that extracts transcripts from YouTube videos and helps students learn more effectively through AI-generated summaries, topic breakdowns, and interactive Q&A — all in one place.

> Powered by Google Gemini and Whisper ASR.

---

## 🚀 Features

- 🎥 **YouTube Transcript Extraction**  
  Extracts transcript using YouTube API or automatic transcription (Whisper) for multilingual support.

- 🧠 **AI Summarization**  
  Get high-level overviews of full lectures or selected sections.

- 📌 **Key Topics Identification**  
  Extract main topics covered in the video for faster revision.

- ❓ **Interactive Q&A**  
  Ask general or time-based questions about the content using Gemini AI.

- 🌐 **Multilingual Support**  
  Supports YouTube's caption languages or auto-transcription fallback.

- ✨ **Custom AI Tutor Personality**  
  Personalize the tone and style of your AI assistant.

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/lecturelens-ai.git
cd lecturelens-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.streamlit/secrets.toml` file and add your Gemini API key:

```toml
[api_keys]
gemini_api_key = "your_google_gemini_api_key"
```

---

## ▶️ Usage

Run the app locally with:

```bash
streamlit run app.py
```

Then open in your browser at `http://localhost:8501`.

Or try it online:  
👉 [Launch LectureLens AI](https://lecturelens-aigit-g3kah6rvgmbsyvtzwktkd5.streamlit.app/)

---

## 📂 Project Structure

```
lecturelens-ai/
│
├── app.py                    # Main Streamlit app
├── utils.py                  # Utility functions
├── requirements.txt
└── .streamlit/
    └── secrets.toml          # API keys
```

---

## 💡 Example Prompts

- **"Summarize this lecture in bullet points."**  
- **"List the main topics discussed between 10:00 and 20:00."**  
- **"What are the key takeaways from the last section?"**  
- **"Explain quantum tunneling like I’m 10."**

---

## 🔒 Privacy & Security

- No data is stored or logged.
- All transcript processing is done in-session and deleted on reset.

---

## 📜 License

MIT License © 2025 Pugazhmani

---

## 🙌 Acknowledgments

- [Google Gemini API](https://ai.google.dev/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Streamlit](https://streamlit.io/)
- [YouTube Transcript API](https://pypi.org/project/youtube-transcript-api/)
## 🙌 Application link
- url (https://lecturelens-aigit-g3kah6rvgmbsyvtzwktkd5.streamlit.app/)
