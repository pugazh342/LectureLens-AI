LectureLens AI: Your Smart YouTube Tutor
LectureLens AI is an AI-powered Streamlit application designed to help you get the most out of educational YouTube videos. It extracts transcripts, allows you to ask questions about the video content (both general and specific sections), generates summaries, extracts key topics, and even lets you customize the AI's "tutor personality."

This version includes a basic login/sign-up page to simulate user authentication.

Features
User Authentication: Basic login and sign-up functionality (simulated for this environment).

YouTube Transcript Extraction: Fetches transcripts from YouTube videos.

Multilingual Transcript Support: Automatically detects and allows selection of available transcript languages.

YouTube Video Player: Embeds the video player directly into the application for concurrent viewing and learning.

AI-Powered Q&A: Ask questions about the entire video transcript.

Section-Specific Q&A: Query specific time ranges within the video using start and end timestamps.

Video Summarization: Generate concise summaries of the entire video content.

Key Topic & Phrase Extraction: Identify important topics and key phrases discussed in the video.

Custom Tutor Instructions: Personalize the AI's answering style and behavior.

Full Transcript Viewer: View the complete transcript with timestamps.

Chat History: Keeps a record of all your questions and the AI's answers.

Clear/Reset Functionality: Easily clear the current video data and chat history.

Setup and Installation
To run LectureLens AI locally, follow these steps:

1. Clone the Repository (or save the file)
If this were a repository, you would clone it. For now, ensure you have the test.py file (the one provided in the previous response) saved in a directory on your local machine. Let's assume you save it as lecture_lens_ai.py.

2. Create a Virtual Environment (Recommended)
It's good practice to use a virtual environment to manage dependencies.

python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using pip.

pip install streamlit langchain-google-genai langchain-community pytube youtube-transcript-api python-dotenv

4. Obtain a Google API Key
LectureLens AI uses Google's Generative AI models (Gemini). You need an API key to access them.

Go to Google AI Studio.

Sign in with your Google account.

Create a new API key.

Copy the generated API key.

5. Configure Environment Variables
Create a file named .env in the same directory as your lecture_lens_ai.py file.

Open the .env file and add your Google API Key:

GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"

Replace "YOUR_GEMINI_API_KEY_HERE" with the actual API key you obtained.

6. Run the Application
Once all dependencies are installed and your API key is configured, you can run the Streamlit application:

streamlit run lecture_lens_ai.py

This command will open the LectureLens AI application in your default web browser.

Usage
Authentication:

When you first open the app, you'll see a login/sign-up form in the sidebar.

You can "Sign Up" to create a new account or "Login" if you've "signed up" before. (Note: For this local version, the authentication is simulated and does not connect to a real Firebase backend. Any email/password combination will "work" for demonstration purposes).

Once "logged in," the main application features will become visible.

Process Video:

Enter a YouTube video URL (e.g., a lecture, tutorial, or educational talk) into the "Enter YouTube Video URL" text box.

Click the "✨ Process Video Transcript" button. The app will fetch available transcripts.

Load Transcript:

If multiple languages are available, select your preferred language from the "Choose a language" dropdown.

Click the "▶️ Load Selected Transcript" button to load the transcript and prepare the AI for questioning.

Explore Features:

Video Player: Watch the video directly in the app.

Video Summary: Click "Generate Summary" to get a concise overview.

Key Topics & Phrases: Click "Extract Topics & Phrases" to see a bulleted list of main points.

Full Video Transcript: Expand this section to read the entire transcript with timestamps.

Ask About a Specific Section: Enter start and end times (MM:SS or HH:MM:SS) and a question to query a particular segment.

Custom Tutor Instructions: Use the text area in "⚙️ Tutor Settings" to guide the AI's responses (e.g., "Explain like I'm five").

Ask a General Question: Type any question about the video's content.

Chat History: Review your past questions and the AI's answers.

Clear/Reset:

Click "🔄 Clear Current Video & Chat" to reset the application and start with a new video.

Click "Logout" in the sidebar to return to the authentication screen.

Important Notes
API Key Security: Never expose your GOOGLE_API_KEY directly in your code or commit it to public repositories. The .env file method is a secure way to manage it locally.

Firebase Simulation: The Firebase authentication in this version is a simulation for demonstration purposes. For a production-ready application, you would need to set up a proper Firebase project and integrate the client-side Firebase SDK (e.g., using st-firebase-auth or custom JavaScript components in Streamlit) to handle real user authentication and data persistence.

Transcript Availability: Not all YouTube videos have transcripts available, or they might only have auto-generated ones, which can vary in quality.

Model Usage: The application uses gemini-2.0-flash for general Q&A and summarization. Be mindful of API usage limits and costs if deploying this application.

Developed by: K.Pugazhmani