import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError # Import the specific error

import re # Import regex for time parsing

# --- Load Environment Variables (for local development) ---
# This will load variables from .env file if it exists locally.
load_dotenv()

# --- IMPORTANT: Set your Google API Key (for Gemini) ---
# Prioritize Streamlit secrets for deployment, fallback to os.getenv for local.
google_api_key = None # Initialize to None

try:
    # Attempt to get API key from Streamlit secrets (for Streamlit Cloud deployment)
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError): # Catch both KeyError and the specific Streamlit error
    # Fallback to os.getenv for local development (reads from .env)
    google_api_key = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    # Catch any other unexpected errors during secret loading
    st.error(f"An unexpected error occurred while loading API key: {e}")
    st.stop()


# Ensure the API key is set in the environment for Langchain
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    st.error(
        "🚨 Google API Key not found! "
        "Please set it in your `.env` file (for local) or Streamlit Cloud Secrets (for deployment)."
    )
    st.stop() # Stop the app if API key is missing

# --- Firebase Initialization (Conceptual - as before) ---
# Note: In a real production Streamlit app, you'd typically use a service account key file
# for firebase_admin or integrate client-side Firebase JS SDK for auth.
# For this environment, we'll manage auth state via session_state
# and assume the Canvas handles the actual Firebase calls based on the token.
# The `firebase_admin` SDK is typically for server-side operations.
# We'll use a placeholder for `firebase_admin.initialize_app` as it's not directly compatible
# with the client-side auth token provided.
# No changes needed here from previous version, just a reminder of its conceptual nature.

def get_youtube_transcript_options(url):
    """
    Fetches available transcripts for a YouTube video.
    Returns the video_id and a dictionary mapping language code to transcript object data.
    Handles various exceptions related to transcript retrieval.
    """
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        available_transcripts = {}
        for transcript in transcript_list:
            available_transcripts[transcript.language_code] = {
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'transcript_object': transcript # Store the actual transcript object
            }
        return video_id, available_transcripts
    except TranscriptsDisabled:
        st.error("Transcripts are disabled for this video.")
    except NoTranscriptFound:
        st.error("No transcript found for this video. Please ensure English transcripts are available.")
    except VideoUnavailable:
        st.error("This video is unavailable. Please check the URL.")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region or due to other issues.")
    except Exception as e:
        st.error(f"An unexpected error occurred while getting the transcript options: {e}")
    return None, {} # Return None for video_id and empty dict on error

def fetch_specific_transcript(transcript_object):
    """Fetches the actual text content (list of snippet objects) of a given transcript object."""
    try:
        transcript_data = transcript_object.fetch()
        return transcript_data
    except Exception as e:
        st.error(f"Error fetching specific transcript content: {e}")
        return []

def save_transcript_to_file(text, filename="transcript.txt"):
    """
    Saves the fetched transcript text to a local file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    except IOError as e:
        st.error(f"Error saving transcript to file: {e}")

def parse_time_to_seconds(time_str):
    """Parses MM:SS or HH:MM:SS format to total seconds."""
    parts = list(map(int, time_str.split(':')))
    if len(parts) == 2: # MM:SS
        return parts[0] * 60 + parts[1]
    elif len(parts) == 3: # HH:MM:SS
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return 0

def get_qa_chain(documents):
    """Initializes and returns the RetrievalQA chain."""
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# --- Streamlit UI ---
st.set_page_config(page_title="LectureLens AI", layout="centered") # Updated page title

st.title("📚 LectureLens AI: Your Smart YouTube Tutor") # Updated main title
st.markdown(
    """
    Welcome to your personal AI tutor! Enter a YouTube video URL (preferably a lecture or educational video)
    to extract its transcript. Once processed, you can ask questions about the video's content,
    and the AI will provide answers based on the transcript.
    """
)

# Initialize session state variables for authentication
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user_email' not in st.session_state:
    st.session_state.current_user_email = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False # To toggle between login and signup forms

# Initialize other session state variables if they don't exist
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'full_transcript_data' not in st.session_state:
    st.session_state.full_transcript_data = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'video_id' not in st.session_state:
    st.session_state.video_id = None
if 'available_languages' not in st.session_state:
    st.session_state.available_languages = {}
if 'selected_language_code' not in st.session_state:
    st.session_state.selected_language_code = 'en' # Default to English
if 'custom_instructions' not in st.session_state:
    st.session_state.custom_instructions = "Act as a helpful and patient tutor. Explain complex topics clearly."


# --- Authentication Functions (Simulated for Streamlit's stateless nature) ---
# In a real deployed app, these would interact with Firebase Auth directly.
# Here, we're using placeholders to demonstrate the UI flow.
def simulate_signup(email, password):
    """Simulates user sign-up."""
    try:
        # In a real app, you'd call auth.create_user_with_email_and_password()
        # For this environment, we'll just set session state
        st.session_state.logged_in = True
        st.session_state.current_user_email = email
        st.success(f"Account created and logged in as {email}!")
        return True
    except Exception as e:
        st.error(f"Sign-up failed: {e}")
        return False

def simulate_login(email, password):
    """Simulates user login."""
    try:
        # In a real app, you'd call auth.sign_in_with_email_and_password()
        # For this environment, we'll just set session state
        st.session_state.logged_in = True
        st.session_state.current_user_email = email
        st.success(f"Logged in as {email}!")
        return True
    except Exception as e:
        st.error(f"Login failed: {e}")
        return False

def logout_user():
    """Logs out the current user."""
    st.session_state.logged_in = False
    st.session_state.current_user_email = None
    # Clear all app-specific session state data on logout
    st.session_state.qa_chain = None
    st.session_state.full_transcript_data = []
    st.session_state.chat_history = []
    st.session_state.video_processed = False
    st.session_state.video_id = None
    st.session_state.available_languages = {}
    st.session_state.selected_language_code = 'en'
    st.session_state.custom_instructions = "Act as a helpful and patient tutor. Explain complex topics clearly."
    if os.path.exists("transcript.txt"):
        os.remove("transcript.txt")
    if os.path.exists("temp_section_transcript.txt"):
        os.remove("temp_section_transcript.txt")
    st.info("Logged out successfully!")

# --- Authentication UI ---
if not st.session_state.logged_in:
    st.sidebar.title("Authentication")
    if st.session_state.show_signup:
        st.sidebar.subheader("Sign Up")
        new_email = st.sidebar.text_input("Email", key="signup_email")
        new_password = st.sidebar.text_input("Password", type="password", key="signup_password")
        if st.sidebar.button("Create Account"):
            if new_email and new_password:
                simulate_signup(new_email, new_password)
            else:
                st.sidebar.error("Please enter both email and password.")
        st.sidebar.markdown("Already have an account? [Login](#login)")
        if st.sidebar.button("Go to Login", key="go_to_login_button"):
            st.session_state.show_signup = False
            st.rerun() # Rerun to switch to login form
    else:
        st.sidebar.subheader("Login")
        email = st.sidebar.text_input("Email", key="login_email")
        password = st.sidebar.text_input("Password", type="password", key="login_password")
        if st.sidebar.button("Login"):
            if email and password:
                simulate_login(email, password)
            else:
                st.sidebar.error("Please enter both email and password.")
        st.sidebar.markdown("Don't have an account? [Sign Up](#signup)")
        if st.sidebar.button("Go to Sign Up", key="go_to_signup_button"):
            st.session_state.show_signup = True
            st.rerun() # Rerun to switch to signup form

    st.stop() # Stop execution if not logged in
else: # This 'else' correctly pairs with 'if not st.session_state.logged_in:'
    # --- Main Application Content (only visible when logged in) ---
    st.sidebar.write(f"Welcome, {st.session_state.current_user_email}!")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun() # Rerun to show login page

    # Input for YouTube video URL
    video_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=your_video_id")

    # Process Video Button (Lists available transcripts)
    if st.button("✨ Process Video Transcript"):
        if video_url:
            with st.spinner("Fetching available transcripts..."):
                video_id, available_transcripts = get_youtube_transcript_options(video_url)
                if video_id and available_transcripts:
                    st.session_state.video_id = video_id # Store video ID
                    st.session_state.available_languages = available_transcripts
                    
                    # Try to pre-select English, otherwise select the first available
                    if 'en' in available_transcripts:
                        st.session_state.selected_language_code = 'en'
                    elif available_transcripts:
                        st.session_state.selected_language_code = list(available_transcripts.keys())[0]

                    st.success("Transcripts listed. Select a language below and click 'Load Transcript'.")
                    st.session_state.video_processed = False # Mark as not fully processed yet for QA
                    st.session_state.qa_chain = None # Reset QA chain
                    st.session_state.chat_history = [] # Clear chat history
                else:
                    st.warning("Could not find any transcripts for this video or video ID could not be extracted.")
        else:
            st.warning("⚠️ Please enter a valid YouTube URL to proceed.")

    # --- Language Selection and Load Transcript Button ---
    if st.session_state.available_languages:
        st.markdown("---")
        st.subheader("🌐 Select Transcript Language")
        
        lang_options = {code: f"{data['language']} ({'Generated' if data['is_generated'] else 'Manual'})" 
                        for code, data in st.session_state.available_languages.items()}
        
        selected_lang_code = st.selectbox(
            "Choose a language:",
            options=list(lang_options.keys()),
            format_func=lambda x: lang_options[x],
            key="lang_selector",
            index=list(lang_options.keys()).index(st.session_state.selected_language_code) if st.session_state.selected_language_code in lang_options else 0
        )
        st.session_state.selected_language_code = selected_lang_code

        if st.button("▶️ Load Selected Transcript"):
            with st.spinner(f"Loading {lang_options[selected_lang_code]} transcript..."):
                selected_transcript_obj = st.session_state.available_languages[selected_lang_code]['transcript_object']
                transcript_data = fetch_specific_transcript(selected_transcript_obj)
                
                if transcript_data:
                    st.session_state.full_transcript_data = transcript_data
                    full_text = " ".join([item.text for item in transcript_data])
                    save_transcript_to_file(full_text)

                    if full_text.strip():
                        try:
                            loader = TextLoader("transcript.txt", encoding="utf-8")
                            documents = loader.load()
                            st.session_state.qa_chain = get_qa_chain(documents)
                            st.session_state.video_processed = True
                            st.session_state.chat_history = [] # Clear history for new transcript
                            st.success(f"✅ {lang_options[selected_lang_code]} transcript loaded successfully! You can now ask questions.")
                        except Exception as e:
                            st.error(f"Error during Langchain processing: {e}. Please ensure your GOOGLE_API_KEY is valid and correctly configured.")
                else:
                    st.warning(f"The retrieved {lang_options[selected_lang_code]} transcript was empty. Please try another language or video.")
        else:
                st.warning(f"Could not retrieve {lang_options[selected_lang_code]} transcript content.")

    # Clear/Reset Button
    if st.session_state.video_processed or st.session_state.chat_history or st.session_state.available_languages:
        if st.button("🔄 Clear Current Video & Chat"):
            st.session_state.qa_chain = None
            st.session_state.full_transcript_data = []
            st.session_state.chat_history = []
            st.session_state.video_processed = False
            st.session_state.video_id = None
            st.session_state.available_languages = {}
            st.session_state.selected_language_code = 'en'
            st.session_state.custom_instructions = "Act as a helpful and patient tutor. Explain complex topics clearly."

            if os.path.exists("transcript.txt"):
                os.remove("transcript.txt")
            if os.path.exists("temp_section_transcript.txt"):
                os.remove("temp_section_transcript.txt")
            st.success("Application reset. Enter a new video URL.")

    st.markdown("---")

    # Conditional UI elements after video is processed
    if st.session_state.video_processed:
        # --- Video Player Embedding ---
        if st.session_state.get('video_id'):
            st.subheader("▶️ Video Player")
            st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")
            st.markdown("---")

        # --- Summarization Feature ---
        st.subheader("📝 Video Summary")
        if st.button("Generate Summary"):
            with st.spinner("Generating summary..."):
                try:
                    full_text_for_summary = " ".join([item.text for item in st.session_state.full_transcript_data])
                    summary_prompt = f"Please provide a concise summary of the following text:\n\n{full_text_for_summary}"
                    llm_summary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
                    summary = llm_summary.invoke(summary_prompt).content
                    with st.expander("View Summary"):
                        st.info(f"**Summary:**\n{summary}")
                except Exception as e:
                    st.error(f"Error generating summary: {e}")

        st.markdown("---")

        # --- Topic Modeling / Keyphrase Extraction ---
        st.subheader("💡 Key Topics & Phrases")
        if st.button("Extract Topics & Phrases"):
            with st.spinner("Extracting key information..."):
                try:
                    full_text_for_topics = " ".join([item.text for item in st.session_state.full_transcript_data])
                    
                    # Prompt for topic extraction
                    topic_prompt = (
                        f"Analyze the following text and extract the most important key topics and phrases. "
                        f"Present them as a bulleted list. Limit to 5-10 key points.\n\n"
                        f"Text:\n{full_text_for_topics}"
                    )
                    
                    llm_topics = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
                    topics = llm_topics.invoke(topic_prompt).content
                    
                    with st.expander("View Key Topics & Phrases"):
                        st.markdown(topics) # Use markdown as LLM output might be formatted
                except Exception as e:
                    st.error(f"Error extracting topics and phrases: {e}")

        st.markdown("---")

        # --- Transcript Viewer ---
        st.subheader("📖 Full Video Transcript")
        with st.expander("Click to view transcript"):
            if st.session_state.full_transcript_data:
                # Display transcript with timestamps
                for item in st.session_state.full_transcript_data:
                    minutes = int(item.start // 60)
                    seconds = int(item.start % 60)
                    st.write(f"**[{minutes:02d}:{seconds:02d}]** {item.text}")
            else:
                st.write("Transcript not available.")

        st.markdown("---")

        # --- Specific Section Q&A ---
        st.subheader("🔍 Ask About a Specific Section")
        col1, col2 = st.columns(2)
        with col1:
            start_time_str = st.text_input("Start Time (MM:SS or HH:MM:SS)", value="00:00", key="start_time")
        with col2:
            end_time_str = st.text_input("End Time (MM:SS or HH:MM:SS)", value="99:59", key="end_time") # Default to a large end time

        section_question = st.text_input("Ask a question about this specific section:", key="section_q")
        
        if st.button("Ask Section Question"):
            if section_question:
                start_seconds = parse_time_to_seconds(start_time_str)
                end_seconds = parse_time_to_seconds(end_time_str)

                if start_seconds >= end_seconds:
                    st.error("Start time must be before end time.")
                else:
                    filtered_transcript_text = []
                    for item in st.session_state.full_transcript_data:
                        snippet_start = item.start
                        snippet_end = item.start + item.duration
                        
                        if (snippet_start >= start_seconds and snippet_start < end_seconds) or \
                           (snippet_end > start_seconds and snippet_end <= end_seconds) or \
                           (start_seconds >= snippet_start and end_seconds <= snippet_end):
                            filtered_transcript_text.append(item.text)

                    if filtered_transcript_text:
                        section_text = " ".join(filtered_transcript_text)
                        save_transcript_to_file(section_text, "temp_section_transcript.txt")
                        
                        try:
                            section_documents = TextLoader("temp_section_transcript.txt", encoding="utf-8").load()
                            section_qa_chain = get_qa_chain(section_documents)
                            
                            # Apply custom instructions to section question
                            current_instructions = st.session_state.get('custom_instructions', '')
                            if current_instructions:
                                section_full_query = f"{current_instructions}\n\nQuestion: {section_question}"
                            else:
                                section_full_query = section_question

                            with st.spinner("Getting answer for section..."):
                                section_answer = section_qa_chain.run(section_full_query)
                                with st.expander(f"View Answer for section [{start_time_str}-{end_time_str}]"):
                                    st.info(f"**Answer:** {section_answer}")
                                st.session_state.chat_history.append({"question": f"[Section {start_time_str}-{end_time_str}] {section_question}", "answer": section_answer})
                        except Exception as e:
                            st.error(f"Error processing section question: {e}")
                        finally:
                            if os.path.exists("temp_section_transcript.txt"):
                                os.remove("temp_section_transcript.txt")
                    else:
                        st.warning("No transcript found for the specified time range. Please adjust the times.")
            else:
                st.warning("Please enter a question for the specific section.")

        st.markdown("---")

        # --- Custom Prompts/Personality ---
        st.subheader("⚙️ Tutor Settings")
        st.session_state.custom_instructions = st.text_area(
            "Custom Tutor Instructions (Optional):",
            value=st.session_state.custom_instructions, # Use session state value
            help="Provide instructions on how the AI should answer, e.g., 'Explain like I'm five', 'Be very detailed', 'Focus on practical applications'."
        )
        st.markdown("---")

        # --- Ask General Question (main Q&A) ---
        st.subheader("❓ Ask a General Question")
        user_question = st.text_input("Type your question here:", placeholder="e.g., What is the main topic of this lecture?", key="general_q")
        if user_question:
            with st.spinner("Getting your answer..."):
                try:
                    # Apply custom instructions to general question
                    current_instructions = st.session_state.get('custom_instructions', '')
                    if current_instructions:
                        full_query = f"{current_instructions}\n\nQuestion: {user_question}"
                    else:
                        full_query = user_question

                    answer = st.session_state.qa_chain.run(full_query)
                    with st.expander("View Answer"):
                        st.info(f"**Answer:** {answer}")
                    st.session_state.chat_history.append({"question": user_question, "answer": answer})
                except Exception as e:
                    st.error(f"Error getting answer: {e}. Please try again.")

        st.markdown("---")

        # --- Chat History Display ---
        st.subheader("💬 Chat History")
        if st.session_state.chat_history:
            for i, chat in enumerate(st.session_state.chat_history):
                st.markdown(f"**Q{i+1}:** {chat['question']}")
                st.markdown(f"**A{i+1}:** {chat['answer']}")
                st.markdown("---")
        else:
            st.write("No chat history yet for this video.")

    else: # This 'else' correctly pairs with 'if st.session_state.video_processed:'
        st.info("⬆️ Enter a YouTube video URL and click 'Process Video Transcript' to begin.")
        if not st.session_state.available_languages: # Only show if no languages are listed yet
            st.info("After processing, you will be able to select the transcript language.")

# Add developer name at the bottom
st.markdown("---")
st.markdown("Developed by: **K.Pugazhmani**")
