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
import PyPDF2 # Import PyPDF2 for PDF processing
from fpdf import FPDF # Import FPDF for PDF generation

import re # Import regex for time parsing
import io # For downloadable content

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

# --- Transcript Retrieval Functions ---

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
        st.error("Transcripts are disabled for this video. Please try another video.")
        print(f"DEBUG: TranscriptsDisabled for URL: {url}")
    except NoTranscriptFound:
        st.error("No transcript found for this video. Please ensure transcripts are available or try another video.")
        print(f"DEBUG: NoTranscriptFound for URL: {url}")
    except VideoUnavailable:
        st.error("This video is unavailable. Please check the URL or try another video.")
        print(f"DEBUG: VideoUnavailable for URL: {url}")
    except CouldNotRetrieveTranscript:
        st.error("Could not retrieve transcript. It may not be available in your region or due to other issues. Please try another video.")
        print(f"DEBUG: CouldNotRetrieveTranscript for URL: {url}")
    except Exception as e:
        st.error(f"An unexpected error occurred while getting the transcript options: {e}. Please check the URL or try another video.")
        print(f"DEBUG: Error in get_youtube_transcript_options: {e} for URL: {url}")
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

def get_pdf_text(pdf_file):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Handle pages with no extractable text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None
    return text

def generate_quiz(text_content, num_questions=3):
    """Generates a multiple-choice quiz from text content using LLM."""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
    quiz_prompt = f"""
    Generate a {num_questions}-question multiple-choice quiz based on the following text.
    For each question, provide 4 options (A, B, C, D) and clearly indicate the correct answer.
    Format the output clearly with questions and options.

    Text:
    {text_content[:4000]} # Limit text to avoid token limits
    """
    try:
        response = llm.invoke(quiz_prompt).content
        return response
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return "Could not generate quiz."

def create_pdf_from_text(text_content, title="Document"):
    """Creates a PDF from a string of text, handling UTF-8 characters."""
    pdf = FPDF()
    # Add a font that supports Unicode (like DejaVuSansCondensed)
    # You might need to place the .ttf font file in a 'fonts' directory in your project
    # and uncomment the line below. For basic characters, Arial might work, but for
    # special characters, a Unicode font is necessary.
    # pdf.add_font('DejaVuSansCondensed', '', 'fonts/DejaVuSansCondensed.ttf', uni=True)
    # pdf.set_font("DejaVuSansCondensed", size=12) # Use the added font
    
    # Using Arial for now, but be aware of its limitations with non-latin characters
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Add title
    pdf.set_font("Arial", 'B', 16)
    # Use write() for simple text that doesn't need wrapping, or multi_cell with encoding
    pdf.write(10, title.encode('latin1', 'replace').decode('latin1')) # Encode/decode to handle title characters
    pdf.ln(10) # Line break
    
    # Add content
    pdf.set_font("Arial", size=12)
    # FPDF's multi_cell handles UTF-8 if uni=True is set on add_font, and text is encoded
    # If not using a Unicode font, we need to encode to latin1 and replace unencodable chars
    
    # Process text in chunks and encode to latin1, replacing unencodable characters
    # This will prevent the UnicodeEncodeError but might show '?' for unsupported characters.
    # The ideal solution is to use a Unicode font as commented above.
    chunk_size = 2000 
    for i in range(0, len(text_content), chunk_size):
        chunk = text_content[i:i+chunk_size]
        # Encode to latin1 and replace characters that cannot be encoded
        encoded_chunk = chunk.encode('latin1', 'replace').decode('latin1')
        pdf.multi_cell(0, 10, encoded_chunk)
        pdf.ln(5) # Small line break between chunks
    
    return pdf.output(dest='S').encode('latin1') # Return as bytes, still using latin1 for output stream

# --- Streamlit UI ---
st.set_page_config(page_title="LectureLens AI", layout="centered") # Updated page title

st.title("📚 LectureLens AI: Your Smart YouTube Tutor & Document Analyst") # Updated main title
st.markdown(
    """
    Welcome to your personal AI tutor! Enter a YouTube video URL (preferably a lecture or educational video)
    to extract its transcript. Once processed, you can ask questions about the video's content,
    and the AI will provide answers based on the transcript.
    
    **New!** You can now also upload PDF documents and ask questions about their content.
    """
)

# Initialize session state variables
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
if 'pdf_qa_chain' not in st.session_state: # New: QA chain for PDF
    st.session_state.pdf_qa_chain = None
if 'pdf_text_processed' not in st.session_state: # New: Flag for PDF processing
    st.session_state.pdf_text_processed = False
if 'pdf_chat_history' not in st.session_state: # New: Separate chat history for PDF
    st.session_state.pdf_chat_history = []
if 'video_summary' not in st.session_state: # New: Store video summary
    st.session_state.video_summary = ""
if 'video_topics' not in st.session_state: # New: Store video topics
    st.session_state.video_topics = ""
if 'pdf_summary' not in st.session_state: # New: Store PDF summary
    st.session_state.pdf_summary = ""
if 'pdf_topics' not in st.session_state: # New: Store PDF topics
    st.session_state.pdf_topics = ""
if 'current_pdf_text' not in st.session_state: # New: Store current PDF text
    st.session_state.current_pdf_text = ""
if 'video_quiz' not in st.session_state: # New: Store video quiz
    st.session_state.video_quiz = ""
if 'pdf_quiz' not in st.session_state: # New: Store pdf quiz
    st.session_state.pdf_quiz = ""


# --- Tabbed Interface for Video vs. PDF ---
tab1, tab2 = st.tabs(["🎥 YouTube Video Tutor", "📄 Talk to PDF Document"])

with tab1:
    st.header("YouTube Video Analysis")
    # Input for YouTube video URL
    video_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=your_video_id", key="youtube_url_input")

    # Process Video Button (Lists available transcripts)
    if st.button("✨ Process Video Transcript", key="process_youtube_button"):
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
                    st.session_state.video_summary = "" # Clear summary
                    st.session_state.video_topics = "" # Clear topics
                    st.session_state.video_quiz = "" # Clear quiz
                    # Ensure PDF related states are reset when processing a new video
                    st.session_state.pdf_qa_chain = None
                    st.session_state.pdf_text_processed = False
                    st.session_state.pdf_chat_history = []
                    st.session_state.pdf_summary = ""
                    st.session_state.pdf_topics = ""
                    st.session_state.pdf_quiz = ""
                    st.session_state.current_pdf_text = ""
                else:
                    st.warning("Could not find any transcripts for this video or video ID could not be extracted.")
                    st.info("Please ensure the URL is correct and the video is publicly accessible without restrictions.")
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

        if st.button("▶️ Load Selected Transcript", key="load_transcript_button"):
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
                            st.session_state.video_summary = "" # Clear summary on new load
                            st.session_state.video_topics = "" # Clear topics on new load
                            st.session_state.video_quiz = "" # Clear quiz on new load
                            st.success(f"✅ {lang_options[selected_lang_code]} transcript loaded successfully! You can now ask questions.")
                        except Exception as e:
                            st.error(f"Error during Langchain processing: {e}. Please ensure your GOOGLE_API_KEY is valid and correctly configured.")
                    else:
                        st.warning(f"The retrieved {lang_options[selected_lang_code]} transcript was empty. Please try another language or video.")
                else:
                    st.warning(f"Could not retrieve {lang_options[selected_lang_code]} transcript content.")

    # Clear/Reset YouTube Data Button
    if st.session_state.video_processed or st.session_state.chat_history or st.session_state.available_languages:
        if st.button("🔄 Clear Current Video Data & Chat", key="clear_youtube_data_button"):
            st.session_state.qa_chain = None
            st.session_state.full_transcript_data = []
            st.session_state.chat_history = []
            st.session_state.video_processed = False
            st.session_state.video_id = None
            st.session_state.available_languages = {}
            st.session_state.selected_language_code = 'en'
            st.session_state.video_summary = ""
            st.session_state.video_topics = ""
            st.session_state.video_quiz = ""
            # No change to custom instructions or PDF data
            if os.path.exists("transcript.txt"):
                os.remove("transcript.txt")
            if os.path.exists("temp_section_transcript.txt"):
                os.remove("temp_section_transcript.txt")
            st.success("YouTube video data and chat reset.")

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
        col_sum1, col_sum2 = st.columns([0.7, 0.3])
        with col_sum1:
            if st.button("Generate Summary", key="generate_summary_button"):
                with st.spinner("Generating summary..."):
                    try:
                        full_text_for_summary = " ".join([item.text for item in st.session_state.full_transcript_data])
                        summary_prompt = f"Please provide a concise summary of the following text:\n\n{full_text_for_summary}"
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
                        summary = llm_summary.invoke(summary_prompt).content
                        st.session_state.video_summary = summary # Store summary
                        with st.expander("View Summary"):
                            st.info(f"**Summary:**\n{summary}")
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        with col_sum2:
            if st.session_state.video_summary:
                # Download button for PDF summary
                pdf_bytes = create_pdf_from_text(st.session_state.video_summary, "Video Summary")
                st.download_button(
                    label="Download Summary (PDF)",
                    data=pdf_bytes,
                    file_name="video_summary.pdf",
                    mime="application/pdf",
                    key="download_video_summary_pdf"
                )
        if st.session_state.video_summary and not col_sum1.button("Generate Summary", key="generate_summary_button_re"): # Only show if summary exists and button isn't clicked again
            with st.expander("View Stored Summary"):
                st.info(f"**Summary:**\n{st.session_state.video_summary}")


        st.markdown("---")

        # --- Topic Modeling / Keyphrase Extraction ---
        st.subheader("💡 Key Topics & Phrases")
        col_topic1, col_topic2 = st.columns([0.7, 0.3])
        with col_topic1:
            if st.button("Extract Topics & Phrases", key="extract_topics_button"):
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
                        st.session_state.video_topics = topics # Store topics
                        with st.expander("View Key Topics & Phrases"):
                            st.markdown(topics) # Use markdown as LLM output might be formatted
                    except Exception as e: 
                        st.error(f"Error extracting topics and phrases: {e}")
        with col_topic2:
            if st.session_state.video_topics:
                # Download button for PDF topics
                pdf_bytes = create_pdf_from_text(st.session_state.video_topics, "Video Key Topics")
                st.download_button(
                    label="Download Topics (PDF)",
                    data=pdf_bytes,
                    file_name="video_topics.pdf",
                    mime="application/pdf",
                    key="download_video_topics_pdf"
                )
        if st.session_state.video_topics and not col_topic1.button("Extract Topics & Phrases", key="extract_topics_button_re"): # Only show if topics exist and button isn't clicked again
            with st.expander("View Stored Key Topics & Phrases"):
                st.markdown(st.session_state.video_topics)


        st.markdown("---")

        # --- Quiz Generation ---
        st.subheader("🧠 Generate Quiz")
        col_quiz1, col_quiz2 = st.columns([0.7, 0.3])
        with col_quiz1:
            if st.button("Generate Quiz from Video", key="generate_video_quiz_button"):
                with st.spinner("Generating quiz..."):
                    video_full_text = " ".join([item.text for item in st.session_state.full_transcript_data])
                    quiz_content = generate_quiz(video_full_text)
                    st.session_state.video_quiz = quiz_content # Store the quiz
                    with st.expander("View Quiz"):
                        st.markdown(quiz_content)
        with col_quiz2:
            if st.session_state.get('video_quiz'):
                # Download button for PDF quiz
                pdf_bytes = create_pdf_from_text(st.session_state.video_quiz, "Video Quiz")
                st.download_button(
                    label="Download Quiz (PDF)",
                    data=pdf_bytes,
                    file_name="video_quiz.pdf",
                    mime="application/pdf",
                    key="download_video_quiz_pdf"
                )
        if st.session_state.get('video_quiz') and not col_quiz1.button("Generate Quiz from Video", key="generate_video_quiz_button_re"):
            with st.expander("View Stored Quiz"):
                st.markdown(st.session_state.video_quiz)


        st.markdown("---")

        # --- Transcript Viewer ---
        st.subheader("📖 Full Video Transcript")
        col_transcript1, col_transcript2 = st.columns([0.7, 0.3])
        with col_transcript1:
            with st.expander("Click to view transcript"):
                if st.session_state.full_transcript_data:
                    transcript_display_text = ""
                    for item in st.session_state.full_transcript_data:
                        minutes = int(item.start // 60)
                        seconds = int(item.start % 60)
                        transcript_display_text += f"[{minutes:02d}:{seconds:02d}] {item.text}\n"
                    st.text_area("Full Transcript", value=transcript_display_text, height=300, disabled=True)
                else:
                    st.write("Transcript not available.")
        with col_transcript2:
            if st.session_state.full_transcript_data:
                full_text_for_download = "\n".join([f"[{int(item.start // 60):02d}:{int(item.start % 60):02d}] {item.text}" for item in st.session_state.full_transcript_data])
                # Download button for PDF transcript
                pdf_bytes = create_pdf_from_text(full_text_for_download, "Video Transcript")
                st.download_button(
                    label="Download Transcript (PDF)",
                    data=pdf_bytes,
                    file_name="video_transcript.pdf",
                    mime="application/pdf",
                    key="download_video_transcript_pdf"
                )

        st.markdown("---")

        # --- Specific Section Q&A ---
        st.subheader("🔍 Ask About a Specific Section")
        col1, col2 = st.columns(2)
        with col1:
            start_time_str = st.text_input("Start Time (MM:SS or HH:MM:SS)", value="00:00", key="start_time")
        with col2:
            end_time_str = st.text_input("End Time (MM:SS or HH:MM:SS)", value="99:59", key="end_time") # Default to a large end time

        section_question = st.text_input("Ask a question about this specific section:", key="section_q")
        
        if st.button("Ask Section Question", key="ask_section_question_button"):
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
            chat_history_text = ""
            for i, chat in enumerate(st.session_state.chat_history):
                chat_history_text += f"Q{i+1}: {chat['question']}\n"
                chat_history_text += f"A{i+1}: {chat['answer']}\n---\n"
            st.text_area("Conversation Log", value=chat_history_text, height=300, disabled=True, key="video_chat_display")
            
            # Download button for PDF chat history
            pdf_bytes = create_pdf_from_text(chat_history_text, "Video Chat History")
            st.download_button(
                label="Download Chat History (PDF)",
                data=pdf_bytes,
                file_name="video_chat_history.pdf",
                mime="application/pdf",
                key="download_video_chat_pdf"
            )
        else:
            st.write("No chat history yet for this video.")

    else: # This 'else' correctly pairs with 'if st.session_state.video_processed:'
        st.info("⬆️ Enter a YouTube video URL and click 'Process Video Transcript' to begin.")
        if not st.session_state.available_languages: # Only show if no languages are listed yet
            st.info("After processing, you will be able to select the transcript language.")

with tab2:
    st.header("Talk to PDF Document")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", key="pdf_uploader")

    if uploaded_file is not None:
        if st.button("📚 Process PDF Document", key="process_pdf_button"):
            with st.spinner("Processing PDF... This may take a moment for large files."):
                pdf_text = get_pdf_text(uploaded_file)
                if pdf_text:
                    st.session_state.current_pdf_text = pdf_text # Store PDF text
                    # Reset video-related states when processing a new PDF
                    st.session_state.qa_chain = None
                    st.session_state.full_transcript_data = []
                    st.session_state.chat_history = []
                    st.session_state.video_processed = False
                    st.session_state.video_id = None
                    st.session_state.available_languages = {}
                    st.session_state.selected_language_code = 'en'
                    st.session_state.video_summary = ""
                    st.session_state.video_topics = ""
                    st.session_state.video_quiz = ""

                    # Process PDF text for QA
                    try:
                        # Langchain expects a list of Document objects, so we create one
                        from langchain_core.documents import Document
                        pdf_documents = [Document(page_content=pdf_text)]
                        
                        st.session_state.pdf_qa_chain = get_qa_chain(pdf_documents)
                        st.session_state.pdf_text_processed = True
                        st.session_state.pdf_chat_history = [] # Clear PDF chat history
                        st.session_state.pdf_summary = "" # Clear PDF summary
                        st.session_state.pdf_topics = "" # Clear PDF topics
                        st.session_state.pdf_quiz = "" # Clear PDF quiz
                        st.success("✅ PDF processed successfully! You can now ask questions about its content.")
                    except Exception as e:
                        st.error(f"Error during Langchain processing for PDF: {e}. Please ensure your GOOGLE_API_KEY is valid and correctly configured.")
                else:
                    st.warning("Could not extract text from the PDF. It might be an image-based PDF or corrupted.")
    
    # Clear PDF Data Button
    if st.session_state.pdf_text_processed or st.session_state.pdf_chat_history or st.session_state.current_pdf_text:
        if st.button("🔄 Clear Current PDF Data & Chat", key="clear_pdf_data_button"):
            st.session_state.pdf_qa_chain = None
            st.session_state.pdf_text_processed = False
            st.session_state.pdf_chat_history = []
            st.session_state.pdf_summary = ""
            st.session_state.pdf_topics = ""
            st.session_state.pdf_quiz = ""
            st.session_state.current_pdf_text = ""
            st.success("PDF data and chat reset.")

    st.markdown("---")

    if st.session_state.pdf_text_processed:
        # --- PDF Summarization Feature ---
        st.subheader("📝 PDF Summary")
        col_pdf_sum1, col_pdf_sum2 = st.columns([0.7, 0.3])
        with col_pdf_sum1:
            if st.button("Generate PDF Summary", key="generate_pdf_summary_button"):
                with st.spinner("Generating PDF summary..."):
                    try:
                        summary_prompt = f"Please provide a concise summary of the following text:\n\n{st.session_state.current_pdf_text}"
                        llm_summary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)
                        summary = llm_summary.invoke(summary_prompt).content
                        st.session_state.pdf_summary = summary # Store PDF summary
                        with st.expander("View PDF Summary"):
                            st.info(f"**Summary:**\n{summary}")
                    except Exception as e:
                        st.error(f"Error generating PDF summary: {e}")
        with col_pdf_sum2:
            if st.session_state.pdf_summary:
                # Download button for PDF summary
                pdf_bytes = create_pdf_from_text(st.session_state.pdf_summary, "PDF Summary")
                st.download_button(
                    label="Download PDF Summary (PDF)",
                    data=pdf_bytes,
                    file_name="pdf_summary.pdf",
                    mime="application/pdf",
                    key="download_pdf_summary_pdf"
                )
        if st.session_state.pdf_summary and not col_pdf_sum1.button("Generate PDF Summary", key="generate_pdf_summary_button_re"):
            with st.expander("View Stored PDF Summary"):
                st.info(f"**Summary:**\n{st.session_state.pdf_summary}")

        st.markdown("---")

        # --- PDF Topic Modeling / Keyphrase Extraction ---
        st.subheader("💡 PDF Key Topics & Phrases")
        col_pdf_topic1, col_pdf_topic2 = st.columns([0.7, 0.3])
        with col_pdf_topic1:
            if st.button("Extract PDF Topics & Phrases", key="extract_pdf_topics_button"):
                with st.spinner("Extracting key information from PDF..."):
                    try: 
                        topic_prompt = (
                            f"Analyze the following text and extract the most important key topics and phrases. "
                            f"Present them as a bulleted list. Limit to 5-10 key points.\n\n"
                            f"Text:\n{st.session_state.current_pdf_text}"
                        )
                        llm_topics = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
                        topics = llm_topics.invoke(topic_prompt).content
                        st.session_state.pdf_topics = topics # Store PDF topics
                        with st.expander("View PDF Key Topics & Phrases"):
                            st.markdown(topics) # Use markdown as LLM output might be formatted
                    except Exception as e: 
                        st.error(f"Error extracting PDF topics and phrases: {e}")
        with col_pdf_topic2:
            if st.session_state.pdf_topics:
                # Download button for PDF topics
                pdf_bytes = create_pdf_from_text(st.session_state.pdf_topics, "PDF Key Topics")
                st.download_button(
                    label="Download PDF Topics (PDF)",
                    data=pdf_bytes,
                    file_name="pdf_topics.pdf",
                    mime="application/pdf",
                    key="download_pdf_topics_pdf"
                )
        if st.session_state.pdf_topics and not col_pdf_topic1.button("Extract PDF Topics & Phrases", key="extract_pdf_topics_button_re"):
            with st.expander("View Stored PDF Key Topics & Phrases"):
                st.markdown(st.session_state.pdf_topics)

        st.markdown("---")

        # --- PDF Quiz Generation ---
        st.subheader("🧠 Generate Quiz from PDF")
        col_pdf_quiz1, col_pdf_quiz2 = st.columns([0.7, 0.3])
        with col_pdf_quiz1:
            if st.button("Generate Quiz from PDF", key="generate_pdf_quiz_button"):
                with st.spinner("Generating quiz from PDF..."):
                    quiz_content = generate_quiz(st.session_state.current_pdf_text)
                    st.session_state.pdf_quiz = quiz_content # Store the quiz
                    with st.expander("View Quiz"):
                        st.markdown(quiz_content)
        with col_pdf_quiz2:
            if st.session_state.get('pdf_quiz'):
                # Download button for PDF quiz
                pdf_bytes = create_pdf_from_text(st.session_state.pdf_quiz, "PDF Quiz")
                st.download_button(
                    label="Download Quiz (PDF)",
                    data=pdf_bytes,
                    file_name="pdf_quiz.pdf",
                    mime="application/pdf",
                    key="download_pdf_quiz_pdf"
                )
        if st.session_state.get('pdf_quiz') and not col_pdf_quiz1.button("Generate Quiz from PDF", key="generate_pdf_quiz_button_re"):
            with st.expander("View Stored Quiz"):
                st.markdown(st.session_state.pdf_quiz)

        st.markdown("---")

        # --- PDF Text Viewer ---
        st.subheader("📖 Full PDF Text")
        col_pdf_text1, col_pdf_text2 = st.columns([0.7, 0.3])
        with col_pdf_text1:
            with st.expander("Click to view PDF text"):
                if st.session_state.current_pdf_text:
                    st.text_area("Full PDF Text", value=st.session_state.current_pdf_text, height=300, disabled=True, key="pdf_text_display")
                else:
                    st.write("PDF text not available.")
        with col_pdf_text2:
            if st.session_state.current_pdf_text:
                # Download button for PDF text
                pdf_bytes = create_pdf_from_text(st.session_state.current_pdf_text, "Full PDF Text")
                st.download_button(
                    label="Download PDF Text (PDF)",
                    data=pdf_bytes,
                    file_name="full_pdf_text.pdf",
                    mime="application/pdf",
                    key="download_pdf_text_pdf"
                )

        st.markdown("---")

        st.subheader("❓ Ask a Question about the PDF")
        pdf_question = st.text_input("Type your question here:", placeholder="e.g., What is the main conclusion of this document?", key="pdf_general_q")
        if pdf_question:
            with st.spinner("Getting your answer from PDF..."):
                try:
                    # Apply custom instructions to PDF question
                    current_instructions = st.session_state.get('custom_instructions', '')
                    if current_instructions:
                        pdf_full_query = f"{current_instructions}\n\nQuestion: {pdf_question}"
                    else:
                        pdf_full_query = pdf_question

                    pdf_answer = st.session_state.pdf_qa_chain.run(pdf_full_query)
                    with st.expander("View PDF Answer"):
                        st.info(f"**Answer:** {pdf_answer}")
                    st.session_state.pdf_chat_history.append({"question": pdf_question, "answer": pdf_answer})
                except Exception as e:
                    st.error(f"Error getting answer from PDF: {e}. Please try again.")
        
        st.markdown("---")
        st.subheader("💬 PDF Chat History")
        if st.session_state.pdf_chat_history:
            pdf_chat_history_text = ""
            for i, chat in enumerate(st.session_state.pdf_chat_history):
                pdf_chat_history_text += f"Q{i+1}: {chat['question']}\n"
                pdf_chat_history_text += f"A{i+1}: {chat['answer']}\n---\n"
            st.text_area("Conversation Log (PDF)", value=pdf_chat_history_text, height=300, disabled=True, key="pdf_chat_display")
            
            # Download button for PDF chat history
            pdf_bytes = create_pdf_from_text(pdf_chat_history_text, "PDF Chat History")
            st.download_button(
                label="Download PDF Chat History (PDF)",
                data=pdf_bytes,
                file_name="pdf_chat_history.pdf",
                mime="application/pdf",
                key="download_pdf_chat_pdf"
            )
        else:
            st.write("No chat history yet for this PDF.")
    else:
        st.info("⬆️ Upload a PDF document and click 'Process PDF Document' to begin asking questions.")


# Add developer name at the bottom
st.markdown("---")
st.markdown("Developed by: **K.Pugazhmani**")
