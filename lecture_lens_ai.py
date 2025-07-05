import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from pytube import YouTube
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError
import re

# Load environment variables
load_dotenv()

# --- Google API Key ---
google_api_key = None
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except (KeyError, StreamlitSecretNotFoundError):
    google_api_key = os.getenv("GOOGLE_API_KEY")
except Exception as e:
    st.error(f"Unexpected error loading API key: {e}")
    st.stop()

if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    st.error("🚨 Google API Key not found! Add it to your `.env` or Streamlit secrets.")
    st.stop()


# --- Helper Functions ---

def get_youtube_transcript_options(url):
    try:
        video_id = YouTube(url).video_id
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return video_id, {
            t.language_code: {
                'language': t.language,
                'language_code': t.language_code,
                'is_generated': t.is_generated,
                'transcript_object': t
            } for t in transcript_list
        }
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, CouldNotRetrieveTranscript) as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Unexpected error: {e}")
    return None, {}

def fetch_specific_transcript(transcript_object):
    try:
        return transcript_object.fetch()
    except Exception as e:
        st.error(f"Transcript fetch error: {e}")
        return []

def save_transcript_to_file(text, filename="transcript.txt"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)
    except IOError as e:
        st.error(f"File save error: {e}")

def parse_time_to_seconds(time_str):
    parts = list(map(int, time_str.split(':')))
    return sum(x * 60**i for i, x in enumerate(reversed(parts)))

def get_qa_chain(documents):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5)
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)


# --- UI Setup ---

st.set_page_config(page_title="LectureLens AI", layout="centered")
st.title("📚 LectureLens AI: Your Smart YouTube Tutor")

st.markdown("""
Welcome to your personal AI tutor!  
Enter a YouTube video URL to extract its transcript.  
Once processed, you can ask questions about the content, generate summaries, and more!
""")

# --- Session State Init ---

for key, default in {
    'qa_chain': None,
    'full_transcript_data': [],
    'chat_history': [],
    'video_processed': False,
    'video_id': None,
    'available_languages': {},
    'selected_language_code': 'en',
    'custom_instructions': "Act as a helpful and patient tutor. Explain complex topics clearly."
}.items():
    st.session_state.setdefault(key, default)

# --- YouTube Video Input ---

video_url = st.text_input("🔗 Enter YouTube Video URL:", placeholder="e.g., https://www.youtube.com/watch?v=xyz")

if st.button("✨ Process Video Transcript"):
    if video_url:
        with st.spinner("Fetching transcripts..."):
            video_id, transcripts = get_youtube_transcript_options(video_url)
            if video_id and transcripts:
                st.session_state.video_id = video_id
                st.session_state.available_languages = transcripts
                st.session_state.selected_language_code = 'en' if 'en' in transcripts else list(transcripts.keys())[0]
                st.success("Transcripts listed! Select a language below and click 'Load Transcript'.")
                st.session_state.video_processed = False
                st.session_state.qa_chain = None
                st.session_state.chat_history = []
            else:
                st.warning("No transcripts found or invalid video URL.")
    else:
        st.warning("Please enter a valid YouTube URL.")

# --- Language Selection ---

if st.session_state.available_languages:
    st.markdown("---")
    st.subheader("🌐 Select Transcript Language")

    lang_options = {
        code: f"{data['language']} ({'Generated' if data['is_generated'] else 'Manual'})"
        for code, data in st.session_state.available_languages.items()
    }

    selected_code = st.selectbox(
        "Choose a language:",
        options=list(lang_options.keys()),
        format_func=lambda x: lang_options[x],
        index=list(lang_options.keys()).index(st.session_state.selected_language_code)
    )
    st.session_state.selected_language_code = selected_code

    if st.button("▶️ Load Selected Transcript"):
        with st.spinner(f"Loading {lang_options[selected_code]}..."):
            transcript_data = fetch_specific_transcript(
                st.session_state.available_languages[selected_code]['transcript_object']
            )
            if transcript_data:
                st.session_state.full_transcript_data = transcript_data
                full_text = " ".join([item.text for item in transcript_data])
                save_transcript_to_file(full_text)

                if full_text.strip():
                    try:
                        documents = TextLoader("transcript.txt", encoding="utf-8").load()
                        st.session_state.qa_chain = get_qa_chain(documents)
                        st.session_state.video_processed = True
                        st.success("✅ Transcript loaded! Ask your questions below.")
                    except Exception as e:
                        st.error(f"LangChain processing error: {e}")
                else:
                    st.warning("Transcript was empty.")
            else:
                st.warning("Could not load the transcript content.")

# --- Reset Button ---

if any([st.session_state.video_processed, st.session_state.chat_history, st.session_state.available_languages]):
    if st.button("🔄 Clear Current Video & Chat"):
        for key in [
            'qa_chain', 'full_transcript_data', 'chat_history', 'video_processed',
            'video_id', 'available_languages', 'selected_language_code'
        ]:
            st.session_state[key] = [] if isinstance(st.session_state[key], list) else None
        st.session_state.selected_language_code = 'en'
        st.session_state.custom_instructions = "Act as a helpful and patient tutor. Explain complex topics clearly."
        for fname in ["transcript.txt", "temp_section_transcript.txt"]:
            if os.path.exists(fname):
                os.remove(fname)
        st.success("Reset complete.")

# --- Post-Transcript Features ---

if st.session_state.video_processed:
    st.subheader("▶️ Video Player")
    st.video(f"https://www.youtube.com/watch?v={st.session_state.video_id}")

    st.markdown("---")
    st.subheader("📝 Video Summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                text = " ".join([item.text for item in st.session_state.full_transcript_data])
                prompt = f"Summarize the following text:\n\n{text}"
                summary = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7).invoke(prompt).content
                with st.expander("View Summary"):
                    st.info(summary)
            except Exception as e:
                st.error(f"Summary error: {e}")

    st.subheader("💡 Key Topics & Phrases")
    if st.button("Extract Topics & Phrases"):
        with st.spinner("Extracting..."):
            try:
                text = " ".join([item.text for item in st.session_state.full_transcript_data])
                prompt = (
                    "Extract 5–10 key topics and phrases from the text below. Return as a bullet list:\n\n" + text
                )
                topics = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.5).invoke(prompt).content
                with st.expander("View Key Topics"):
                    st.markdown(topics)
            except Exception as e:
                st.error(f"Topic extraction error: {e}")

    st.subheader("📖 Full Transcript")
    with st.expander("Click to view transcript"):
        for item in st.session_state.full_transcript_data:
            m, s = divmod(int(item.start), 60)
            st.write(f"**[{m:02d}:{s:02d}]** {item.text}")

    st.subheader("🔍 Ask About a Specific Section")
    col1, col2 = st.columns(2)
    with col1:
        start = st.text_input("Start Time (MM:SS or HH:MM:SS)", "00:00")
    with col2:
        end = st.text_input("End Time (MM:SS or HH:MM:SS)", "99:59")

    section_q = st.text_input("Ask a section question:")
    if st.button("Ask Section Question") and section_q:
        start_sec = parse_time_to_seconds(start)
        end_sec = parse_time_to_seconds(end)
        if start_sec >= end_sec:
            st.error("Start must be before end.")
        else:
            section_texts = [
                item.text for item in st.session_state.full_transcript_data
                if start_sec <= item.start <= end_sec
            ]
            if section_texts:
                temp_text = " ".join(section_texts)
                save_transcript_to_file(temp_text, "temp_section_transcript.txt")
                try:
                    docs = TextLoader("temp_section_transcript.txt", encoding="utf-8").load()
                    section_chain = get_qa_chain(docs)
                    query = f"{st.session_state.custom_instructions}\n\nQuestion: {section_q}"
                    with st.spinner("Getting answer..."):
                        answer = section_chain.run(query)
                        with st.expander(f"Answer for [{start}-{end}]"):
                            st.info(answer)
                        st.session_state.chat_history.append({"question": f"[{start}-{end}] {section_q}", "answer": answer})
                except Exception as e:
                    st.error(f"Section Q&A error: {e}")
                finally:
                    os.remove("temp_section_transcript.txt")
            else:
                st.warning("No transcript in selected range.")

    st.subheader("⚙️ Tutor Settings")
    st.session_state.custom_instructions = st.text_area(
        "Customize Tutor Instructions:",
        value=st.session_state.custom_instructions,
        help="e.g., 'Explain like I’m five', 'Use examples', etc."
    )

    st.subheader("❓ Ask a General Question")
    general_q = st.text_input("Ask your question here:")
    if general_q:
        with st.spinner("Answering..."):
            try:
                prompt = f"{st.session_state.custom_instructions}\n\nQuestion: {general_q}"
                answer = st.session_state.qa_chain.run(prompt)
                with st.expander("View Answer"):
                    st.info(answer)
                st.session_state.chat_history.append({"question": general_q, "answer": answer})
            except Exception as e:
                st.error(f"General Q&A error: {e}")

    st.subheader("💬 Chat History")
    for i, chat in enumerate(st.session_state.chat_history):
        st.markdown(f"**Q{i+1}:** {chat['question']}")
        st.markdown(f"**A{i+1}:** {chat['answer']}")
        st.markdown("---")
else:
    st.info("⬆️ Enter a video URL and click 'Process Video Transcript' to get started.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed by: **K.Pugazhmani**")
