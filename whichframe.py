import streamlit as st
import cv2
from PIL import Image
import clip as openai_clip
import torch
import math
from humanfriendly import format_timespan
import numpy as np
import time
import os
import yt_dlp
import io

EXAMPLE_URL = "https://www.youtube.com/watch?v=zTvJJnoWIPk"
CACHED_DATA_PATH = "cached_data/"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = openai_clip.load("ViT-B/32", device=device)

def fetch_video(url):
    try:
        ydl_opts = {
            'format': 'bestvideo[height<=360][ext=mp4][vcodec=avc1]/best[height<=360][ext=mp4]',
            'quiet': True,
            'no_warnings': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)
            if duration >= 300:  # 5 minutes
                st.error("Please find a YouTube video shorter than 5 minutes.")
                st.stop()
            video_url = info['url']
            return None, video_url
            
    except Exception as e:
        st.error(f"Error fetching video: {str(e)}")
        st.error("Try another YouTube video or check if the URL is correct.")
        st.stop()

def extract_frames(video, status_text, progress_bar):
    cap = cv2.VideoCapture(video)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps/2))
    total_frames = frame_count // step
    frame_indices = []
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(i)
            
            current_frame = len(frames)
            status_text.text(f'Extracting frames... ({min(current_frame, total_frames)}/{total_frames})')
            progress = min(current_frame / total_frames, 1.0)
            progress_bar.progress(progress)
    
    cap.release()
    return frames, fps, frame_indices

def encode_frames(video_frames, status_text):
    batch_size = 256
    batches = math.ceil(len(video_frames) / batch_size)
    video_features = torch.empty([0, 512], dtype=torch.float32).to(device)
    
    for i in range(batches):
        batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features = batch_features.float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)
        video_features = torch.cat((video_features, batch_features))
        status_text.text(f'Encoding frames... ({(i+1)*batch_size}/{len(video_frames)})')
    
    return video_features

def img_to_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def get_youtube_timestamp_url(url, frame_idx, frame_indices):
    frame_count = frame_indices[frame_idx]
    fps = st.session_state.fps
    seconds = frame_count / fps
    seconds_rounded = int(seconds)
    
    if url == EXAMPLE_URL:
        video_id = "zTvJJnoWIPk"
    else:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query)['v'][0]
        except:
            return None, None
    
    return f"https://youtu.be/{video_id}?t={seconds_rounded}", seconds

def display_results(best_photo_idx, video_frames):
    st.subheader("Top 10 Results")
    for frame_id in best_photo_idx:
        result = video_frames[frame_id]
        st.image(result, width=400)
        
        timestamp_url, seconds = get_youtube_timestamp_url(st.session_state.url, frame_id, st.session_state.frame_indices)
        if timestamp_url:
            st.markdown(f"[▶️ Play video at {format_timespan(int(seconds))}]({timestamp_url})")

def text_search(search_query, video_features, video_frames, display_results_count=10):
    display_results_count = min(display_results_count, len(video_frames))
    
    with torch.no_grad():
        text_tokens = openai_clip.tokenize(search_query).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    video_features = video_features.float()
    
    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames)

def image_search(query_image, video_features, video_frames, display_results_count=10):
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(query_image)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    video_features = video_features.float()
    
    similarities = (100.0 * video_features @ image_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames)

def text_and_image_search(search_query, query_image, video_features, video_frames, display_results_count=10):
    with torch.no_grad():
        text_tokens = openai_clip.tokenize(search_query).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(query_image)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    combined_features = (text_features + image_features) / 2
    
    video_features = video_features.float()
    similarities = (100.0 * video_features @ combined_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames)

def load_cached_data(url):
    if url == EXAMPLE_URL:
        try:
            video_frames = np.load(f"{CACHED_DATA_PATH}example_frames.npy", allow_pickle=True)
            video_features = torch.load(f"{CACHED_DATA_PATH}example_features.pt")
            fps = np.load(f"{CACHED_DATA_PATH}example_fps.npy")
            frame_indices = np.load(f"{CACHED_DATA_PATH}example_frame_indices.npy")
            return video_frames, video_features, fps, frame_indices
        except:
            return None, None, None, None
    return None, None, None, None

def save_cached_data(url, video_frames, video_features, fps, frame_indices):
    if url == EXAMPLE_URL:
        os.makedirs(CACHED_DATA_PATH, exist_ok=True)
        np.save(f"{CACHED_DATA_PATH}example_frames.npy", video_frames)
        torch.save(video_features, f"{CACHED_DATA_PATH}example_features.pt")
        np.save(f"{CACHED_DATA_PATH}example_fps.npy", fps)
        np.save(f"{CACHED_DATA_PATH}example_frame_indices.npy", frame_indices)

def clear_cached_data():
    if os.path.exists(CACHED_DATA_PATH):
        try:
            for file in os.listdir(CACHED_DATA_PATH):
                file_path = os.path.join(CACHED_DATA_PATH, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(CACHED_DATA_PATH)
        except Exception as e:
            print(f"Error clearing cache: {e}")

st.set_page_config(page_title="Which Frame? 🎞️🔍", page_icon = "🔍", layout = "centered", initial_sidebar_state = "collapsed")

hide_streamlit_style = """
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
* {
    font-family: Avenir;
}
.block-container {
    max-width: 800px;
    padding: 2rem 1rem;
}
.stTextInput input {
    border-radius: 8px;
    border: 1px solid #E0E0E0;
    padding: 0.75rem;
    font-size: 1rem;
}
.stRadio [role="radiogroup"] {
    background: #F8F8F8;
    padding: 1rem;
    border-radius: 12px;
}
h1 {text-align: center;}
.css-gma2qf {display: flex; justify-content: center; font-size: 36px; font-weight: bold;}
a:link {text-decoration: none;}
a:hover {text-decoration: none;}
.st-ba {font-family: Avenir;}
.st-button {text-align: center;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if 'progress' not in st.session_state:
    st.session_state.progress = 1
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = None
if 'video_features' not in st.session_state:
    st.session_state.video_features = None
if 'fps' not in st.session_state:
    st.session_state.fps = None
if 'video_name' not in st.session_state:
    st.session_state.video_name = 'videos/example.mp4'

st.title("Which Frame? 🎞️🔍")
st.markdown("""
Search a video semantically. For example, which frame has "a person with sunglasses"?
Search using text, images, or a mix of text + image. WhichFrame uses [CLIP](https://github.com/openai/CLIP) for zero-shot frame classification.
""")

if 'url' not in st.session_state:
    st.session_state.url = ''

url = st.text_input("Enter a YouTube URL (e.g., https://www.youtube.com/watch?v=zTvJJnoWIPk)", key="url_input")

if st.button("Process Video"):
    if not url:
        st.error("Please enter a YouTube URL first")
    else:
        try:
            cached_frames, cached_features, cached_fps, cached_frame_indices = load_cached_data(url)
            
            if cached_frames is not None:
                st.session_state.video_frames = cached_frames
                st.session_state.video_features = cached_features
                st.session_state.fps = cached_fps
                st.session_state.frame_indices = cached_frame_indices
                st.session_state.url = url
                st.session_state.progress = 2
                st.success("Loaded cached video data!")
            else:
                with st.spinner('Fetching video...'):
                    video, video_url = fetch_video(url)
                    st.session_state.url = url
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Extract frames
                st.session_state.video_frames, st.session_state.fps, st.session_state.frame_indices = extract_frames(video_url, status_text, progress_bar)
                
                # Encode frames
                st.session_state.video_features = encode_frames(st.session_state.video_frames, status_text)
                
                save_cached_data(url, st.session_state.video_frames, st.session_state.video_features, st.session_state.fps, st.session_state.frame_indices)
                status_text.text('Finalizing...')
                st.session_state.progress = 2
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                st.success("Video processed successfully!")
                
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

if st.session_state.progress == 2:
    search_type = st.radio("Search Method", ["Text Search", "Image Search", "Text + Image Search"], index=0)
    
    if search_type == "Text Search":  # Text Search
        text_query = st.text_input("Type a search query (e.g., 'red car' or 'person with sunglasses')")
        if st.button("Search"):
            if not text_query:
                st.error("Please enter a search query first")
            else:
                text_search(text_query, st.session_state.video_features, st.session_state.video_frames)
    elif search_type == "Image Search":  # Image Search
        uploaded_file = st.file_uploader("Upload a query image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, caption="Query Image", width=200)
        if st.button("Search"):
            if uploaded_file is None:
                st.error("Please upload an image first")
            else:
                image_search(query_image, st.session_state.video_features, st.session_state.video_frames)
    else:  # Text + Image Search
        text_query = st.text_input("Type a search query")
        uploaded_file = st.file_uploader("Upload a query image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, caption="Query Image", width=200)
        
        if st.button("Search"):
            if not text_query or uploaded_file is None:
                st.error("Please provide both text query and image")
            else:
                text_and_image_search(text_query, query_image, st.session_state.video_features, st.session_state.video_frames)

st.markdown("---")
st.markdown(
    "By [David Chuan-En Lin](https://chuanenlin.com/). "
    "Play with the code at [https://github.com/chuanenlin/whichframe](https://github.com/chuanenlin/whichframe)."
)
