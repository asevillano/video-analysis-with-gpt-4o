# Import libraries
import streamlit as st
import cv2
import os
import time
import json
from dotenv import load_dotenv
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
from openai import AzureOpenAI
import base64
from mimetypes import guess_type
import yt_dlp
from yt_dlp.utils import download_range_func

# Default configuration
SEGMENT_DURATION = 0 # In seconds, Set to 0 to not split the video
SYSTEM_PROMPT = "You are a helpful assistant that describes in detail a video. Response in the same language than the transcription."
USER_PROMPT = "These are the frames from the video."

# Load configuration
load_dotenv(override=True)

# Configuration of OpenAI GPT-4o
aoai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
aoai_apikey = os.environ["AZURE_OPENAI_API_KEY"]
aoai_model_name = os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"]
print(f'aoai_endpoint: {aoai_endpoint}, aoai_model_name: {aoai_model_name}')
# Create AOAI client for answer generation
aoai_client = AzureOpenAI(
    azure_deployment=aoai_model_name,
    api_version='2024-02-15-preview',
    azure_endpoint=aoai_endpoint,
    api_key=aoai_apikey
)

# Configuration of Whisper
whisper_endpoint = os.environ["WHISPER_ENDPOINT"]
whisper_apikey = os.environ["WHISPER_API_KEY"]
whisper_model_name = os.environ["WHISPER_DEPLOYMENT_NAME"]
# Create AOAI client for whisper
whisper_client = AzureOpenAI(
    api_version='2024-02-01',
    azure_endpoint=whisper_endpoint,
    api_key=whisper_apikey
)

# Function to encode a local video into frames
def process_video(video_path, seconds_per_frame=2):
    base64Frames = []

    # Prepare the video analysis
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    curr_frame=0

    # Loop through the video and extract frames at specified sampling rate
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        height, width, _ = frame.shape
        frame = cv2.resize(frame, (width // 2, height // 2))

        _, buffer = cv2.imencode(".jpg", frame)

        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip
    video.release()
    print(f"Extracted {len(base64Frames)} frames")
    
    return base64Frames

# Function to transcript the audio from the local video with Whisper
def process_audio(video_path):
    base_video_path, _ = os.path.splitext(video_path)
    audio_path = f"{base_video_path}.mp3"
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, bitrate="32k")
    clip.audio.close()
    clip.close()
    print(f"Extracted audio to {audio_path}")

    # Transcribe the audio
    transcription = whisper_client.audio.transcriptions.create(
        model=whisper_model_name,
        file=open(audio_path, "rb"),
    )
    print("Transcript: ", transcription.text + "\n\n")
        
    return transcription

# Function to analyze the video with GPT-4o
def analyze_video(base64frames, system_prompt, user_prompt, transcription):

    try:
        if transcription != '': # Include the audio transcription
            response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "These are the frames from the video.",},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                        {"type": "text", "text": f"The audio transcription is: {transcription.text}"},
                        # *question_messages
                    ]}
                ],
                temperature=0.5,
                max_tokens=4096
            )
        else: # Without the audio transcription
            response = aoai_client.chat.completions.create(
                model=aoai_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "These are the frames from the video.",},
                    {"role": "user", "content": [
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f'data:image/jpg;base64,{x}', "detail": "auto"}}, base64frames),
                    ]}
                ],
                temperature=0.5,
                max_tokens=4096
            )

        json_response = json.loads(response.model_dump_json())
        #print(f'RESPONSE: [{response.model_dump_json(indent=2)}]')
        response = json_response['choices'][0]['message']['content']

    except Exception as ex:
        print(f'ERROR: {ex}')
        response = f'ERROR: {ex}'

    return response

# Split the video in segments of N seconds (by default 3 minutes). If segment_length is 0 the full video is processed
def split_video(video_path, output_dir, segment_length=180):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    if segment_length == 0: # Do not split
        segment_length = int(duration)

    for start_time in range(0, int(duration), segment_length):
        end_time = min(start_time + segment_length, duration)
        output_file = os.path.join(output_dir, f'segment_{start_time}-{end_time}_secs.mp4')
        ffmpeg_extract_subclip(video_path, start_time, end_time, targetname=output_file)
        yield output_file

# Process the video
def execute_video_processing(st, segment_path, system_prompt, user_prompt):
    # Show the video on the screen
    st.write(f"Video: {segment_path}:")
    st.video(segment_path)

    with st.spinner(f"Analyzing segment: {segment_path}"):
        # Extract 1 frame per second. Adjust the `seconds_per_frame` parameter to change the sampling rate
        with st.spinner(f"Extracting frames..."):
            inicio = time.time()
            base64frames = process_video(segment_path, seconds_per_frame=1)
            fin = time.time()
            print(f'\\t>>>> Frames extraction took {(fin - inicio):.3f} seconds <<<<')
            st.write(f'Extracted {len(base64frames)} frames in {(fin - inicio):.3f} seconds')

        # Extract the transcription of the audio
        if audio_transcription:
            with st.spinner(f"Transcribing audio from video file..."):
                inicio = time.time()
                transcription = process_audio(segment_path)
                fin = time.time()
                st.write(f'Transcription finished in {(fin - inicio):.3f} seconds')
                print(f'\t>>>> Audio transcription took {(fin - inicio):.3f} seconds <<<<')
        else:
            transcription = ''
        # Analyze the video frames and the audio transcription with GPT-4o
        with st.spinner(f"Analyzing frames and audio with GPT-4o..."):
            inicio = time.time()
            analysis = analyze_video(base64frames, system_prompt, user_prompt, transcription)
            fin = time.time()
            print(f'\t>>>> Analysys with GPT-4o took {(fin - inicio):.3f} seconds <<<<')

    st.write(f"**Analysis of segment {segment_path}** ({(fin - inicio):.3f} seconds)")
    fin = time.time()
    print(f'\t>>>> {(fin - inicio):.6f} segundos <<<<')
    st.success("Analysis completed.")

    return analysis

# Streamlit User Interface
st.set_page_config(
    page_title="Video Analysis with GPT-4o",
    layout="centered",
    initial_sidebar_state="auto",
)
st.image("microsoft.png", width=100)
st.title('Video Analysis with GPT-4o')

with st.sidebar:
    file_or_url = st.selectbox("Video source:", ["File", "URL"], index=0, help="Select the source, file or url")
    audio_transcription = st.checkbox('Transcript audio', True, help="Extract the audio transcription and use in the analysis or not")
    seconds = int(st.text_input('Number of seconds to split the video (0 to not split)', str(SEGMENT_DURATION)))
    system_prompt = st.text_area('System Prompt to analyze the video', SYSTEM_PROMPT)
    user_prompt = st.text_area('User Prompt to analyze the video', USER_PROMPT)

# Prepare the segment directory
output_dir = "segments"
os.makedirs(output_dir, exist_ok=True)

# Video file or Video URL
if file_or_url == 'File':
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
else:
    url = st.text_area("Enter de url:", value='https://www.youtube.com/watch?v=CI9djzS3Ld0', height=10)
    continuous_transmision = st.checkbox('Continuous transmision', False, help="Video of a continuous transmision")

# Analyze the video when the button is pressed
if st.button("Analize video", use_container_width=True, type='primary'):
    if file_or_url == 'URL': # Process Youtube video
        st.write(f'Analyzing video from url {url}...')
        
        ydl_opts = {
                'format': 'best',
                'outtmpl': 'segment_%(start)s.mp4',
                'force_keyframes_at_cuts': True,
        }
        ydl = yt_dlp.YoutubeDL(ydl_opts)
        if continuous_transmision == False:
            info_dict = ydl.extract_info(url, download=False)
            video_duration = info_dict.get('duration', 0)
        else:
            video_duration = 48*60*60
        
        if seconds == 0:
            duracion_segmento=video_duration
        else:
            duracion_segmento=seconds #SEGMENT_DURATION
        
        for start in range(0, video_duration, duracion_segmento):
            end = start + duracion_segmento
            filename = f'segments/segment_{start}.mp4'
            with st.spinner(f"Downloading video from second {start} to {end}..."):
                ydl_opts['outtmpl']['default'] = filename
                ydl_opts['download_ranges'] = download_range_func(None, [(start, end)])

                print(f'start: {start}, video_duration: {video_duration}, duracion_segmento: {duracion_segmento}')
                try:
                    ydl.download([url])
                except:
                    break

            segment_path = filename
            print(f"Segment downloaded: {segment_path}")

            # Process the video segment
            analysis = execute_video_processing(st, segment_path, system_prompt, user_prompt)
            st.write(f"{analysis}")

            event="guitarra eléctrica"
            if event in analysis:
                st.write(f'**Detected event "{event}" in segment {segment_path}**')
            
            # Delete the video segment
            os.remove(segment_path)

    else: # Process the fideo file
        if video_file is not None:
            video_path = os.path.join("temp", video_file.name)
        with open(video_path, "wb") as f:
            f.write(video_file.getbuffer())

        # Show the video on the screen
        #st.write("The full video:")
        #st.video(video_path)

        # Splitting video in segment of N seconds
        for segment_path in split_video(video_path, output_dir, seconds):
            # Process the video segment
            analysis = execute_video_processing(st, segment_path, system_prompt, user_prompt)
            st.write(f"{analysis}")

            # Delete the video segment
            os.remove(segment_path)

