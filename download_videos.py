import pandas as pd
import yt_dlp as youtube_dl
import os

categories_of_interest = ['sleeping']

num_videos=100

# Set up paths
base_output_folder = 'kinetics_videos'
if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder)

# Load train.csv (metadata file)
csv_path = 'train.csv'  # Change this to the path of your CSV file
data = pd.read_csv(csv_path)

# Filter the dataframe for Falling and Faceplanting categories
filtered_data = data[data['label'].isin(categories_of_interest)]

# Get Youtube IDs for the categories (assuming 'youtube_id' column has the video IDs)
video_ids = filtered_data['youtube_id'].tolist() [:num_videos] # Limit to first 150 videos
time_starts = filtered_data['time_start'].tolist() [:num_videos] # Corresponding start times
time_ends = filtered_data['time_end'].tolist()[:num_videos]  # Corresponding end times
labels = filtered_data['label'].tolist()  [:num_videos]# Corresponding labels
print(video_ids)



# Download videos using yt-dlp
def download_video(youtube_id, start_time, end_time, label, base_output_folder):
    try:
        # URL of the video
        video_url = f'https://www.youtube.com/watch?v={youtube_id}'
        
        # Create folder for the label if it doesn't exist
        label_folder = os.path.join(base_output_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        # Output filename pattern
        output_path = os.path.join(label_folder, f'{youtube_id}.mp4')
        
        # YT-DLP options (video-only download, force MP4 format)
        ydl_opts = {
            'outtmpl': output_path,  # Save video with the YouTube ID as part of the filename
            'format': 'bestvideo[ext=mp4]',  # Force mp4 format
            'noplaylist': True,  # Don't download playlists
            'postprocessors': [],
            'postprocessor_args': [
                '-ss', str(start_time),  # Start time
                '-to', str(end_time),  # End time
            ],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])  # Download the video
        print(f"Downloaded video from {video_url} from {start_time} to {end_time} into {label_folder}")

    except Exception as e:
        print(f"Error downloading {youtube_id}: {e}")

# Download the videos
for video_id, start_time, end_time, label in zip(video_ids, time_starts, time_ends, labels):
    download_video(video_id, start_time, end_time, label, base_output_folder)

print("Video download completed!")
