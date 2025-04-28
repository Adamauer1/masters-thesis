import moviepy
import glob


image_files = sorted(glob.glob("images/song_2/my_gen/*.png"))

# Create a clip from the images
video = moviepy.ImageSequenceClip(image_files, fps=24)  # You can set fps (frames per second)
audio = moviepy.AudioFileClip("Data/audio_files/genre_set/classical.00000.wav")
video_with_audio = video.with_audio(audio)
# Write the video file
video_with_audio.write_videofile("final_video_my.mp4", codec="libx264", audio_codec="aac")