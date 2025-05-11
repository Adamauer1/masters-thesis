import moviepy
import glob
import random


def render_video(song, audio_file):
    # image_files = sorted(glob.glob(f"images/test/*.png"))
    # video = moviepy.ImageSequenceClip(image_files, fps=24)  
    # audio = moviepy.AudioFileClip(audio_file)
    # video_with_audio = video.with_audio(audio)
    # video_with_audio.write_videofile(f"test.mp4", codec="libx264", audio_codec="aac")
    image_files = sorted(glob.glob(f"images/{song}/my_gen/*.png"))
    #random.shuffle(image_files)
    #image_files.reverse()
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_my_gen.mp4", codec="libx264", audio_codec="aac")
    
    image_files = sorted(glob.glob(f"images/{song}/paper_gen/*.png"))
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_paper_gen.mp4", codec="libx264", audio_codec="aac")
    
    image_files = sorted(glob.glob(f"images/{song}/random_gen/*.png"))
    image_files.reverse()
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_random_gen.mp4", codec="libx264", audio_codec="aac")

    image_files = sorted(glob.glob(f"images/{song}/random_2_gen/*.png"))
    image_files.reverse()
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_random_2_gen.mp4", codec="libx264", audio_codec="aac")
        
        
#"Data/audio_files/genre_set/classical.00000.wav"
render_video(song="song_2", audio_file="audio/classical_30_secs.mp3")
render_video(song="song_3", audio_file="audio/country_30_secs.mp3")
render_video(song="song_4", audio_file="audio/disco_30_secs.mp3")
render_video(song="song_5", audio_file="audio/hiphop_30_secs.mp3")
render_video(song="song_6", audio_file="audio/jazz_30_secs.mp3")
render_video(song="song_7", audio_file="audio/metal_30_secs.mp3")
render_video(song="song_8", audio_file="audio/pop_30_secs.mp3")
render_video(song="song_9", audio_file="audio/reggae_30_secs.mp3")
render_video(song="song_10", audio_file="audio/rock_30_secs.mp3")

