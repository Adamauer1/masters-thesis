import moviepy
import glob


def render_video(song, audio_file):
    image_files = sorted(glob.glob(f"images/{song}/my_gen/*.png"))
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_my_gen.mp4", codec="libx264", audio_codec="aac")
    
    image_files = sorted(glob.glob(f"images/{song}/paper_gen/*.png"))
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    #audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_paper_gen.mp4", codec="libx264", audio_codec="aac")
    
    image_files = sorted(glob.glob(f"images/{song}/random_gen/*.png"))
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    #audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_random_gen.mp4", codec="libx264", audio_codec="aac")

    image_files = sorted(glob.glob(f"images/{song}/random_2_gen/*.png"))
    video = moviepy.ImageSequenceClip(image_files, fps=24)  
    #audio = moviepy.AudioFileClip(audio_file)
    video_with_audio = video.with_audio(audio)
    video_with_audio.write_videofile(f"videos/{song}/{song}_random_2_gen.mp4", codec="libx264", audio_codec="aac")
        
        
#"Data/audio_files/genre_set/classical.00000.wav"
render_video(song="song_10", audio_file="Data/audio_files/genre_set/rock.00000.wav")

