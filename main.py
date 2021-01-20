import sys
from audio_model import select_train_or_test_audio
from video_model import select_train_or_test_video
from audio_video_model import select_train_or_test_audio_and_video

if __name__ == "__main__":

    model = sys.argv[1]

    if model == "audio_model":
        select_train_or_test_audio()
    if model == "video_model":
        select_train_or_test_video() 
    if model == "audio_video_model":
        select_train_or_test_audio_and_video()

