import sys
from feature_extractor import preprocess
from train_val_set import train_test_val_split
from match_audio_video import match_audio_video


if __name__ == "__main__":
    audio_dataset = sys.argv[1]
    video_dataset = sys.argv[2]
    tmpdir = sys.argv[3]

    preprocess(audio_dataset, video_dataset, tmpdir)
    train_test_val_split(tmpdir)
    match_audio_video(tmpdir)
