# emotion-recognition-RAVDESS

The RAVDESS dataset files can be downloaded at the following link: https://zenodo.org/record/1188976#.YAiBZnqg-Uk 

*)Create the 'features' folder with the following subfolders:
  - 'audio_features' in which the audio files of the RAVDESS dataset have to be inserted
  - 'video_features' in which the video files of the RAVDESS dataset have to be inserted
  - 'temp_features'

**) Download the PrivateTest_model.t7 file from https://drive.google.com/file/d/1Oy_9YmpkSKX1Q8jkOhJbz3Mc7qjyISzU/view and place it in the 'models' folder

To extract audio and video features and prepare datasets: python main_preprocessing.py <path_to_audio> <path_to_video> <temporary_directory_path>

To train or test the model: python main.py

MODELS: 
audio_model for the audio model 
video_model for the video model 
audio_video_model for the model that uses audio and video features

experiments_and_results contains the checkpoints obtained during the learning phase (can be used to test models without having to retrain networks)
