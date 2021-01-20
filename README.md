# emotion-recognition-RAVDESS

I file del dataset RAVDESS sono scaricabili al seguente link: https://zenodo.org/record/1188976#.YAiBZnqg-Uk 

*) Creare la cartella 'features' con le seguenti sottocartelle:
  - 'audio_features' in cui vanno inseriti i file audio del dataset RAVDESS
  - 'video_features' in cui vanno inseriti i file video del dataset RAVDESS
  - 'temp_features'

**) Scaricare il file PrivateTest_model.t7 da https://drive.google.com/file/d/1Oy_9YmpkSKX1Q8jkOhJbz3Mc7qjyISzU/view e inserirlo nella cartella 'models'


Per estrarre le feature audio e video e preparare i dataset: python main_preprocessing.py <path_to_audio> <path_to_video> <temporary_directory_path>
Per addestrare o testare il modello: python main.py <model>

MODELLI: 
  audio_model per il modello audio
  video_model per il modello video
  audio_video_model per il modello che utilizza feature audio e video
  
 experiments_and_results contiene i checkpoint ottenuti durante la fase di apprendimento (possibile utilizzarli per testare i modelli senza dover riaddestrare le reti)
