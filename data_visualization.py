import pandas as pd
import numpy as np

import os
import sys

import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import Audio
import cv2

audio_path = 'features/audio_features'

file_emotion = []
file_path = []
for subdir, dirs, files in os.walk(audio_path, topdown=True):
    for file in files:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append( os.path.join(subdir,file))

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
Ravdess_df.head()

plt.title('Count of Emotions', size=16)
sns.countplot(Ravdess_df.Emotions)
count = 0
for em in Ravdess_df.Emotions:
    if (em=='surprise'):
        count += 1
#print(count)
 
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()
        
def create_waveplot(data, sr, e):
    plt.figure(figsize=(10, 3.5))
    plt.title('Waveplot for audio: {} emotion'.format(e), size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()

def create_spectrogram(data, sr, e):
    # stft function converts the data into short term fourier transform
    X = librosa.stft(data)
    Xdb = librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(12, 4))
    plt.title('Spectrogram for audio: {} emotion'.format(e), size=20)
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')   
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

def feature_extractor_audio(X, sampling_rate, e):
    plt.figure(figsize=(10, 4))
    mfccs = librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc=30)
    plt.title('MFCCs for audio: {} emotion'.format(e), size=20)
    librosa.display.specshow(mfccs, sr=sampling_rate, x_axis='time', y_axis='hz') 
    plt.colorbar()  

    plt.figure(figsize=(12, 4))
    log_mel=librosa.power_to_db(librosa.feature.melspectrogram(X, sr=sampling_rate))
    plt.title('Log Mel-spectrogram for audio: {} emotion'.format(e), size=20)
    librosa.display.specshow(log_mel, sr=sampling_rate, x_axis='time', y_axis='hz')   
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar()

emotion='angry'
path = np.array(Ravdess_df.Path[Ravdess_df.Emotions==emotion])[1]
data, sampling_rate = librosa.load(path, sr = 22050*2, duration = 2.5, offset = 1.0)

length = librosa.get_duration(data)
if (librosa.get_duration(data) < 5.0):
    pad = 5.0 -(length)
    data = np.concatenate([data, np.zeros(int(pad*sampling_rate/2))])

if (librosa.get_duration(data) > 5.0):
    data = data[0:5*sampling_rate]

create_waveplot(data, sampling_rate, emotion)
create_spectrogram(data, sampling_rate, emotion)
Audio(path)
feature_extractor_audio(data, sampling_rate, emotion)

def pitch_l(data):
    n_steps = -1
    x_pitch_l1 = librosa.effects.pitch_shift(
        data, sampling_rate, n_steps)
    return x_pitch_l1

def shift(data):
    s_range = int(np.random.uniform(low=-90, high=90)*500)
    x_shifting = np.roll(data, s_range)
    return x_shifting

def noise(data):
    noise = 0.035*np.random.uniform()*np.amax(data)
    x_noise = data.astype('float64') + noise * \
        np.random.normal(size=data.shape[0])
    return x_noise

x = noise(data)
create_waveplot(x, sampling_rate, "with random noise "+emotion)
Audio(x, rate=sampling_rate)

x = pitch_l(data)
create_waveplot(x, sampling_rate, "changed pitch "+ emotion)
Audio(x, rate=sampling_rate)

x = shift(data)
create_waveplot(x, sampling_rate, "shifted " + emotion)
Audio(x, rate=sampling_rate)


video_path = 'features/video_features'

video_file_emotion = []
video_file_path = []
for subdir, dirs, files in os.walk(video_path, topdown=True):
    for file in files:     
        if file[0:2] == '02': # use only video files without audio
            video_part = file.split('.')[0]
            video_part = video_part.split('-')
            # third part in each file represents the emotion associated to that file.
            video_file_emotion.append(int(video_part[2]))
            video_file_path.append( os.path.join(subdir,file))

# dataframe for emotion of files
video_emotion_df = pd.DataFrame(video_file_emotion, columns=['Emotions'])
# dataframe for path of files.
video_path_df = pd.DataFrame(video_file_path, columns=['Path'])
video_Ravdess_df = pd.concat([video_emotion_df, video_path_df], axis=1)

video_Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
video_Ravdess_df.head()


plt.title('Count of Emotions', size=16)
sns.countplot(video_Ravdess_df.Emotions)

video_count = 0
for em in video_Ravdess_df.Emotions:
    if (em=='disgust'):
        video_count += 1
#print(video_count)

 
plt.ylabel('Count', size=12)
plt.xlabel('Emotions', size=12)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.show()


emotion='angry'
path = np.array(video_Ravdess_df.Path[video_Ravdess_df.Emotions==emotion])[1]

v_cap = cv2.VideoCapture(path)
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

frame_list= np.linspace(1, v_len-1, 16+1).astype(int)

frames = []
for fn in range(v_len):
    success, img = v_cap.read() 
    if success is False :
        continue

    if (fn in frame_list):
    
        # Load classifier
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # Detect faces
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face = face_cascade.detectMultiScale(gray, 1.1, 5)
                            
        # Draw rectangle around the faces
        for (x, y, w, h) in face:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
                                
        img = img[y:y+h, x:x+w]
        frames.append(img)

    #plot images
fig = plt.figure(figsize=(30, 2))
for idx in np.arange(16): 

            ax = fig.add_subplot(2, 16/2, idx+1, xticks = [], yticks = [])
            frames[idx] = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
            ax.imshow(frames[idx])
plt.show()