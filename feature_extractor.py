import os
import librosa
import numpy as np
import cv2
import matplotlib.pyplot as plt
import librosa.display
from scipy import ndimage
import torch
import sklearn
import skimage as sk
from skimage import transform
from torchvision import transforms
from skimage import util
from skimage import exposure

import pandas as pd
import librosa.display
import seaborn as sns

from models import *

def preprocess(audio_path, video_path, featuredir):
    
    aud_path = featuredir + '/' + 'aud_features/'
    vid_path = featuredir + '/' + 'vid_features/'
    make_vid = False
    make_aud = False

    if not os.path.exists(vid_path):
        os.mkdir(vid_path)
        make_vid = True
    
    if not os.path.exists(aud_path):
        os.mkdir(aud_path)
        make_aud = True

    if (not make_aud and not make_vid):
        return
    
    countSurAud = 0
    countDisAud = 0

    countSurVid = 0
    countDisVid = 0

    
    for subdir, dirs, files in os.walk(audio_path, topdown=True):
        for file in files:
            try:
                X, sample_rate = librosa.load(os.path.join(subdir,file), sr = 22050*2, duration = 2.5, offset = 1.0)

                #Make every audio length = 5sec
                length = librosa.get_duration(X)
                if (length < 5.0):
                    pad = 5.0 -(length)
                    X = np.concatenate([X, np.zeros(int(pad*sample_rate/2))])
                if (length > 5.0):
                    X = X[0:5*sample_rate]

                ext_feature = feature_extractor_audio(X, sample_rate)
                features = np.asarray(ext_feature)
            
                #DATA AUGMENTATION 
                #Pitch tuning
                n_steps = 1
                x_pitch = librosa.effects.pitch_shift(X, sample_rate, n_steps)
                ext_feature_pitch = feature_extractor_audio(x_pitch, sample_rate)
                feature_array = np.asarray(ext_feature_pitch_r1)
                
                #Random shifting
                s_range = int(np.random.uniform(low = -90, high = 90)*500)
                x_shifting = np.roll(X, s_range)
                ext_feature_shifting = feature_extractor_audio(x_shifting, sample_rate)
                feature_array_s = np.asarray(ext_feature_shifting)     
                
                #Adding white noise
                noise = 0.035*np.random.uniform()*np.amax(X)
                x_noise = X.astype('float64') + noise * np.random.normal(size=X.shape[0])
                ext_feature_noise = feature_extractor_audio(x_noise, sample_rate)
                feature_array_n = np.asarray(ext_feature_noise)


                if (file[6:8] == '07'): 
                    countSurAud += 1 
                    
                if (file[6:8] == '08'):
                    countDisAud += 1

            
                if (file[6:8] == '01') or ((file[6:8] == '07') and (countSurAud < 185)) or ((file[6:8] == '08') and (countDisAud < 185)):
                    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
                    X = np.roll(X, shift_range)

                    n_steps = 1
                    added_x_pitch = librosa.effects.pitch_shift(X, sample_rate, n_steps)
                    added_ext_feature_pitch = feature_extractor_audio(added_x_pitch_r1, sample_rate)
                    added_feature_array_r1 = np.asarray(added_ext_feature_pitch_r1)
                    np.save(aud_path + file[0:-4] + "-added_pitch_r1", added_feature_array_r1)
                    
                    n_steps = -1
                    added_x_pitch_l1 = librosa.effects.pitch_shift(X, sample_rate, n_steps)
                    added_ext_feature_pitch_l1 = feature_extractor_audio(added_x_pitch_l1, sample_rate)
                    added_feature_array_l1 = np.asarray(added_ext_feature_pitch_l1)
                    np.save(aud_path + file[0:-4] + "-added_pitch_l1", added_feature_array_l1)

                    added_x_shifting = np.roll(X, s_range)
                    added_ext_feature_shifting = feature_extractor_audio(added_x_shifting, sample_rate)
                    added_feature_array_s = np.asarray(added_ext_feature_shifting)    
                    np.save(aud_path + file[0:-4] + "-added_shift", added_feature_array_s)

                    added_x_noise = X.astype('float64') + noise * np.random.normal(size=X.shape[0])
                    added_ext_feature_noise = feature_extractor_audio(added_x_noise, sample_rate)
                    added_feature_array_n = np.asarray(added_ext_feature_noise)
                    np.save(aud_path + file[0:-4] + "-added_noise", added_feature_array_n)
                    
                
                np.save(aud_path + file[0:-4] + "-pitch_r1", feature_array_r1)
                np.save(aud_path + file[0:-4] + "-noise", feature_array_n)
                np.save(aud_path + file[0:-4] + "-shift", feature_array_s)
                np.save(aud_path + file[0:-4], features)
      
            except ValueError:
                continue  

    audio_path = 'features/temp_features/aud_features'

    file_emotion = []
    file_path = []
    for subdir, dirs, files in os.walk(audio_path, topdown=True):
        for file in files:
            part = file.split('.')[0]
            part = part.split('-')
            file_emotion.append(int(part[2]))
            file_path.append( os.path.join(subdir,file))

    # dataframe for emotion of files
    emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])
    # dataframe for path of files.
    path_df = pd.DataFrame(file_path, columns=['Path'])
    Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

    # changing integers to actual emotions.
    Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    Ravdess_df.head()

    plt.title('Count of Emotions', size=16)
    sns.countplot(Ravdess_df.Emotions)

    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()
        
    
    for subdir, dirs, files in os.walk(video_path):
        model = VGG('VGG19')

        checkpoint = torch.load(os.path.join('models/PrivateTest_model.t7'))
        model.load_state_dict(checkpoint['net'])

        for file in files:        
            if file[0:2] == '02': # video files without audio

                    file_path = os.path.join(subdir, file)
                    v_cap = cv2.VideoCapture(file_path)
                    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    #print(v_len)

                    frame_list= np.linspace(1, v_len-1, 32+1).astype(int)

                    images = []
                    images_flip = []
                    images_noise = []
                    images_contrast = []
                    images_gamma = []
                    added_images = []
                    added_images_noise = []
                    added_images_flip = []
                    added_images_contrast = []
                    added_images_gamma = []


                    train_transformer = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                    ])

                    if (file[6:8] == '07'): 
                        countSurVid += 1             
                    if (file[6:8] == '08'):
                        countDisVid += 1
 
                    for fn in range(v_len):
                        success, img = v_cap.read() 
                        
                        if success is False :
                            continue
                        if (fn in frame_list) and len(images) != 32:

                                # Load classifier
                                face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

                                # Detect faces
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                face = face_cascade.detectMultiScale(gray, 1.1, 5)
                            
                                # Draw rectangle around the faces
                                for (x, y, w, h) in face:
                                    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 1)
                                
                                img = img[y:y+h, x:x+w]
                                img = cv2.resize(img, (48, 48))  
                                image = train_transformer(img)
                                image = image.type(torch.FloatTensor)
                                image = image.unsqueeze(0)
                                output_image = model(image)

                                images.append(output_image.detach().numpy())
                                
                                #AUGMENTATION

                                #noise
                                img_noise = sk.util.random_noise(img, mode = 'poisson')
                                img_noise = (255*img_noise).astype(np.uint8)

                                image_noise = train_transformer(img_noise)
                                image_noise = image_noise.type(torch.FloatTensor)
                                image_noise = image_noise.unsqueeze(0)
                                output_image_noise = model(image_noise)
                                images_noise.append(output_image_noise.detach().numpy())
                                

                                #change contrast
                                v_min, v_max = np.percentile(img, (0.2, 99.8))
                                image_contrast = exposure.rescale_intensity(img, in_range=(v_min, v_max))

                                image_contrast = train_transformer(image_contrast)
                                image_contrast = image_contrast.type(torch.FloatTensor)
                                image_contrast = image_contrast.unsqueeze(0)
                                output_image_contrast = model(image_contrast)

                                images_contrast.append(output_image_contrast.detach().numpy())
                                
                                #change gamma
                                image_gamma = exposure.adjust_gamma(img, gamma=0.4, gain=0.9)
                                
                                image_gamma = train_transformer(image_gamma)
                                image_gamma = image_gamma.type(torch.FloatTensor)
                                image_gamma = image_gamma.unsqueeze(0)
                                output_image_gamma = model(image_gamma)

                                images_gamma.append(output_image_gamma.detach().numpy())        

                            
                                if (file[6:8] == '01') or ((file[6:8] == '07') and (countSurVid < 185)) or ((file[6:8] == '08') and (countDisVid < 185)):
                                                                    
                                    #horizontal flip 
                                    image_flip = img[:, ::-1]
                                    img_flip = train_transformer(image_flip.copy())
                                    img_flip = img_flip.type(torch.FloatTensor)
                                    img_flip = img_flip.unsqueeze(0)
                                    output_image_flip = model(img_flip)

                                    added_images_flip.append(output_image_flip.detach().numpy())

                                    #add noise
                                    added_img_noise = sk.util.random_noise(image_flip, mode = 'poisson')
                                    added_img_noise = (255*added_img_noise).astype(np.uint8)

                                    added_image_noise = train_transformer(added_img_noise)
                                    added_image_noise = added_image_noise.type(torch.FloatTensor)
                                    added_image_noise = added_image_noise.unsqueeze(0)
                                    output_added_image_noise = model(added_image_noise)
                                    added_images_noise.append(output_added_image_noise.detach().numpy())
                                    
                                    #change contrast
                                    added_v_min, added_v_max = np.percentile(image_flip, (0.2, 99.8))
                                    added_image_contrast = exposure.rescale_intensity(image_flip, in_range=(added_v_min, added_v_max))

                                    added_image_contrast = train_transformer(added_image_contrast)
                                    added_image_contrast = added_image_contrast.type(torch.FloatTensor)
                                    added_image_contrast = added_image_contrast.unsqueeze(0)
                                    output_added_image_contrast = model(added_image_contrast)
                                    added_images_contrast.append(output_added_image_contrast.detach().numpy())
                                
                                    #change gamma
                                    added_image_gamma = exposure.adjust_gamma(image_flip, gamma=0.4, gain=0.9)
                                    added_image_gamma = train_transformer(added_image_gamma)
                                    added_image_gamma = added_image_gamma.type(torch.FloatTensor)
                                    added_image_gamma = added_image_gamma.unsqueeze(0)
                                    output_added_image_gamma = model(added_image_gamma)

                                    added_images_gamma.append(output_added_image_gamma.detach().numpy())        

                                            
                                
                    v_cap.release()
                    
                    if (file[6:8] == '01') or ((file[6:8] == '07') and (countSurVid < 185)) or ((file[6:8] == '08') and (countDisVid < 185)):
                        np.save(vid_path + file[0:-4] + "-added_frame_contrast", added_images_contrast)  
                        np.save(vid_path + file[0:-4] + "-added_frame_gamma", added_images_gamma)
                        np.save(vid_path + file[0:-4] + "-added_frame_flip", added_images_flip)
                        np.save(vid_path + file[0:-4] + "-added_frame_noise", added_images_noise)
                    
                    np.save(vid_path + file[0:-4], images)
                    np.save(vid_path + file[0:-4] + "-frame_noise", images_noise)
                    np.save(vid_path + file[0:-4] + "-frame_gamma", images_gamma)
                    np.save(vid_path + file[0:-4] + "-frame_contrast", images_contrast)
    
    video_path = 'features/temp_features/vid_features'

    video_file_emotion = []
    video_file_path = []
    for subdir, dirs, files in os.walk(video_path, topdown=True):
        for file in files:     
            if file[0:2] == '02': # use only video files without audio
                video_part = file.split('.')[0]
                video_part = video_part.split('-')
                video_file_emotion.append(int(video_part[2]))
                video_file_path.append( os.path.join(subdir,file))

    # dataframe for emotion of files
    video_emotion_df = pd.DataFrame(video_file_emotion, columns=['Emotions'])
    # dataframe for path of files.
    video_path_df = pd.DataFrame(video_file_path, columns=['Path'])
    video_Ravdess_df = pd.concat([video_emotion_df, video_path_df], axis=1)

    # changing integers to actual emotions.
    video_Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)
    video_Ravdess_df.head()

    plt.title('Count of Emotions', size=16)
    sns.countplot(video_Ravdess_df.Emotions)
    
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()    

def feature_extractor_audio(X, sample_rate):
        
        # Mel 
        log_mel=librosa.power_to_db(librosa.feature.melspectrogram(X, sr=sample_rate))
        mel =librosa.util.normalize(log_mel)

        '''
        #Uncomment this if you want to extract MFCCs features
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30)
        return mfccs
        '''
        return mel