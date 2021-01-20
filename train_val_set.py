import os
import numpy as np
import cv2
import librosa


folders = ['0', '1', '2', '3', '4', '5', '6', '7']
def train_test_val_split(tmpdir):
    aud_file_path = tmpdir + '/' + 'aud_features/'
    aud_train_set = tmpdir + '/' + 'aud_train_tmp/'
    aud_val_set = tmpdir + '/' + 'aud_val_tmp/'
    aud_test_set = tmpdir + '/' + 'aud_test_tmp/'

    vid_file_path = tmpdir + '/' + 'vid_features/' 
    vid_train_set = tmpdir + '/' + 'vid_train/'
    vid_val_set = tmpdir + '/' + 'vid_val/'
    vid_test_set = tmpdir + '/' + 'vid_test/'

    created = False
    paths = [vid_train_set, vid_val_set, vid_test_set, aud_train_set, aud_val_set, aud_test_set]

    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            created = True
            for i in folders:
                subfolder = path + i
                os.mkdir(subfolder)

    if not created: 
        return
    

    #Write audio train val test set

    aud_data = []
    aud_files = []

    for file in os.listdir(aud_file_path):
        aud = np.load(os.path.join(aud_file_path, file))
        aud_data.append(aud)
        aud_files.append(file)
    
    aud_train = []
    aud_test = []
    aud_val = []
    aud_train_files = []
    aud_test_files = []
    aud_val_files = []

    for f in range(len(aud_files)):     
        if  (aud_files[f][18:20] == '23') or (aud_files[f][18:20] == '24'):
            aud_test_files.append(aud_files[f])
            aud_test.append(aud_data[f])
        elif (aud_files[f][18:20] == '22') or (aud_files[f][18:20] == '21'):
            aud_val_files.append(aud_files[f])
            aud_val.append(aud_data[f])
        else: 
            aud_train_files.append(aud_files[f])
            aud_train.append(aud_data[f])

    #normalize
    mean = np.mean(aud_train, axis=0)
    std = np.std(aud_train, axis=0)

    aud_train = (aud_train - mean)/std
    aud_test = (aud_test - mean)/std
    aud_val = (aud_val - mean)/std

    print("writing audio data")

    for i in range(len(aud_train)):
        aud = aud_train[i]
        fp = aud_train_set + str(int(aud_train_files[i][7:8]) - 1) + '/'
        np.save(fp + "aud%d" %i, aud)

    for i in range(len(aud_test)):
        aud = aud_test[i]
        fp = aud_test_set + str(int(aud_test_files[i][7:8]) - 1) + '/'
        np.save(fp + "aud%d" %i, aud)

    for i in range(len(aud_val)):
        aud = aud_val[i]
        fp = aud_val_set + str(int(aud_val_files[i][7:8]) - 1) + '/'
        np.save(fp + "aud%d" %i, aud)


    #write video train val test set

    frame_data = []
    frame_name = []

    img_train = []
    img_test = []
    img_val = []
    img_train_files = []
    img_test_files = []
    img_val_files = []

    for file in os.listdir(vid_file_path):
        image = np.load(os.path.join(vid_file_path, file))
        frame_data.append(image)
        frame_name.append(file)
    
    for f in range(len(frame_name)):     
        if  (frame_name[f][18:20] == '23') or (frame_name[f][18:20] == '24'):
            img_test_files.append(frame_name[f])
            img_test.append(frame_data[f])
        elif (frame_name[f][18:20] == '22') or (frame_name[f][18:20] == '21'):
            img_val_files.append(frame_name[f])
            img_val.append(frame_data[f])
        else: 
            img_train_files.append(frame_name[f])
            img_train.append(frame_data[f])
        
    print("writing video data")
    
    for i in range(len(img_train)):
        img = img_train[i]
        fp = vid_train_set + str(int(img_train_files[i][7:8]) - 1) + '/'
        np.save(fp + "vid%d" %i, img)
    
    for i in range(len(img_test)):
        img = img_test[i]
        fp = vid_test_set + str(int(img_test_files[i][7:8]) - 1) + '/'
        np.save(fp + "vid%d" %i, img)

    for i in range(len(img_val)):
        img = img_val[i]
        fp = vid_val_set + str(int(img_val_files[i][7:8]) - 1) + '/'
        np.save(fp + "vid%d" %i, img)    
    
