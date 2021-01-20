from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
import torch.optim as optim
import numpy as np
import joblib
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from early_stopping import EarlyStopping
import cv2
from torch.utils.data import Dataset
from early_stopping import EarlyStopping

DEVICE = torch.device("cuda")
CLASSES = 8
DIR = 'features/temp_features'
EPOCHS = 50


emotions={
  0:'neutral',
  1:'calm',
  2:'happy',
  3:'sad',
  4:'angry',
  5:'fearful',
  6:'disgust',
  7:'surprised'
}

class RAVDESS(Dataset):
    def __init__(self, dataset_video, dataset_audio):
        self.dataset_video = dataset_video
        self.dataset_audio = dataset_audio

    def __getitem__(self, index):
        x2 = self.dataset_audio[index]
        x1 = self.dataset_video[index]
        return x1, x2

    def __len__(self):
        return min(len(self.dataset_video), len(self.dataset_audio)) 


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.img_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.aud_conv1 = nn.Conv2d(1, 16, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.img_pool = nn.MaxPool2d(2, 2)
        self.aud_pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear((16*64*108)+(16*256*16),  8)
        
    def forward(self, x_img, x_aud):
        x_img = self.drop(self.img_pool(self.relu(self.img_conv1(x_img))))
        x_aud = self.drop(self.aud_pool(self.relu(self.aud_conv1(x_aud))))

        x_img = x_img.view(x_img.size(0), -1)
        x_aud = x_aud.view(x_aud.size(0), -1)

        x = torch.cat([x_img, x_aud], dim=1)

        x = self.fc1(x)
        return x


def train(loader, model, optimizer):
    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True, path='audio_video_checkpoint.pt')

    model.train()
    for epoch in range (EPOCHS):
        total = 0
        correct = 0 
        for i, data in enumerate(loader['train']):
            # get the inputs
            img_data, aud_data = data
            img_inputs, img_labels = img_data
            img_inputs = img_inputs.type(torch.FloatTensor)
            img_inputs, img_labels = img_inputs.to(DEVICE), img_labels.to(DEVICE)
            
            aud_inputs, aud_labels = aud_data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(DEVICE), aud_labels.to(DEVICE)
            aud_inputs = aud_inputs.unsqueeze(1)
            img_inputs = img_inputs.permute(0, 2, 1, 3)

            optimizer.zero_grad()
            outputs = model(img_inputs, aud_inputs)
            loss = nn.CrossEntropyLoss()(outputs, img_labels)
            _, predicted = torch.max(outputs.data, 1)
            total += img_labels.size(0)
            correct += (predicted == img_labels).sum().item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        train_acc.append(correct/total)
        train_accuracy = correct/total

        model.eval()        
        correct = 0
        total = 0
    
        with torch.no_grad():
            for i, data in enumerate(loader['val'], 0):
                img_data, aud_data = data
                img_inputs, img_labels = img_data
                img_inputs = img_inputs.type(torch.FloatTensor)
                img_inputs, img_labels = img_inputs.to(DEVICE), img_labels.to(DEVICE)

                aud_inputs, aud_labels = aud_data
                aud_inputs = aud_inputs.type(torch.FloatTensor)
                aud_inputs, aud_labels = aud_inputs.to(DEVICE), aud_labels.to(DEVICE)
                
                img_inputs = img_inputs.permute(0, 2, 1, 3)
                aud_inputs = aud_inputs.unsqueeze(1)
                        
                outputs = model(img_inputs, aud_inputs)
                _, predicted = torch.max(outputs.data, 1)
                loss = nn.CrossEntropyLoss()(outputs, img_labels)
                total += img_labels.size(0)
                correct += (predicted == img_labels).sum().item()
                valid_losses.append(loss.item())

        valid_acc.append(correct/total)
        accuracy = correct/total
        
        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
            
        epoch_len = len(str(EPOCHS))
            
        print_msg = (f'[{epoch+1:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                    f'train_loss: {train_loss:.5f} ' +
                    f'valid_loss: {valid_loss:.5f}')
            
        print(print_msg)
            
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Accuracy of the network on the train images: %d %%' % (100 * train_accuracy))
        print('Accuracy of the network on the validation images: %d %%' % (100 * accuracy))

    print('Finished Training')

    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(1,len(avg_train_losses)+1),avg_train_losses,label='Training Loss')
    plt.plot(range(1,len(avg_valid_losses)+1),avg_valid_losses,label='Validation Loss')
    # find position of lowest validation loss
    minposs = avg_valid_losses.index(min(avg_valid_losses))+1 
    plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 2) # consistent scale
    plt.xlim(0, len(avg_train_losses)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
 
    fig = plt.figure(figsize=(10,10))
    plt.plot(range(1,len(train_acc)+1),train_acc,label='Training Accuracy')
    plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Accuracy')
    minposs = valid_acc.index(min(valid_acc))+1 

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_acc)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test(loader, model):
    total_labels = []
    total_predicted = []
    correct = 0 
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for i, data in enumerate(loader['test'], 0):
            img_data, aud_data = data
            img_inputs, img_labels = img_data
            img_inputs = img_inputs.type(torch.FloatTensor)
            img_inputs, img_labels = img_inputs.to(DEVICE), img_labels.to(DEVICE)

            aud_inputs, aud_labels = aud_data
            aud_inputs = aud_inputs.type(torch.FloatTensor)
            aud_inputs, aud_labels = aud_inputs.to(DEVICE), aud_labels.to(DEVICE)
                
            aud_inputs = aud_inputs.unsqueeze(1)
            img_inputs = img_inputs.permute(0, 2, 1, 3)
                
            outputs = model(img_inputs, aud_inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += img_labels.size(0)
            correct += (predicted == img_labels).sum().item()

            for single_aud_label in aud_labels: 
                total_labels.append(single_aud_label.cpu())
            for single_predicted in predicted:
                total_predicted.append(single_predicted.cpu())            

    confusion_matrix = plot_confusion_matrix(total_labels, total_predicted, title='Confusion matrix normalized')
    plt.show()
    
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    
def plot_confusion_matrix(labels, test_labels,
                          title='CONFUSION MATRIX',
                          cmap=plt.cm.Blues):
            emotions_true = []
            emotions_pred = []

            labels = np.array(labels)
            test_labels = np.array(test_labels)
            for label in labels:
                emotions_true.append(emotions[label])
            for test_label in test_labels:
                emotions_pred.append(emotions[test_label])
            # Compute confusion matrix
            cm = confusion_matrix(emotions_true, emotions_pred)
            # Only use the labels that appear in the data
            classes = unique_labels(emotions_true, emotions_pred)
    
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
            print(cm)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            # ... and labels them with the respective list entries
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True labels',
            xlabel='Predicted labels')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

            # Loop over data dimensions and create text annotations.
            fmt = '.2f' 
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
             for j in range(cm.shape[1]):
              ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            return ax

np.set_printoptions(precision=2)

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

def audio_and_video_model(BATCHSIZE):
    vid_dataset = ['vid_train', 'vid_test', 'vid_val']
    aud_dataset = ['aud_train', 'aud_test', 'aud_val']

    video_data = {}
    audio_data = {}

    for x in aud_dataset:
        audio_data[x] = torchvision.datasets.DatasetFolder(root = DIR + '/' + x, loader=npy_loader, extensions=('.npy'))
    for x in vid_dataset:
        video_data[x] = torchvision.datasets.DatasetFolder(root=DIR + '/' + x, loader=npy_loader, extensions=('.npy'))

    datasets = ['train', 'test', 'val']
    loader = {}
    for x, y, z in zip(vid_dataset, aud_dataset, datasets):
        ds = RAVDESS(video_data[x], audio_data[y])
        loader[z] = torch.utils.data.DataLoader(ds, batch_size=BATCHSIZE, shuffle=True, num_workers=0)
    
    return loader

def select_train_or_test_audio_and_video():
    BATCHSIZE = 32
    lr = 1e-5

    aud_and_vid_data_loader = audio_and_video_model(BATCHSIZE)

    print("Do you want to train or test? ")
  
    model = Net().to(DEVICE)
    user = input() 

    if user.lower() == 'train':
        optimizer = getattr(optim, 'Adam')(model.parameters(), lr=lr)
        train(aud_and_vid_data_loader, model, optimizer)
    elif user.lower() == 'test':
        model.load_state_dict(torch.load('audio_video_checkpoint.pt'))
        test(aud_and_vid_data_loader, model)