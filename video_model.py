from __future__ import print_function, division
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
from torchvision import transforms
import torch.optim as optim
import numpy as np
from pytorch_lightning import Trainer
import joblib
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from early_stopping import EarlyStopping
import cv2
from torch.nn import Linear, ReLU, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout

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

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            Dropout(0.5)
        )

        self.linear_layers = Sequential(
            Linear(16 * 16 * 256, CLASSES)
        )  

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train(vid_data_loader, model, optimizer):

    train_losses = []
    train_acc = []
    valid_losses = []
    valid_acc = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=10, verbose=True, path='video_checkpoint.pt')

    #Training of the model
    model.train()
    print(model)
    for epoch in range(EPOCHS):
        total = 0
        correct = 0
        for i, data in enumerate(vid_data_loader['vid_train'], 0):
            # get the inputs
            vid_inputs, vid_labels = data
      
            vid_inputs = vid_inputs.permute(0, 2, 1, 3)

            vid_inputs = vid_inputs.type(torch.FloatTensor)
            vid_inputs, vid_labels = vid_inputs.to(DEVICE), vid_labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()
            output = model(vid_inputs)
            loss = nn.CrossEntropyLoss()(output, vid_labels)
            _, predicted = torch.max(output.data, 1)
            total += vid_labels.size(0)
            correct += (predicted == vid_labels).sum().item()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_acc.append(correct/total)
        train_accuracy = correct/total

        #validation of the model
        model.eval()
        total = 0
        correct = 0
    
        with torch.no_grad():
            for i, data in enumerate(vid_data_loader['vid_val'], 0):
                vid_inputs, vid_labels = data
                vid_inputs = vid_inputs.permute(0, 2, 1, 3)

                vid_inputs = vid_inputs.type(torch.FloatTensor)
                vid_inputs, vid_labels = vid_inputs.to(DEVICE), vid_labels.to(DEVICE)
            
                output = model(vid_inputs)
                loss = nn.CrossEntropyLoss()(output, vid_labels)
                _, predicted = torch.max(output.data, 1)
                total += vid_labels.size(0)
                correct += (predicted == vid_labels).sum().item()

                valid_losses.append(loss.item())

        valid_acc.append(correct/total)
        accuracy = correct/total

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
    #plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.ylim(0, 1) # consistent scale
    plt.xlim(0, len(train_acc)+1) # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def test(vid_data_loader, model):

    total_labels = []
    total_inputs = []
    total_predicted = []
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(vid_data_loader['vid_test'], 0):
                vid_inputs, vid_labels = data

                vid_inputs = vid_inputs.permute(0, 2, 1, 3)
                vid_inputs = vid_inputs.type(torch.FloatTensor)
                vid_inputs, vid_labels = vid_inputs.to(DEVICE), vid_labels.to(DEVICE)
                outputs = model(vid_inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += vid_labels.size(0)
                correct += (predicted == vid_labels).sum().item()

                for single_vid_label in vid_labels: 
                    total_labels.append(single_vid_label.cpu())
                for single_predicted in predicted:
                    total_predicted.append(single_predicted.cpu())

    print('Accuractey of the network on the test images: %d %%' % (
            100 * correct / total))

    confusion_matrix = plot_confusion_matrix(total_labels, total_predicted, title='Confusion matrix normalized')
    plt.show()

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
            ax.set(xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=classes, yticklabels=classes,
            title=title,
            ylabel='True labels',
            xlabel='Predicted labels')

            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
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

def video_model(BATCHSIZE):

    vid_dataset = ['vid_train', 'vid_test', 'vid_val']
    video_data = {}

    for x in vid_dataset:
        video_data[x] = torchvision.datasets.DatasetFolder(root=DIR + '/' + x, loader=npy_loader, extensions = ('.npy'))
      
    vid_data_loader = {}
    for x in vid_dataset:
        vid_data_loader[x] = torch.utils.data.DataLoader(video_data[x],
                 batch_size=BATCHSIZE, shuffle=True, num_workers=0)

    return vid_data_loader
    

def select_train_or_test_video():

    BATCHSIZE = 32
    vid_data_loader = video_model(BATCHSIZE)

    print("Do you want to train or test? ")

    model = Net().to(DEVICE)
    user = input() 
    
    if user.lower() == 'train':
        optimizer = getattr(optim,'Adam')(model.parameters(), lr=1e-5)
        train(vid_data_loader, model, optimizer)
    elif user.lower() == 'test':
        model.load_state_dict(torch.load('video_checkpoint.pt'))
        test(vid_data_loader, model)
