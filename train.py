import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np

parser = argparse.ArgumentParser()


parser.add_argument('--data_dir', help = 'add images for  classifier', type= str)
parser.add_argument('--save_dir', default = 'checkpoint.pth', help ='Set directory to save checkpoints', type= str, action='store')
parser.add_argument('--arch', default = 'vgg16',  help ='Choose architecture', type=str, action='store')
parser.add_argument('--learning_rate', default = 0.0003, help ='Choose learning rate', type= int, action='store')
parser.add_argument('--Model_inputs', default = 25088, help ='Choose hidden units',type= int, action='store')
parser.add_argument('--hidden_units_1', default = 1080, help ='Choose hidden units',type= int, action='store')
parser.add_argument('--hidden_units_2', default = 720, help ='Choose hidden units',type= int, action='store')
parser.add_argument('--hidden_units_3', default = 360, help ='Choose hidden units',type= int, action='store')
parser.add_argument('--datasets_number', default = 102, help ='Choose datasets_number',type= int, action='store')
parser.add_argument('--dropout', default = 0.6, help ='Choose dropout',type= int, action='store')
parser.add_argument('--epochs', default = 5, help = 'choose epochs', type= int, action='store')
parser.add_argument('--device', default = "cuda" , help='Use cuda or cpu for training', type= str, action='store')

args = parser.parse_args()

print(args)


data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

device = torch.device(args.device)

def data_loader(data_dir):
    
    # transform data
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # Load the datasets with ImageFolder
    
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    
    # Using the image datasets and the trainforms, define the dataloaders
    # python train.py
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    
    return trainloader, testloader, validloader, train_data, test_data, valid_data


trainloader, testloader, validloader, train_data, test_data, valid_data = data_loader(data_dir)



def classifier(inputs = args.Model_inputs, hl1=args.hidden_units_1, hl2= args.hidden_units_2, hl3= args.hidden_units_3, dropout=   args.dropout , datasetNum= args.datasets_number , arch = args.arch):
    
    
    model =  getattr(models,args.arch)(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                ('inputs', nn.Linear(inputs, hl1)),
                ('relu', nn.ReLU()),
                ('dropout',nn.Dropout(dropout)),
                ('hiddenLayer1', nn.Linear(hl1, hl2)),
                ('relu',nn.ReLU()),
                ('dropout',nn.Dropout(dropout)),
                ('hiddenLayer2',nn.Linear(hl2, hl3)),
                ('relu',nn.ReLU()),
                ('dropout',nn.Dropout(dropout)),
                ('hiddenLayer3',nn.Linear(hl3, datasetNum)),
                ('output', nn.LogSoftmax(dim=1))
        ]))
        


    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = args.learning_rate)
    model.to(device)
    
    return  model, criterion, optimizer

model, criterion, optimizer = classifier(args.Model_inputs , args.hidden_units_1, args.hidden_units_2, args.hidden_units_3, args.dropout, args.datasets_number ,args.arch)

def train():
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                model.train()
                
    return model

The_model = train()

def model_save(model):
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  'class_to_idx':model.class_to_idx,
                  'optimizer_dict':optimizer.state_dict()
                  }

    torch.save(checkpoint, args.save_dir)
    
model_save(The_model)
    



            
        
    
