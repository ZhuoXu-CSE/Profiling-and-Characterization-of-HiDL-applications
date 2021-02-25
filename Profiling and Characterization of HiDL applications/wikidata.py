import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
train_transforms = transforms.Compose([transforms.Resize(255),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
test_transforms = transforms.Compose([transforms.Resize(255),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))])
train_data = datasets.ImageFolder('train/',  
                                    transform=train_transforms)                                       
test_data = datasets.ImageFolder('test/', 
                                    transform=test_transforms)
#Data Loading
trainloader = torch.utils.data.DataLoader(train_data,batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
def age(tag):
    id,dob,cur = tag.split("_")
    year,_,_ = dob.split('-')
    return int(cur)-int(year)
class WikiDataLoader(DataLoader):
    def __getitem__(self,idx):
        img = images[idx]
        label = labels[idx]
        return img, age(label)
