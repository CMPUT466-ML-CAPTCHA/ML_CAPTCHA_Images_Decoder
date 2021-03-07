from torch.optim import optimizer
from CustomDataset import CustomDataset
from pathlib import Path
import random
from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from RNN import RNN   


# Load the data from the Google Drive
# data_dir = Path("/content/drive/MyDrive/Data")

# path of data set for local
dataDir = Path("./archive")

# images: the list contain the path of each images
images = list(dataDir.glob("*.jpg"))
random.shuffle(images)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))

# single validation
testData = images[8000:]  # 2000 for test

# the part for training
training = images[:8000]
validData = training[6000:]  # 2000 for validation
trainData = training[:6000]  # 6000 for train
BATCH_SIZE = 100

# init database
testSet = CustomDataset(testData, transform=transforms.ToTensor())
trainSet = CustomDataset(trainData, transform=transforms.ToTensor())
validSet = CustomDataset(validData, transform=transforms.ToTensor())

trainDataloader = DataLoader(dataset=trainSet, batch_size=BATCH_SIZE, shuffle=True)
testDataloader = DataLoader(dataset=testSet, batch_size=BATCH_SIZE, shuffle=True)
validDataloader = DataLoader(dataset=validSet, batch_size=BATCH_SIZE, shuffle=True)

IMAGE_SIZE = trainSet.height * trainSet.width
model = RNN(IMAGE_SIZE, BATCH_SIZE, 36*6).to(device)
criterion = torch.nn.MultiLabelSoftMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train(model, trainLoader, validLoader, device):  
    model.train() 
    for epoch in range(20):
        model.train()
        for _, (images, labels) in enumerate(trainLoader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            
            predict = model(images, None)[0]
            optimizer.zero_grad()
            
            loss = criterion(predict, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 5 == 0:
            accuracy = valid(model, validLoader, device)
            print("epoch: {} loss: {:.10f} accuracy: {:.4f}".format(
                (epoch+1), loss.item(), accuracy))

def valid(model, validLoader, device):
    num_correct = 0  # the counter for the correct items
    num_total = len(validLoader)*100  # the counter for the total items
    mean_acc = 0  # the accuracy of the validation
    model.eval()  # set the evaluation state of the model
    with torch.no_grad():
        for _, (images, labels) in enumerate(validLoader):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images, None)
            labels = labels.reshape((100, 6, 36))
            output = output.reshape((100, 6, 36))
            labels = torch.argmax(labels, dim=2)
            output = torch.argmax(output, dim=2)
            num_correct += ((output == labels).sum(dim=1) == 6).sum().item()
        mean_acc = num_correct / num_total * 100
        return mean_acc
    
train(model, trainDataloader, validDataloader, device)