from CustomDataset import CustomDataset
from torchvision import transforms


def test_dataset_class(dataset):
    # the part for training
    test_data = dataset[8000:]
    training = dataset[:8000]
    valid_data = training[6000:]  # 2000 for validation
    train_data = training[:6000]  # 6000 for train
    train_set = CustomDataset(train_data, transform=transforms.ToTensor)
    valid_set = CustomDataset(valid_data, transform=transforms.ToTensor)
    test_set = CustomDataset(test_data, transform=transforms.ToTensor)