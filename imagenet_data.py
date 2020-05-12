import requests
import zipfile
import torchvision
import torchvision.transforms as transforms

def downloadData(download_location='/content/',extract_location='/content/'):
    url = 'https://s3.amazonaws.com/pytorch-tutorial-assets/imagenet_1k.zip'
    filename = download_location+'imagenet_1k_data.zip'

    r = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(r.content)

    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall(extract_location)
    zip_ref.close()

def createDataLoaders(data_path='/content/imagenet_1k',train_batch=30,eval_batch=30):
    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

    dataset_train = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    
    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_batch,
        sampler=train_sampler)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=eval_batch,
        sampler=test_sampler)
    
    return data_loader,data_loader_test

    
