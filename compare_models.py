import torch.nn as nn
from QAlexnet import QAlexnet
from per_channel import per_channel_quantization
from per_tensor import per_tensor_quantization
from utils import evaluate
from imagenet_data import createDataLoaders,downloadData

downloadData()
print("Imagenet data is downloaded")
data_loader, data_loader_test = createDataLoaders()
print("Data loaders are created")

def compare_model(model_name,num_eval_batches=30,criterion = nn.CrossEntropyLoss()):
    
    if(model_name=="AlexNet"):
        model=QAlexnet()
        per_tensor_model = per_tensor_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)
        per_channel_model = per_channel_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)


    tensor_top1, tensor_top5 = evaluate(per_tensor_model, criterion, data_loader_test, neval_batches=num_eval_batches)

    channel_top1, channel_top5 = evaluate(per_channel_model, criterion, data_loader_test, neval_batches=num_eval_batches)

    print(f"{model_name} accuracy on imagenet data with {str(num_eval_batches)} batches and per_tensor_quantization is {str(tensor_top1.avg)}")
    print(f"{model_name} accuracy on imagenet data with {str(num_eval_batches)} batches and per_channel_quantization is {str(channel_top1.avg)}")