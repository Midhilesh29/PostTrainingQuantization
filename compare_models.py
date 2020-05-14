import torch.nn as nn
from QAlexnet import QAlexnet
from QMobilenetv2 import QMobilenetv2
from per_channel import per_channel_quantization
from per_tensor import per_tensor_quantization
from utils import evaluate,print_size_of_model
from imagenet_data import createDataLoaders,downloadData

downloadData()
print("Imagenet data is downloaded")
data_loader, data_loader_test = createDataLoaders()
print("Data loaders are created")

def compare_model(model_name,num_eval_batches=10,criterion = nn.CrossEntropyLoss()):

    model=None
    per_tensor_model=None
    per_channel_model=None
    if(model_name=="AlexNet"):
        model=QAlexnet()
    elif(model_name=="MobileNetv2"):
        model=QMobilenetv2()
    print("Checking size of "+f"{model_name}")
    print("Size (MB) of "+f"{model_name} is:",print_size_of_model(model))
    print("Evaluvating "+ f"{model_name} on the imagenet data\n")
    _,model_top5 = evaluate(model, criterion, data_loader, neval_batches=num_eval_batches)

    per_tensor_model = per_tensor_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)
    print("Checking model size after per_tensor quantization")
    print("Print per-tensor quantized model size (MB):\n",print_size_of_model(per_tensor_model))
    print("Evaluvating the per-tensor model on the imagenet data\n")
    _,tensor_top5 = evaluate(per_tensor_model, criterion, data_loader, neval_batches=num_eval_batches)

    per_channel_model = per_channel_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)
    print("Checking model size after per_tensor quantization")
    print("Print per-channel quantized model size (MB):\n",print_size_of_model(per_tensor_model))
    print("Evaluvating the per-tensor model on the imagenet data\n")
    _,channel_top5 = evaluate(per_channel_model, criterion, data_loader_test, neval_batches=num_eval_batches)

    print("\n")
    print(f"{model_name} per-tensor quantized top5 accuracy on imagenet data with {str(num_eval_batches)} batches is {str(tensor_top5.avg)}")
    print("\n")
    print(f"{model_name} per-chennal quantized top5 accuracy on imagenet data with {str(num_eval_batches)} batches is {str(channel_top5.avg)}")
    print("\n")
    print(f"{model_name} top5 accuracy on imagenet data with {str(num_eval_batches)} batches {str(model_top5.avg)}")