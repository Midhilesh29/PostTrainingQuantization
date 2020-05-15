import torch.nn as nn
from QAlexnet import QAlexnet
from QMobilenetv2 import QMobilenetv2
from QResnet import resnet18,resnet50
from per_channel import per_channel_quantization
from per_tensor import per_tensor_quantization
from utils import evaluate,print_size_of_model
from imagenet_data import createDataLoaders,downloadData

downloadData()
print("Imagenet data is downloaded")
data_loader, data_loader_test = createDataLoaders()
print("Data loaders are created")

def compare_model(deep_learning_models,num_eval_batches=10,criterion = nn.CrossEntropyLoss()):

    statistics={}
    for model_name in deep_learning_models:
        tensor_size=None
        tensor_accuracy=None
        channel_size=None
        channel_accuracy=None
        size_of_model=None
        accuracy_of_model=None

        model=None
        per_tensor_model=None
        per_channel_model=None
        if(model_name=="AlexNet"):
            model=QAlexnet()
            model.eval()
        elif(model_name=="MobileNetv2"):
            model=QMobilenetv2()
            model.eval()
            model.fuse_model()
        elif(model_name=="resnet18"):
            model = resnet18(pretrained=True)
            model.eval()
            model.fuse_model()
        elif(model_name=="resnet50"):
            model = resnet50(pretrained=True)
            model.eval()
            model.fuse_model()
        else:
            print("Provide model names among AlexNet,MobineNetv2,resnet18,resnet50")
            break

        print("Checking size of "+f"{model_name}")
        size_of_model = print_size_of_model(model)
        print("Size (MB) of "+f"{model_name} is:",size_of_model)
        print("Evaluvating "+ f"{model_name} on the imagenet data\n")
        _,model_top5 = evaluate(model, criterion, data_loader, neval_batches=num_eval_batches)

        per_tensor_model = per_tensor_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)
        print("Checking model size after per_tensor quantization")
        tensor_size = print_size_of_model(per_tensor_model)
        print("Print per-tensor quantized model size (MB):\n",tensor_size)
        print("Evaluvating the per-tensor model on the imagenet data\n")
        _,tensor_top5 = evaluate(per_tensor_model, criterion, data_loader, neval_batches=num_eval_batches)

        per_channel_model = per_channel_quantization(model=model,criterion = nn.CrossEntropyLoss(),data_loader=data_loader)
        print("Checking model size after per_tensor quantization")
        channel_size = print_size_of_model(per_channel_model)
        print("Print per-channel quantized model size (MB):\n",channel_size)
        print("Evaluvating the per-tensor model on the imagenet data\n")
        _,channel_top5 = evaluate(per_channel_model, criterion, data_loader_test, neval_batches=num_eval_batches)

        print("\n")
        print(f"{model_name} per-tensor quantized top5 accuracy on imagenet data with {str(num_eval_batches)} batches is {str(tensor_top5.avg)}")
        print("\n")
        print(f"{model_name} per-channel quantized top5 accuracy on imagenet data with {str(num_eval_batches)} batches is {str(channel_top5.avg)}")
        print("\n")
        print(f"{model_name} top5 accuracy on imagenet data with {str(num_eval_batches)} batches {str(model_top5.avg)}")
        print("\n")

        tensor_accuracy=tensor_top5.avg
        channel_accuracy=channel_top5.avg
        accuracy_of_model = model_top5.avg
        statistics[model_name]={"parent_model_size":size_of_model,
                                "parent_top5_accuracy":accuracy_of_model,
                                "tensor_quantized_size":tensor_size,
                                "tensor_quantized_top5_accuracy":tensor_accuracy,
                                "channel_quantized_size":channel_size,
                                "channel_quantized_top5_accuracy":channel_accuracy}
    return statistics