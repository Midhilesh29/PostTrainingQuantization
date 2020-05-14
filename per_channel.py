import torch.quantization
import torch.nn as nn
from utils import evaluate,print_size_of_model

def per_channel_quantization(model,data_loader,num_calibration_batches = 10,criterion = nn.CrossEntropyLoss()):
    
    model.eval()

    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    print("Coniguration of quantization:\n", model.qconfig)
    
    torch.quantization.prepare(model, inplace=True)

    evaluate(model, criterion, data_loader, neval_batches=num_calibration_batches)
    torch.quantization.convert(model, inplace=True)
    print('Post Training Quantization: Convert done\n')


    return model