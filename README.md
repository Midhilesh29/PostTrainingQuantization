# post-training-quantization
compares different pretrained object classification with per-layer and per-channel quantization using pytorch

```
Steps to work with google colab
1. Clone the repository
2. Add the path to sys.path so that libraries can be traversed correctly
3. create folder in google colab /content/data/
4. import the compare_models.py like from compare_models import compare_model
5. This package supports AlexNet,MobileNetv2, resnet50, resnet18 models only
6. Run the following commands
6. list = ["AlexNet","MobileNetv2", "resnet50", "resnet18"] //creating a list
7. statistics = compare_model(list) //passing the list as argument such that these models will be compared
```
