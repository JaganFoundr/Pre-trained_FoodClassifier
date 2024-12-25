# 1. torch related libraries
import torch
import torch.nn as nn
from torchvision.datasets import Food101
#import torchvision.transforms as T

import torchvision.models as models

from torchvision.models import EfficientNet_B2_Weights

#Modular Pytorch
from Pytorch_Modules import custom_zipfile_download
from Pytorch_Modules import torch_prebuilt_data_folder_format
from Pytorch_Modules import plotting
from Pytorch_Modules import datasets
#from Pytorch_Modules import custom_model_builder
from Pytorch_Modules import metrics
from Pytorch_Modules import model_runtime

#os
import os

#model running time
from timeit import default_timer as timer

#current model info
from torchinfo import summary

# 2. Setting Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#6. Plotting non transformed(raw) random images from the whole dataset.
plotting.plot_raw_random("data2/pizza_steak_sushi")

#7. Building the Custom CNN model
#custom_model = custom_model_builder.CustomCNN(input_shape=3,hidden_units=32,output_shape=101)

#pre-trained model
EfficientNet=models.efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
EfficientNet.classifier = nn.Linear(1408, 3)  # Replace the final layer for 101 classes
EfficientNet = EfficientNet.to(device)

# Freeze all base layers by setting requires_grad attribute to False
for param in EfficientNet.parameters():
    param.requires_grad = False

#keeping the last linear layer
for param in EfficientNet.classifier.parameters():
    param.requires_grad=True

#8. Setup pretrained weights (plenty of these available in torchvision.models)
EfficientNet_weights = EfficientNet_B2_Weights.DEFAULT
# Get transforms from weights (these are the transforms that were used to obtain the weights)
EfficientNet_transforms = EfficientNet_B2_Weights.transforms()
print(f"Automatically created 18 transforms: {EfficientNet_transforms}")

batch_size=32
summary(model=EfficientNet, 
        input_size=(batch_size, 3, 288, 288),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# 10. Transforming and plotting the same raw images.
plotting.plot_transformed_random("data2/pizza_steak_sushi",transform=EfficientNet_transforms)
#plotting.plot_transformed_random("data2/pizza_steak_sushi",transform=efficientnet_transforms)

#11 .Now creating the format for training and testing dataset in order to upload to the dataloader.

efficientnet_train_data,efficientnet_test_data=datasets.create_dataset(train_folder="data2/pizza_steak_sushi/train",
                                             test_folder="data2/pizza_steak_sushi/test",

                                             train_transform=EfficientNet_transforms,
                                             test_transform=EfficientNet_transforms,
                                             
                                             target_train_transform=None,
                                             target_test_transform=None)


#12. Preparing Dataloader

efficientnet_train_loader,efficientnet_test_loader=datasets.Dataloader(train_dataset=efficientnet_train_data,
                                             test_dataset=efficientnet_test_data,

                                             batch_size=32,
                                             num_workers=os.cpu_count(),
                                             
                                             train_shuffle=True,
                                             test_shuffle=False,
                                             pin_memory=True)

#13. Untrained Prediction
torch.manual_seed(42)
for images, labels in efficientnet_train_loader:
  images, labels = images.to(device), labels.to(device)
  prediction = EfficientNet(images)
  break
prediction[0]

#14. Loss Function and Optimizer
loss_function = nn.CrossEntropyLoss()

#resnet_opt=torch.optim.SGD(resnet.parameters(),lr=0.01,momentum=0.9)
efficientnet_opt = torch.optim.Adam(EfficientNet.parameters(), betas=(0.9,0.999),lr=0.001,weight_decay=0.3)

#15. accuracy function
def accuracy(output, labels):
    '''# Accuracy Function'''
    _, pred = torch.max(output, dim=1)
    return torch.sum(pred == labels).item() / len(pred) * 100

#16. Training the Model
start_time = timer()

experiment_configs = [
    {
        'model': EfficientNet,  
        'optimizer': efficientnet_opt,
        'epochs': 10, 
        'name': 'efficientnet_b2_Food_Classifying_Exp'
    }
]

# Define the DataLoader for each model (train and test)
train_loaders = [efficientnet_train_loader]
test_loaders = [efficientnet_test_loader]

# Call the training function
metrics.train_plot_tensorboard_multiple_experiments(experiment_configs, train_loaders, test_loaders, loss_function)

end_time = timer()
model_runtime.run_time(start_time, end_time, device=device)

#17. confusion matrix for both train and test
metrics.conf_matrix_for_train(model=EfficientNet,
                              image_path="data2/pizza_steak_sushi/train",
                              train_loader=efficientnet_train_loader)

metrics.conf_matrix_for_test(model=EfficientNet,
                             image_path="data2/pizza_steak_sushi/test",
                             test_loader=efficientnet_test_loader)

#18. Train and Test images prediction
metrics.train_prediction(class_names_parent_path="data2/pizza_steak_sushi/train",model=EfficientNet,
                        image_path="data2/pizza_steak_sushi/train",
                        )

metrics.test_prediction(class_names_parent_path="data2/pizza_steak_sushi/test",model=EfficientNet,
                        image_path="data2/pizza_steak_sushi/test",
                        )

#19. Saving and Loading Model
torch.save(EfficientNet.state_dict(), "efficient_b2FE_10epoch_Food_Classifier.pth")
load_model = EfficientNet

load_model.load_state_dict(torch.load("efficientnet_b2FE_10epoch_Food_Classifier.pth"))

#20. Testing the custom image
metrics.custom_image_plot(class_names_parent_path="data/pizza_steak_sushi/test",
                          image_path="data/pizza_steak_sushi/pizza.jpg",
                          device=device,
                          model=EfficientNet)
