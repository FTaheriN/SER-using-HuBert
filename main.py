import time
import numpy as np
import pandas as pd

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from dataloaders import get_dataloader
from deeplearning.train import *
from deeplearning.test import *
from model import HuBertWithNNClassifier_1, HuBertWithNNClassifier
from utils import *

############################## Reading Model Parameters ##############################
config = read_yaml_config()
file_path = config['file_path']

model = config['model']

model_pth = config['model_pth']

section = config['section']

####################################      Main     #################################### 

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_loader, valid_loader, test_loader = get_dataloader(file_path)

    if section == 1:
        hubert_model = HuBertWithNNClassifier()
    else: 
        hubert_model = HuBertWithNNClassifier_1()

    optimizer = torch.optim.Adam(hubert_model.parameters(), lr=0.00002)
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.7)
    criterion = nn.CrossEntropyLoss()

    train_loss_1, train_acc_1, valid_loss_1, valid_acc_1 = train(hubert_model, 
                                                         train_loader, valid_loader, 
                                                         criterion, optimizer, lr_scheduler,
                                                         model_pth, device)
    
    report_path = "/content/drive/MyDrive/MSC/DeepLearning/HW4/Q2/report/" + model_pth
    save_report(train_loss_1, valid_loss_1, train_acc_1, valid_acc_1, report_path)
    plot_loss_acc(report_path)
    
    model = torch.load("/content/drive/MyDrive/MSC/DeepLearning/HW4/Q2/checkpoints/" + model_pth)
    test_acc_, preds_, y_test = test(test_loader, model, device, criterion)
    test_report(y_test, preds_)


main()