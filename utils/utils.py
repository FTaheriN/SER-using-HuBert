import torch 
import csv
import pandas as pd
import matplotlib.pyplot as plt



def read_file(file_path):
    return pd.read_csv(file_path, sep='\t', on_bad_lines='skip')


def calc_acc(y_pred, y_test):
    acc = (torch.log_softmax(y_pred, dim=1).argmax(dim=1) == y_test).sum().float() / float(y_test.size(0))
    return acc


def save_report(train_loss, valid_loss, train_acc, valid_acc, file_path):
    
    rows = zip(train_loss, valid_loss, train_acc, valid_acc)
    headers = ['train_loss', 'valid_loss', 'train_acc', 'valid_acc']

    with open(file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow([g for g in headers])
        for row in rows:
            writer.writerow(row)
    return 


def plot_loss_acc(df_path):

    df = pd.read_csv(df_path)

    fig, ax = plt.subplots(2,1,figsize=(6,8));

    ax[0].plot(df['train_loss'])
    ax[0].plot(df['valid_loss'])
    ax[0].set_title('Model Loss')
    ax[0].legend(['train loss', 'valid loss']);

    ax[1].plot(df['train_acc'])
    ax[1].plot(df['valid_acc'])
    ax[1].set_title('Model Accuracy');
    ax[1].legend(['train accuracy', 'valid accuracy'])