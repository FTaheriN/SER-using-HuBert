from tqdm import tqdm
import torch
from utils import calc_acc




def train(model, train_loader, valid_loader, loss_fn, optimizer, lr_scheduler, model_name, device):

    model = model.to(device)
    best_acc=0.0
    total_step = len(train_loader)
    train_loss_ = []
    train_acc_ = []
    valid_loss_ = []
    valid_acc_ = []

    for epoch in range(14):

        # Train Phase:

        model.train()
        total_train_loss = 0
        total_train_acc  = 0
        loop=tqdm(enumerate(train_loader),leave=False,total=len(train_loader))

        for batch_idx, (audio_features, attention_mask, y) in loop: #

            optimizer.zero_grad()

            audio_features = audio_features.to(device)
            attention_mask = attention_mask.to(device)
            labels = y.to(device)

            
            prediction = model(audio_features, attention_mask)

            loss = loss_fn(prediction, labels)
            acc = calc_acc(prediction, labels)

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_acc  += acc.item()

        train_acc  = 100*total_train_acc/len(train_loader)
        train_loss = total_train_loss/len(train_loader)

        train_loss_.append(train_loss)
        train_acc_.append(train_acc)
        
        # Eval Phase

        model.eval()
        total_val_acc  = 0
        total_val_loss = 0

        with torch.no_grad():
           for batch_idx, (audio_features,attention_mask, y) in enumerate(valid_loader): # attention_mask, 

                optimizer.zero_grad()

                audio_features = audio_features.to(device)
                attention_mask = attention_mask.to(device)
                labels = y.to(device)

                # print(labels)

                prediction = model(audio_features, attention_mask)
                
                loss = loss_fn(prediction, labels)
                acc = calc_acc(prediction, labels)

                total_val_loss += loss.item()
                total_val_acc  += acc.item()

        val_acc  = 100* total_val_acc/len(valid_loader)
        val_loss = total_val_loss/len(valid_loader)

        valid_loss_.append(val_loss)
        valid_acc_.append(val_acc)

        if val_acc > best_acc: 
            torch.save(model, '/content/drive/MyDrive/MSC/DeepLearning/HW4/Q2/checkpoints/' + model_name)
            best_acc = val_acc 

        print(f'Epoch {epoch+1}:\n train_loss: {train_loss:.2f}  val_loss: {val_loss:.2f} \n train_acc : {train_acc:.1f}  val_acc : {val_acc:.1f} \n')

        lr_scheduler.step()

    return train_loss_, train_acc_, valid_loss_, valid_acc_