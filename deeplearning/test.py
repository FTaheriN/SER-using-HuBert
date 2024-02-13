import torch 
from utils import calc_acc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



def test(test_loader, model, device, loss_fn): 
    
    model.eval()
    
    predictions = []
    y_test = []
    total_test_acc = 0 
    total_test_loss = 0
 
    with torch.no_grad():
        for batch_idx, (audio_features, attention_mask, y) in enumerate(test_loader): #attention_mask,

                # optimizer.zero_grad()

                audio_features = audio_features.to(device)
                attention_mask = attention_mask.to(device)
                labels = y.to(device)

                # print(labels)
                
                prediction = model(audio_features, attention_mask)
                y_test += y.cpu()

                loss = loss_fn(prediction, labels)
                
                acc = calc_acc(prediction, labels)

                total_test_loss += loss.item()
                total_test_acc  += acc.item()

                predictions += torch.log_softmax(prediction, dim=1).argmax(dim=1).cpu()

    test_acc  = 100*total_test_acc/len(test_loader)
    test_loss = total_test_loss/len(test_loader) 
 
    print(f'Test Set Accuracy: % {test_acc:.3f}')
    return test_acc, predictions, y_test


def test_report(y_test, pred):
    # y_test = []
    # for batch_idx, (input_ids, token_type_ids, attention_mask, y) in enumerate(test_loader):
    #     y_test += y.cpu()

    print(classification_report(y_test, pred))
    cm2 = confusion_matrix(y_test, pred,)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
    disp2.plot()
    return