import torch
import numpy as np
import torch.nn as nn
from networks.resnet import resnet50
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, classification_report
import matplotlib.pyplot as plt
from options.test_options import TestOptions
from data import create_dataloader

from tqdm import tqdm


def validate(model, opt):
    data_loader = create_dataloader(opt)
    criterion = nn.CrossEntropyLoss() if opt.num_classes > 1 else nn.BCEWithLogitsLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_feats = []

    with torch.no_grad():
        for img, label in tqdm(data_loader):
            img, label = img.to('cuda'), label.to('cuda')
            pred = model(img)
            loss = criterion(pred, label)
            total_loss += loss.item() * img.size(0)
            _, predicted = torch.max(pred, 1)
            total_correct += (predicted == label).sum().item()
            total_samples += label.size(0)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    purity = compute_purity(np.array(all_preds), np.array(all_labels))
    nmi = compute_nmi(np.array(all_preds), np.array(all_labels))
    # print(f'Accuracy: {accuracy:.4f}, Purity: {purity:.4f}, NMI: {nmi:.4f}, Validation Loss: {avg_loss:.4f}')
    
    class_report = classification_report(all_labels, all_preds, target_names=opt.class_names, digits=4, zero_division=0)
    print('val class_report:')
    print(class_report)
    
    return accuracy, purity, nmi, avg_loss, class_report

def compute_purity(pred, labels):
    """
    compute purity of the predicted 'cluster'
    INPUT: pred  (N) predicted labels
           labels (N) groundtruth labels
    Output: scalar
    """
    conf = confusion_matrix(labels, pred)
    purity = 0
    for i in range(len(conf)):
        purity += conf[:, i].max()
    purity /= conf.sum()
    return purity 

def compute_nmi(pred, labels):
    """
    normalized mutual information
    """
    return normalized_mutual_info_score(labels, pred)

