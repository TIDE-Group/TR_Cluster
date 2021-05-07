from typing import Text, Union, List
import utils
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import random
from transformers import BigBirdTokenizer, BigBirdModel, BertTokenizer, BertModel, AutoTokenizer, AutoModel
from torch import autograd
import torch.nn as nn, torch.optim as optim
import torch.cuda as cuda
import torch.nn.functional as F
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ClassifierModel(nn.Module):
    def __init__(
        self, 
        tokenizer:Union[BertTokenizer, BigBirdTokenizer, AutoTokenizer], 
        featurizer:Union[BertModel, BigBirdModel, AutoModel], 
        cls:int = 2,
        dropout:float = 0.5
    ):
        super(ClassifierModel, self).__init__()
        self.tokenizer = tokenizer
        self.featurizer = featurizer
        self.cls = cls

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.Linear(512, cls)
        )

    def forward(self, texts:List[str], log_softmax:bool=True):
        tokens = self.tokenizer(text=texts, return_tensors='pt', padding=True)
        for key in tokens.data.keys():
            tokens.data[key] = tokens.data[key][:, :512]
        if cuda.is_available():
            for key in tokens.data.keys():
                tokens.data[key] = tokens.data[key].cuda()
        feature = self.featurizer.forward(**tokens).pooler_output
        outputs = self.fc.forward(feature)
        if log_softmax:
            outputs = F.log_softmax(outputs, dim=-1)
        return outputs

class ClassifierTrainer(object):
    def __init__(
        self, 
        model:ClassifierModel, 
        lr:float=1e-3, 
        weight_decay:float=1e-5
    ):
        super().__init__()
        self.model = model
        self.optimizer = optim.Adam(
            filter(lambda p : p.requires_grad, model.parameters()),
            lr=lr,
            weight_decay=weight_decay
        )
        self.criterion = nn.NLLLoss()
    
    def train_a_epoch(self, dataloader:DataLoader, epoch_num:int):
        self.model.train()

        loss_list = []
        pred_list = []
        label_list = []

        with autograd.detect_anomaly():
            for i_batch, (texts, labels) in \
                tqdm(enumerate(dataloader), total=len(dataloader), desc = 'training epoch {}'.format(epoch_num)):
                cuda.empty_cache()
                self.optimizer.zero_grad()
                outputs = self.model.forward(texts)

                loss = self.criterion.forward(outputs, labels)
                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                label_list.append(labels.cpu().numpy())
        
        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        acc, Pmacro, Rmacro, f1 = utils.calMacro(y_pred, y_true)
        cuda.empty_cache()
        return loss, acc, Pmacro, Rmacro, f1
    
    def evaluate_a_epoch(self, dataloader:DataLoader, epoch_num:int):
        self.model.eval()

        loss_list = []
        pred_list = []
        label_list = []

        with torch.no_grad():
            for i_batch, (texts, labels) in \
                tqdm(enumerate(dataloader), total=len(dataloader), desc = 'evaluating epoch {}'.format(epoch_num)):
                cuda.empty_cache()
                outputs = self.model.forward(texts)
                loss = self.criterion.forward(outputs, labels)

                loss_list.append(loss.item())
                pred_list.append(torch.argmax(outputs, dim=-1).cpu().numpy())
                label_list.append(labels.cpu().numpy())
        
        cuda.empty_cache()
        y_pred = np.concatenate(pred_list)
        y_true = np.concatenate(label_list)

        loss = np.mean(loss_list)
        
        acc, Pmacro, Rmacro, f1 = utils.calMacro(y_pred, y_true)
        
        return loss, acc, Pmacro, Rmacro, f1

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    batch_size = 2
    epoch_cnt = 150
    dropout = 0.5
    lr = 1e-3
    wd = 1e-5

    seed = 1
    np.random.seed(seed)
    random.seed(seed)
    cuda.manual_seed(seed)
    torch.manual_seed(seed)

    data = pd.read_excel('./cnn_labeled_all.xlsx')
    data = data[data['value'].notna()]
    data = data[data['label'].notna()]
    texts = data['value'].tolist()
    labels = data['label'].values
    labels = (labels == 0).astype(int)

    idx = np.array(range(len(data)), dtype=int)
    np.random.shuffle(idx)

    train_indice = idx[:int(len(data) * 0.7)]
    test_indice = idx[int(len(data) * 0.7):]

    train_texts = [texts[index] for index in train_indice]
    test_texts = [texts[index] for index in test_indice]
    train_labels, test_labels = labels[train_indice], labels[test_indice]

    train_dataset = utils.ClassifierDataSet(train_texts, train_labels)
    test_dataset = utils.ClassifierDataSet(test_texts, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.classifier_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.classifier_collate_fn)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').cuda()

    classifier_model = ClassifierModel(tokenizer=tokenizer, featurizer=model, cls=2, dropout=dropout).cuda()
    trainer = ClassifierTrainer(classifier_model, lr=lr, weight_decay=wd)

    train_losses, train_epochs, train_f1s = [], [], []
    test_losses, test_epochs, test_f1s = [], [], []
    best_test_f1 = 0
    best_test_loss = 1e9
    best_epoch = 0

    result_str = '{} Loss: {:.4f} Acc: {:.2f}% Pmacro: {:.2f}% Rmacro {:.2f}% F1: {:.2f}%'
    for i in range(epoch_cnt):
        train_loss, train_acc, train_Pmacro, train_Rmacro, train_f1 = \
            trainer.train_a_epoch(train_loader, i)
        print(result_str.format('Train', train_loss, train_acc * 100, train_Pmacro * 100, train_Rmacro * 100, train_f1 * 100))
        
        train_epochs.append(i)
        train_losses.append(train_loss)
        train_f1s.append(train_f1)

        if (i + 1) % 5 == 0:
            test_loss, test_acc, test_Pmacro, test_Rmacro, test_f1 = \
                trainer.evaluate_a_epoch(test_loader, i)
            test_epochs.append(i)
            test_losses.append(test_loss)
            test_f1s.append(test_f1)

            print(result_str.format(
                'Test', test_loss, test_acc * 100, test_Pmacro * 100, test_Rmacro * 100, test_f1 * 100
            ))

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                best_test_loss = test_loss
                best_epoch = i
                trainer.save('./pkls/classifier.pkl')
            print('Best Test f1: {:.4f}'.format(best_test_f1))
            print('Best Test loss: {:.4f}'.format(best_test_loss))
            print('Best Test epoch: {}'.format(best_epoch))
        if i - best_epoch > 20:
            break

    print('Best Test f1: {:.4f}'.format(best_test_f1))
    print('Best Test loss: {:.4f}'.format(best_test_loss))
    print('Best Test epoch: {}'.format(best_epoch))
    print(train_epochs)
    print(train_losses)
    print(train_f1s)
    print(test_epochs)
    print(test_losses)
    print(test_f1s)
        
    plt.figure(figsize=(12, 7), dpi=600)
    p1 = plt.plot(train_epochs, train_losses, color='b', label='训练集', linewidth=1.5)
    p2 = plt.plot(test_epochs, test_losses, color='r', label='测试集', linewidth=1.5)
    plt.xlabel('epoch')
    plt.title('训练集和测试集loss的变化')
    plt.savefig('./img/bilstm_att_loss.png', dpi=600)
    plt.legend(['训练集', '测试集'])
    
    plt.show()

    plt.figure(figsize=(12, 7), dpi=600)
    p1 = plt.plot(train_epochs, train_f1s, color='b', label='训练集', linewidth=1.5)
    p2 = plt.plot(test_epochs, test_f1s, color='r', label='测试集', linewidth=1.5)
    plt.xlabel('epoch')
    plt.title('训练集和测试集F1的变化')
    plt.savefig('./img/bilstm_att_f1.png', dpi=600)
    plt.legend(['训练集', '测试集'])
    plt.show()
    