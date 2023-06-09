import torch.nn as nn
import torch.optim as optim
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.hl1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.hl2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.out = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.hl1(x)
        output = self.relu1(output)
        output = self.hl2(output)
        output = self.relu2(output)
        output = self.out(output)
        output = self.sigmoid(output)
        return output

# Veri setlerini oku
training_dataset = pd.read_csv('cure_the_princess_train.csv')
validation_dataset = pd.read_csv('cure_the_princess_validation.csv')
test_dataset = pd.read_csv('cure_the_princess_test.csv')

# Tensore çevir
training_data = torch.tensor(training_dataset.iloc[:, :-1].values, dtype=torch.float)
training_labels = torch.tensor(training_dataset.iloc[:, -1].values)
validation_data = torch.tensor(validation_dataset.iloc[:, :-1].values, dtype=torch.float)
validation_labels = torch.tensor(validation_dataset.iloc[:, -1].values)
test_data = torch.tensor(test_dataset.iloc[:, :-1].values, dtype=torch.float)
test_labels = torch.tensor(test_dataset.iloc[:, -1].values)

#SEED değerini set et
SEED = 180401119
torch.manual_seed(SEED)

# input, hidden layerlerin ve output layerin boyutlarını belirle
input_size = training_data.size()[1] 
hidden_size1 = 100
hidden_size2 = 50
output_size = 2 
patience = 10

#Modeli oluştur
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

# loss function ve optimizer'i tanımla
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# batch size ve epoch sayısını belirle
batch_size = 16
num_epochs = 50

# Early stopping degiskenleri
patience = 10
best_val_loss = float('inf')
best_model = None
counter = 0

#train ve val losslarını tutmak için iki liste oluştur
list_train_loss, list_val_loss = [], []

# Model eğitme döngüsü
for epoch in range(num_epochs):
    train_loss = 0.0
    for i in range(0, len(training_data), batch_size):
        # batchi hazirlayalim
        inputs = training_data[i:i+batch_size]
        labels = training_labels[i:i+batch_size]
        inputs = inputs.view(-1, input_size)
        # Forward 
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # L1 regularizasyonu
        l1_lambda = 0.01
        l1_regularization = 0.0
        for param in model.parameters():
            l1_regularization += torch.norm(param, 1)
        loss += l1_lambda * l1_regularization

        # L2 regularizasyonu
        l2_lambda = 0.01
        l2_regularization = 0.0
        for param in model.parameters():
            l2_regularization += torch.norm(param, 2)
        loss += l2_lambda * l2_regularization
        
        # Backward ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size # bir batch için train lossu
        
    # Epoch için ortalama lossu hesapla ve listeye ekle 
    t_epoch_loss = train_loss / len(training_data) #
    list_train_loss.append(t_epoch_loss)
    
    #Validasyon seti için de aynı işlemler
    with torch.no_grad():
        model.eval()
        val_loss = 0.0
        for j in range(0, len(validation_data), batch_size):
            inputs = validation_data[j:j+batch_size]
            labels = validation_labels[j:j+batch_size]
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * batch_size
        v_epoch_loss = val_loss / len(training_data)
        list_val_loss.append(v_epoch_loss)
    
        if v_epoch_loss < best_val_loss:
            best_val_loss = v_epoch_loss
            best_model = model.state_dict()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping at epoch {epoch}.')
                break
          
    model.train()
from sklearn.metrics import precision_score, recall_score, f1_score
model.load_state_dict(best_model)
model.eval()
test_outputs = model(test_data)
test_predicted = torch.argmax(test_outputs, dim=1)
accuracy = (test_predicted == test_labels).sum().item() / len(test_labels)
precision = precision_score(test_labels, test_predicted)
recall = recall_score(test_labels, test_predicted)
f1 = f1_score(test_labels, test_predicted)
print("Accuracy:", accuracy*100)
print("Precision:", precision*100)
print("Recall:", recall*100)
print("F1 Score:", f1*100)
# RESULTS: Accuracy: 84.84455958549223, Precision: 79.26565874730022, Recall: 94.58762886597938, F1 Score: 86.25146886016452
