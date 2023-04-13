import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

#Veri setlerini oku
training_dataset = pd.read_csv('cure_the_princess_train.csv')
validation_dataset = pd.read_csv('cure_the_princess_validation.csv')
#Tensore cevir
training_data = torch.tensor(training_dataset.iloc[:, :-1].values, dtype=torch.float)
training_labels = torch.tensor(training_dataset.iloc[:, -1].values)
validation_data = torch.tensor(validation_dataset.iloc[:, :-1].values, dtype=torch.float)
validation_labels = torch.tensor(validation_dataset.iloc[:, -1].values)

#SEED degerini set et
SEED = 180401119
torch.manual_seed(SEED)
#input, hidden layerlerin ve output layerin boyutlarını belirle
input_size = training_data.size()[1] 
hidden_size1 = 100
hidden_size2 = 50
output_size = 2 

#Modeli olustur
model = MLP(input_size, hidden_size1, hidden_size2, output_size)

#loss function ve optimizer'i tanimla
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#batch size ve epoch sayisini belirle
batch_size = 16
num_epochs = 50

#train ve val losslarini tutmak icin iki liste olustur.
list_train_loss, list_val_loss = [], []

#Model egitme dongusu
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
        
        # Backward ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size #bir batch icin train lossu.
        
    #Epoch icin ortalama lossu hesapla ve listeye ekle 
    t_epoch_loss = train_loss / len(training_data) #
    list_train_loss.append(t_epoch_loss)
    
#Validasyon seti icin de ayni islemler
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
        list_val_loss.append(loss.item())

            
    model.train()
    print('Epoch [%d/%d], Training Loss: %.4f , Validation Loss: %.4f' % (epoch+1, num_epochs, t_epoch_loss,v_epoch_loss))

import matplotlib.pyplot as plt

plt.plot(list_train_loss, label='Training Loss')
plt.plot(list_val_loss, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

