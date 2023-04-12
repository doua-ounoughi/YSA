import torch

X = torch.tensor([[1, 2, 3], [4, 5, 6]],dtype=torch.float)
# Aktivasyon fonksiyonlarını tanımlayalım
def tanh_act(x):
    return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))

def sigmoid_act(x):
    return 1 / (1 + torch.exp(-x))
    
    
import torch.nn as nn
torch.manual_seed(1)
class YSA(nn.Module):
    def __init__(self, input_size):
        super(YSA, self).__init__()
        self.hidden_layer = nn.Linear(input_size, 50)
        self.output_layer = nn.Linear(50, 1)

    def forward(self, x):
        x = tanh_act(self.hidden_layer(x))
        x =  sigmoid_act(self.output_layer(x))
        return x
        
model = YSA(input_size=X.shape[1])
result = model(X)
print(result)
