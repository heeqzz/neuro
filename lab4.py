import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import torch.nn as nn 
import pandas as pd
from sklearn.preprocessing import StandardScaler  # 1. Импорт нормализатора

df = pd.read_csv('dataset_simple.csv')

# 2. Нормализация данных (Критически важно!)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Нормализуем признаки X
X_scaled = scaler_X.fit_transform(df.iloc[:, [0, 1]].values)
# Нормализуем целевую переменную y (нужно reshape для scaler)
y_scaled = scaler_y.fit_transform(df.iloc[:, 2].values.reshape(-1, 1))

# Превращаем в тензоры
X = torch.Tensor(X_scaled)
y = torch.Tensor(y_scaled)  

class NNet_regression(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet_regression, self).__init__() # Исправлено наследование
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )
    
    def forward(self, X):
        return self.layers(X)

# Параметры сети
inputSize = X.shape[1]
hiddenSizes = 20
outputSize = 1 

net = NNet_regression(inputSize, hiddenSizes, outputSize)

# 3. MSELoss часто лучше для начала обучения
lossFn = nn.MSELoss() 


optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

epochs = 2000 
for i in range(epochs):
    pred = net.forward(X)
    # y имеет размер (N, 1), pred тоже (N, 1). Squeeze делать не обязательно для MSE, но можно
    loss = lossFn(pred, y)  
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0: 
        print(f'Эпоха {i}: Ошибка (MSE) = {loss.item():.6f}')


with torch.no_grad():
    pred = net.forward(X)

err = torch.mean(torch.abs(y - pred)) 
print(f'\nИтоговая ошибка (MAE) на нормализованных данных: {err.item():.6f}')

