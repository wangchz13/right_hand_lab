import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

device = 'cuda:0'

# 假设你的数据为torch.Tensor类型，X是输入数据，y是标签
# 数据预处理
# 你需要自行定义数据的加载和预处理过程
# 例如，将数据转换成PyTorch的Tensor类型，并进行归一化或标准化

data = pd.read_csv(r"\\wsl.localhost\Ubuntu\home\wcz\github\iTransformer\dataset\GAS\new_gas_drop_null.csv",index_col='time')
data.columns = list(range(1, len(data.columns) + 1))
data = data.iloc[:, :100]
data = data.T
reshaped_data = data.values.reshape((100, 539, 48))

y = np.arange(100)
y = np.repeat(y,539)
y_one_hot = np.eye(100)[y]
X = reshaped_data.reshape(-1,48)
X = torch.tensor(X).float().to(device)
y = torch.tensor(y_one_hot).to(device)
# 划分训练集和测试集
total_data = TensorDataset(X, y)
train_size = int(0.8 * len(total_data))
test_size = len(total_data) - train_size
train_data, test_data = random_split(total_data, [train_size, test_size])

# 定义数据加载器
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 定义模型
class MyModel(nn.Module):
    def __init__(self,num_classes):
        super(MyModel, self).__init__()
        self.rnn = nn.LSTM(input_size=48, hidden_size=256, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(512, num_classes)  # num_classes为用户标签的类别数
        # self.rnn = nn.LSTM(input_size=48, hidden_size=128, batch_first=True, bidirectional=False)
        # self.fc = nn.Linear(128, num_classes)  # num_classes为用户标签的类别数
        self.activ = nn.GELU()
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out)  # 取最后一个时间步的输出作为分类的依据
        return self.activ(out)

# 初始化模型、损失函数和优化器
model = MyModel(100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print('loss:',loss,'epoch:',epoch)
        loss.backward()
        optimizer.step()

# 在测试集上评估模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = np.eye(100)[predicted.to('cpu')]
        total += labels.size(0)
        for i in range(len(inputs)):
            pred = np.array_equal(labels[i].to('cpu'),predicted[i])
            if pred is True:
                correct += 1

accuracy = correct / total
print(f'Test Accuracy: {accuracy}')
