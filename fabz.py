
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


data = load_breast_cancer()
x = data.data
y = data.target


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.25, random_state=45)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_train = torch.tensor(Y_train, dtype=torch.float32).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).to(device)


class NeuralNer(nn.Module):
    def __init__(self, inputsize, hiddensize, outputsize):
        super(NeuralNer, self).__init__()
        self.fc1 = nn.Linear(inputsize, hiddensize)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiddensize, outputsize)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # Define the forward pass
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Hyperparameters
inputsize = X_train.shape[1]
hiddensize = 256
outputsize = 1
learning_rate = 0.001
nums_epochs = 200


model = NeuralNer(inputsize, hiddensize, outputsize).to(device)
criterion = nn.BCELoss()
optmzer = optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(nums_epochs):
    model.train()
    optmzer.zero_grad()  # Zero gradients to prevent accumulation
    outputs = model(X_train)  # Forward pass
    loss = criterion(outputs, Y_train.unsqueeze(1))  # Compute loss
    loss.backward()  # Backward pass
    optmzer.step()  # Update weights

   
    with torch.no_grad():
        prediktor = outputs.round()
        hari = (prediktor == Y_train.unsqueeze(1)).float().sum()
        accuracy = hari / Y_train.size(0)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{nums_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%')


with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    test_prediktor = test_outputs.round()
    test_hari = (test_prediktor == Y_test.unsqueeze(1)).float().sum()
    test_accuracy = test_hari / Y_test.size(0)
    print(f'Test Accuracy: {test_accuracy.item() * 100:.2f}%')
