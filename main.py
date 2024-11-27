
import torch as tro
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler






device= tro.device('cuda'if tro.cuda.is_available() else 'cpu')

data=load_breast_cancer()
x=data.data 
y=data.target


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=30)


scaler = StandardScaler()
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)


class BreastCancerNN(nn.Module): 
    def __init__(self):
        super(BreastCancerNN,self).__init__()
        self.fc1= nn.Linear(X_train_scaled.shape[1],64)
        self.hidden_layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(10)])
        self.fc2= nn.Linear(64,32)
        self.fc3= nn.Linear(32,2)
    
    def forward(self,x): 
        x=nn.functional.relu(self.fc1(x))
        for layer in self.hidden_layers:
            x=nn.functional.relu(layer(x))

        x=nn.functional.relu(self.fc2(x))
        x= self.fc3(x) 
        return(x)


model= BreastCancerNN().to(device )
criteration= nn.CrossEntropyLoss() 
optimizer=optim.Adam(model.parameters(),lr=0.001) # can play with the leaning rate but 0.001 is optimal 


#Training 
num_epochs= 200
for epoch in range(num_epochs): 
    model.train() 
    inputs= tro.tensor(X_train_scaled,dtype= tro.float32) .to(device)
    labels= tro.tensor(Y_train,dtype=tro.long).to(device)

    optimizer.zero_grad() 
    outputs=model(inputs)
    loss= criteration(outputs,labels)
    loss.backward() 
    optimizer.step() 

    if (epoch+1)%5 ==0: 
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss:{loss.item():4f}') 


model.eval() 
with tro.no_grad(): 
    test_inputs= tro.tensor(X_test_scaled,dtype=tro.float32).to(device)
    test_labels= tro.tensor(Y_test,dtype=tro.long).to(device)
    test_outputs=model(test_inputs)
    _, predicted=tro.max(test_outputs,1)
    accuracy = (predicted == test_labels).float().sum().item() / len(test_labels)


    print(f'Test Accuracy: {accuracy*100:.2f}%')
