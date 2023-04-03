from flytekit import Resources, task

# Importing all dependencies
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
import pathlib

# Calling dataclasses
@dataclass_json
@dataclass
class Hyperparameters(object):
    file_path: str = "data/data.csv"
    test_size: float = 0.2
    random_state: int = 66
    lr: float = 0.001
    epochs: int = 2000


# Instantiating Hyperparameters class
hp = Hyperparameters()


# Collecting data
def collect_data(file_path):
    df=pd.read_csv((pathlib.Path(__file__).parent / filepath).resolve())
    X=df.drop('Outcome',axis=1).values### independent features
    y=df['Outcome'].values###dependent features
    return X, y

# Splitting train and test datasets and turning them into tensors
def split_dataset(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    return torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.LongTensor(y_train), torch.LongTensor(y_test)

# Creating ANN model with PyTorch
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

# Training model
def train(model, epochs, X_train, y_train, loss_function, optimizer):
    final_losses=[]
    for i in range(epochs):
        i=i+1
        y_pred=model.forward(X_train)
        loss=loss_function(y_pred,y_train)
        final_losses.append(loss)
        if i%10==1:
            print(f"Epoch: {i}    Loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Saving and loading model
    torch.save(model, "model/model.pt")
    model = torch.load("model/model.pt")
    return model


# Running workflow
@task(requests=Resources(cpu="2",mem="2Gi",storage="0Gi",ephemeral_storage="0Gi"),limits=Resources(cpu="2",mem="2Gi",storage="0Gi",ephemeral_storage="0Gi"),retries=3)
def run_wf(hp: Hyperparameters) ->ANN_Model:
    # Instantiating X and y
    X, y = collect_data(hp.file_path)
    # Instantiating X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_dataset(X=X, y=y, test_size=hp.test_size, random_state=hp.random_state)
    # Instantiate ANN_Model
    torch.manual_seed(20)
    model=ANN_Model()
    # Defining the loss function and the optimizer
    loss_function=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=hp.lr)
    # Predicting on X_test
    return train(model, hp.epochs, X_train, y_train, loss_function, optimizer)



if __name__=="__main__":
    run_wf(hp=hp)