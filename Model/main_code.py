import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
threshold = 0.5
num_epochs = 70
batch_size1 = 8
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
class MolecularDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, names_list, targets):
        self.smiles_list = smiles_list
        self.names = names_list  
        self.targets = targets
    def __len__(self):
        return len(self.smiles_list)
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        generator = GetMorganGenerator(radius=4, fpSize=1024)
        fingerprint = generator.GetFingerprint(mol)  
        fingerprint = np.array(fingerprint)
        return torch.tensor(fingerprint, dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32), self.names[idx]
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, names_list):
        self.smiles_list = smiles_list
        self.names_list = names_list
    def __len__(self):
        return len(self.smiles_list)        
    def __getitem__(self, idx):
        mol = Chem.MolFromSmiles(self.smiles_list[idx])
        generator = GetMorganGenerator(radius=4, fpSize=1024)
        fingerprint = generator.GetFingerprint(mol)
        fingerprint = np.array(fingerprint)
        return torch.tensor(fingerprint, dtype=torch.float32), self.names_list[idx]        
    def get_smiles_from_names(self, names_to_find):
        smiles_found = []
        for name in names_to_find:
            try:
                idx = self.names_list.index(name)
                smiles_found.append(self.smiles_list[idx])
            except ValueError:
                print(f"No SMILES found for compound name: {name}")
        return smiles_found
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
class ModelTrainer:
    def __init__(self, model, criterion, optimizer, visualizer=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.visualizer = visualizer
    def train(self, train_dataloader, num_epochs=70):
        train_losses = []
        for epoch in range(num_epochs):
            self.model.train()
            running_train_loss = 0.0
            for fingerprints, targets, _ in train_dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(fingerprints.float())
                loss = self.criterion(outputs, targets.float().view(-1, 1))
                loss.backward()
                self.optimizer.step()
                running_train_loss += loss.item() * fingerprints.size(0)
            epoch_loss = running_train_loss / len(train_dataloader.dataset)
            train_losses.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {epoch_loss:.4f}")
        return train_losses      
    def evaluate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for fingerprints, targets, _ in dataloader:
                outputs = self.model(fingerprints.float())
                loss = self.criterion(outputs, targets.float().view(-1, 1))
                running_loss += loss.item() * fingerprints.size(0)
                predicted = torch.round(outputs)
                correct += (predicted == targets.view(-1, 1)).sum().item()
                total += targets.size(0)
        loss = running_loss / len(dataloader.dataset)
        accuracy = correct / total
        return loss, accuracy        
decoy_scores = pd.read_csv('./dataset.csv')
data_shuffled = decoy_scores.sample(frac=1, random_state=seed).reset_index(drop=True)
smiles_list = data_shuffled['SMILES'].tolist()
X_train = [Chem.CanonSmiles(smiles) if Chem.MolFromSmiles(smiles) is not None else None for smiles in smiles_list]
y_train = data_shuffled['Actividad'].tolist()
names_list = data_shuffled['Title'].tolist()
dataset = MolecularDataset(X_train, names_list, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size1, shuffle=True)
print(f"Filtered to {len(X_train)} valid train SMILES from {len(smiles_list)} total entries.")
#Blind set
blind = pd.read_csv('./for_blind_nluc.csv')
blind_shuffled = blind.sample(frac=1, random_state=seed).reset_index(drop=True)
smiles_blind = blind_shuffled['SMILES'].tolist()
y_blind_full = blind_shuffled['Actividad'].tolist()
names_blind_full = blind_shuffled['Title'].tolist()
X_blind = []
y_blind = []
names_blind = []
for smiles, y, name in zip(smiles_blind, y_blind_full, names_blind_full):
    if pd.notna(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol:
            canon = Chem.MolToSmiles(mol, canonical=True)
            X_blind.append(canon)
            y_blind.append(y)
            names_blind.append(name)
print(f"Filtered to {len(X_blind)} valid blind SMILES from {len(smiles_blind)} total entries.")
#Train
model = NeuralNetwork(input_size=1024, hidden_size=32)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.001)
trainer = ModelTrainer(model, criterion, optimizer)
start_time = time.time()
train_losses = trainer.train(dataloader, num_epochs=num_epochs)
print("Training Losses per Epoch:", train_losses)
#Blind Testing
blind_dataset = MolecularDataset(X_blind, names_blind, y_blind)
blind_dataloader = DataLoader(blind_dataset, batch_size=batch_size1, shuffle=False)
model.eval()
blind_predictions = []
compound_names = []
with torch.no_grad():
    for features, names, _ in blind_dataloader: 
        output = model(features.float())
        blind_predictions.extend(output.cpu().numpy())
        compound_names.extend(names)
blind_predictions_df = pd.DataFrame(blind_predictions)
blind_predictions_df.insert(0, "Compound Name", compound_names)  
blind_predictions_df.to_csv("blind_predictions.csv", index=False)
rounded_values = list(map(lambda x: 1 if x >= threshold else 0, blind_predictions))
f1 = f1_score(y_blind, rounded_values)
print("F1 Score:", f1)
accuracy = accuracy_score(y_blind, rounded_values)
print("Accuracy:", accuracy)
fpr, tpr, thresholds = roc_curve(y_blind, blind_predictions)
auc_score = auc(fpr, tpr)
print("AUC score:", auc_score)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
cm = confusion_matrix(y_blind, rounded_values)
tn, fp, fn, tp = cm.ravel()
print("\nConfusion Matrix:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("confusion_matrix_detailed.png")
plt.show()
#Prediction
eval_scores = pd.read_csv('./unique_compounds.csv')
eval_smiles_list = eval_scores.iloc[:, 1].tolist()
eval_names_list = eval_scores.iloc[:, 0].tolist()
eval_dataset = EvalDataset(eval_smiles_list, eval_names_list)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size1, shuffle=False)
model.eval()
eval_predictions = []
compound_names = []
smiles_strings = []
with torch.no_grad():
    for features, names in eval_dataloader:
        output = model(features.float())
        eval_predictions.extend(output.cpu().numpy())
        compound_names.extend(names)
        smiles_strings.extend(eval_dataset.get_smiles_from_names(names))
rounded_values = [1 if x >= threshold else 0 for x in eval_predictions]
eval_predictions_df = pd.DataFrame({
    "Compound Name": compound_names,
    "SMILES": smiles_strings,
    "Prediction": rounded_values
})
selected_compounds_df = eval_predictions_df[eval_predictions_df['Prediction'] == 1]
selected_compounds_df.to_csv("selected_compounds.csv", index=False)        
