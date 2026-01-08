"""
Ultra-Fast Clean Baseline Experiment - 10 Rounds Only
NO Byzantine Attack - Pure federated learning baseline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import json
import time
from datetime import datetime
import os

device = torch.device('cpu')
print(f"Device: {device}")

class TinyNet(nn.Module):
    """Minimal CNN for fast testing"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(), nn.Linear(32*8*8, 64), nn.ReLU(), nn.Linear(64, 10)
        )
    def forward(self, x):
        return self.fc(self.conv(x))

def load_data(num_clients=10):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    data_dir = 'c:/Users/ADMIN/Documents/project/disertasis3/SIMULASI_EXPERIMENT/seminarproposal/experiments/data'
    train = datasets.CIFAR10(data_dir, True, transform, download=True)
    test = datasets.CIFAR10(data_dir, False, transform, download=True)
    
    # Dirichlet non-IID split
    labels = np.array(train.targets)
    client_idx = [[] for _ in range(num_clients)]
    for k in range(10):
        idx = np.where(labels == k)[0]
        np.random.shuffle(idx)
        props = np.random.dirichlet([0.5]*num_clients)
        splits = (props * len(idx)).astype(int)
        splits[-1] = len(idx) - splits[:-1].sum()
        pos = 0
        for c in range(num_clients):
            client_idx[c].extend(idx[pos:pos+splits[c]])
            pos += splits[c]
    
    loaders = [torch.utils.data.DataLoader(
        torch.utils.data.Subset(train, idx), batch_size=64, shuffle=True, num_workers=0
    ) for idx in client_idx]
    test_loader = torch.utils.data.DataLoader(test, batch_size=256, num_workers=0)
    return loaders, test_loader

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def aggregate(global_model, client_models):
    gd = global_model.state_dict()
    for k in gd:
        gd[k] = sum(m.state_dict()[k] for m in client_models) / len(client_models)
    global_model.load_state_dict(gd)

def train_fedprox(loaders, test_loader, rounds=10, epochs=1, lr=0.01, mu=0.01):
    print(f"\n=== FedProx (Clean) - {rounds} rounds ===")
    model = TinyNet()
    accs = []
    t0 = time.time()
    
    for r in range(rounds):
        gd = {k: v.clone() for k, v in model.state_dict().items()}
        clients = []
        
        for loader in loaders:
            local = TinyNet()
            local.load_state_dict(gd)
            opt = optim.SGD(local.parameters(), lr=lr, momentum=0.9)
            local.train()
            for _ in range(epochs):
                for x, y in loader:
                    opt.zero_grad()
                    loss = nn.CrossEntropyLoss()(local(x), y)
                    # Proximal term
                    prox = sum(((p - gd[n]) ** 2).sum() for n, p in local.named_parameters())
                    loss += (mu/2) * prox
                    loss.backward()
                    opt.step()
            clients.append(local)
        
        aggregate(model, clients)
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"R{r+1:2d}/{rounds} | {acc:.2f}% | {time.time()-t0:.1f}s")
    
    return accs

def train_feddyn(loaders, test_loader, rounds=10, epochs=1, lr=0.01, alpha=0.01):
    print(f"\n=== FedDyn (Clean) - {rounds} rounds ===")
    model = TinyNet()
    accs = []
    t0 = time.time()
    
    h = [{k: torch.zeros_like(v) for k, v in model.state_dict().items()} for _ in loaders]
    
    for r in range(rounds):
        gd = {k: v.clone() for k, v in model.state_dict().items()}
        clients = []
        
        for i, loader in enumerate(loaders):
            local = TinyNet()
            local.load_state_dict(gd)
            opt = optim.SGD(local.parameters(), lr=lr, momentum=0.9)
            local.train()
            for _ in range(epochs):
                for x, y in loader:
                    opt.zero_grad()
                    loss = nn.CrossEntropyLoss()(local(x), y)
                    # FedDyn reg
                    for n, p in local.named_parameters():
                        loss += (alpha/2) * ((p - gd[n]) ** 2).sum()
                        loss -= (h[i][n] * p).sum()
                    loss.backward()
                    opt.step()
            
            for n, p in local.named_parameters():
                h[i][n] = h[i][n] - alpha * (p.data - gd[n])
            clients.append(local)
        
        aggregate(model, clients)
        acc = evaluate(model, test_loader)
        accs.append(acc)
        print(f"R{r+1:2d}/{rounds} | {acc:.2f}% | {time.time()-t0:.1f}s")
    
    return accs

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    
    print("="*50)
    print("CLEAN BASELINE - NO BYZANTINE ATTACK")
    print("="*50)
    
    loaders, test_loader = load_data(10)
    
    fedprox_accs = train_fedprox(loaders, test_loader, rounds=10, epochs=1)
    feddyn_accs = train_feddyn(loaders, test_loader, rounds=10, epochs=1)
    
    results = {
        'experiment': 'clean_baseline_10r',
        'timestamp': datetime.now().isoformat(),
        'config': {'rounds': 10, 'clients': 10, 'attack': 'none'},
        'FedProx': {'final': fedprox_accs[-1], 'max': max(fedprox_accs), 'all': fedprox_accs},
        'FedDyn': {'final': feddyn_accs[-1], 'max': max(feddyn_accs), 'all': feddyn_accs}
    }
    
    out_file = f'c:/Users/ADMIN/Documents/project/disertasis3/SIMULASI_EXPERIMENT/seminarproposal/experiments/results/clean_baseline_10r_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"FedProx Clean: {fedprox_accs[-1]:.2f}%")
    print(f"FedDyn Clean:  {feddyn_accs[-1]:.2f}%")
    print(f"Saved: {out_file}")
