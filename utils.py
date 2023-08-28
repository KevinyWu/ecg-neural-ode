import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix


def anderson(f, x0, m=5, lam=1e-4, max_iter=50, tol=1e-2, beta = 1.0):
    """
    Anderson acceleration for fixed point iteration.
    """
    bsz, d, L = x0.shape
    X = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, d*L, dtype=x0.dtype, device=x0.device)
    X[:,0], F[:,0] = x0.view(bsz, -1), f(x0).view(bsz, -1)
    X[:,1], F[:,1] = F[:,0], f(F[:,0].view_as(x0)).view(bsz, -1)
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1
    
    res = []
    for k in range(2, max_iter):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype,device=x0.device)[None]
        
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)
        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m].view_as(x0)).view(bsz, -1)
        res.append((F[:,k%m] - X[:,k%m]).norm().item()/(1e-5 + F[:,k%m].norm().item()))
        if (res[-1] < tol):
            break
    return X[:,k%m].view_as(x0), res


def count_parameters(model):
    """
    Count number of tunable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch(loader, model, device, opt, lr_scheduler, epoch):
    """
    Training loop for one epoch.
    """
    print(f"Training... epoch {epoch}")
    total_loss, total_acc = 0.,0.
    model.train()
    batch_count = 0
    start = time.time()
        
    for X, y in loader:
        batch_count += 1
        curr_time = time.time()
        percent = round(batch_count/len(loader) * 100, 2)
        elapsed = round((curr_time - start)/60, 2)
        print(f"    Percent trained: {percent}%  Time elapsed: {elapsed} min", end='\r')
        
        X, y = X.to(device), y.to(device)
        yp = model(X)
        
        loss = nn.CrossEntropyLoss()(yp, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
         
        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

    acc = total_acc / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    lr_scheduler.step(loss)
    print(f"\n    Train acc: {round(acc, 3)}")
    print(f"    Train loss: {round(loss, 3)}")


def epoch_eval(loader, model, device):
    """
    Model evaluation.
    """
    print(f"Testing")
    total_loss, total_acc = 0.,0.
    model.eval()
    true_labels = []
    pred_logits = []
    
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        
        loss = nn.CrossEntropyLoss()(yp, y)
        total_acc += (yp.max(dim=1)[1] == y).sum().item()
        total_loss += loss.item() * X.shape[0]

        true_labels.append(y.cpu().data.numpy())
        pred_logits.append(yp.cpu().data.numpy())

    acc = total_acc / len(loader.dataset)
    loss = total_loss / len(loader.dataset)
    print(f"    Test acc: {round(acc, 3)}")
    print(f"    Test loss: {round(loss, 3)}")

    true_labels = np.concatenate(true_labels)
    pred_logits = np.concatenate(pred_logits)
    pred_labels = np.argmax(pred_logits, axis=1)
    print(classification_report(true_labels, pred_labels, zero_division=0.0))
    print(confusion_matrix(true_labels, pred_labels))
    print()