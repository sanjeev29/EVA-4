import torch
from tqdm import tqdm

def train(net, device, trainloader, optimizer, criterion, epoch, train_acc, train_losses):
    net.train()
    pbar = tqdm(trainloader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get the inputs
        data, target = data.to(device), target.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # predict
        y_pred = net(data)

        # loss
        loss = criterion(y_pred, target)

        # backprop
        loss.backward()
        optimizer.step()

        # update pbar tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Epoch= {epoch} Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    
    train_losses.append(loss.item())
    train_acc.append(100 * correct / processed)
