import torch
from tqdm import tqdm

test_acc = []
test_losses = []

def test(net, device, testloader, criterion):
    net.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)  # Get samples
            output = net(data)  # Get trained model output
            loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability

            correct += pred.eq(target).sum().item()

    loss /= len(testloader.dataset)
    test_losses.append(loss)
    test_acc.append(100. * correct / len(testloader.dataset))
    print(f'\nValidation set: Average loss: {loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({test_acc[-1]:.2f}%)\n')


def test_class_performance(net, device, testloader, classes):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        pbar = tqdm(testloader)
        for i, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            outputs = net(data)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == target).squeeze()
            for i in range(4):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
