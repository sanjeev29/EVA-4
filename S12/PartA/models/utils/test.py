import torch
from tqdm import tqdm

correct_pred = []
incorrect_pred = []

def test(net, device, testloader, criterion, last_epoch, test_acc, test_losses):
    net.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        for data, target in testloader:
            img_batch = data
            data, target = data.to(device), target.to(device)  # Get samples
            output = net(data)  # Get trained model output
            loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=False)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))
            
            if last_epoch:
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_pred) < 25:
                        incorrect_pred.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })
                    elif list(result)[i] and len(correct_pred) < 25:
                        correct_pred.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })

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
