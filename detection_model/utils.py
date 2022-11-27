
from __future__ import print_function, division
import torch
import time
import copy
from sklearn.metrics import classification_report

def train_model(model, dataloaders, device, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    """
    Create function to train a model using transfer learning
    Parameters:
        model: based model
        dataloaders: dataloader object
        device: detected device (cpu/gpu)
        dataset_sizes: size of the dataset
        criteion, optimizer, scheduler: required parameters
        num_epochs: number of training epochs
    Return:
        the best train model 
    """
    since = time.time()
    # First best model is the initial model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test(model, dataloaders, device):
    """
    Test function to get accuracy on Test set
    Parameters:
        model: trained model
        dataloaders: dataloader object
        device: detected device
    Return:
        Accuracy
    """
    model.eval()
    test_loss = 0
    correct = 0
    # Not calculate gradients
    with torch.no_grad():
        # Loop through test data in batch to get prediction and target value
        for data, target in dataloaders['test']:
            target.apply_(lambda x: (x))
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(dataloaders['test'].dataset)

    print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
         correct, len(dataloaders['test'].dataset),
        100. * correct / len(dataloaders['test'].dataset)))
    return 100. * correct / len(dataloaders['test'].dataset)

def get_classification_report(model, dataloaders, device):
    """
    Return a classification report using based function from scikit-learn
    Parameters:
        model: trained model
        dataloaders: dataloader object
        device: detected device
    Return:
        Classification report
    """
    total_pred = []
    total_target = []
    # Not calculating gradients
    with torch.no_grad():
         # Loop through test data in batch to get prediction and target value
        for data, target in dataloaders['test']:
            target.apply_(lambda x: (x))
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            
            target_list = target.squeeze(0).detach().cpu().numpy()
            pred_list = pred.squeeze(0).detach().cpu().numpy()
            total_pred.append(pred_list)
            total_target.append(target_list)
    target_ = [ii for i in total_target for ii in i]
    pred_ = [ii for i in total_pred for ii in i]
    report = classification_report(target_, pred_, target_names=['moire', 'non_moire'])
    return report
