import os
import numpy as np
import torch
from datetime import datetime
from Dataloaders.Dataloader import BaselineDataloader
from Model.SimpleCnn import CNN
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix



if __name__ == "__main__":
    # experimentul pentru care vrem sa testam performanta modelului
    exp_name = "baseline2021-11-01-09_28_42"
    exp_path = os.path.join("experiments", exp_name)


    # create a folder where to append all the test results
    res_path = os.path.join(exp_path, "test_results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)


    # For reproductibility
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Dataset
    test_dataset = BaselineDataloader("Data/faces_dataset_preprocessed", split="split.json", phase="test")

    # Dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Model
    model = CNN()
    model.load_state_dict(torch.load(os.path.join(exp_path, "Epoch79_Error0.02")))
    model.eval()

    # evaluation
    y_true = []
    y_pred = []

    bad_items = []

    classes_mapping = {0:'nomask', 1:'maskwrong', 2:'mask'}

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # forward pass
            output = model(images)

            pred = torch.max(output, 1)[1].data.squeeze().item()
            gr = labels.item()

            y_true.append(gr)
            y_pred.append(pred)

            if pred != gr:
                bad_items.append((images.squeeze().cpu().detach().numpy(), classes_mapping[gr], classes_mapping[pred]))

    # compute metrics
    scores = f1_score(y_true, y_pred, average=None)
    print(f"F1 scores are {scores}")

    # plot confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Ground truth', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    # plt.show()
    plt.savefig(os.path.join(res_path, "confusion_matrix.png"))
    plt.clf()

    # save bad cases
    for idx, data in enumerate(bad_items):
        img, gr, pred = data
        
        plt.imshow(img, cmap='gray')
        plt.savefig(os.path.join(res_path, f"Image{idx}_{gr}_{pred}"))


