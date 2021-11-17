from comet_ml import Experiment
import numpy as np
import torch
from datetime import datetime 
import os
from Dataloaders.Dataloader import BaselineDataloader
from Model.SimpleCnn import CNN
from matplotlib import pyplot as plt

if __name__ == "__main__":
    # Experiment part
    exp_name = "baseline"
    exp_path = os.path.join("experiments", exp_name + datetime.today().strftime('%Y-%m-%d-%H_%M_%S'))
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Create an experiment with your api key
    experiment = Experiment(
    api_key="AEieUslWZezS1n4bMfAYu1tWs",
    project_name="maskrecognitionproject",
    workspace="andrey2405",
    )
    
    # For reproductibility
    random_seed = 1
    torch.manual_seed(random_seed)

    # Hyperparameters
    epochs = 100
    batch_size_train = 32
    learning_rate = 0.003
    
    # Run on GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device used: {device}")

    # Dataset
    train_dataset = BaselineDataloader("Data/faces_dataset_preprocessed", split="split.json", phase="train")
    val_dataset = BaselineDataloader("Data/faces_dataset_preprocessed", split="split.json", phase="val")

    # Dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Model
    model = CNN()

    # Afisarea numarului de parametri antrenabili ai modelului
    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Numarul total de parametri antrenabili ai modelului: {model_total_params}")
    experiment.log_parameter("nr_of_model_params", model_total_params)

    # Definirea loss-ului, functia NegativeLogLikeliHood
    criterion = torch.nn.NLLLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ###################### Training #######################
    errors_train = []
    errors_validation = []

    # Training loop
    for epoch in range(epochs):
        # O lista unde vom stoca erorile temporare epocii curente
        temporal_loss_train = [] 
        
        # Functia .train() trebuie apelata explicit inainte de antrenare
        model.train()     

       # Iteram prin toate sample-urile generate de dataloader
        for images, labels in train_loader:   
            images, labels = images.to(device), labels.to(device)
            
            # Clean the gradients
            optimizer.zero_grad()
            
            # Forward propagation
            output = model(images)
            
            t = model.conv_layer1(images)
            experiment.log_image(t[0, 0].cpu().detach().numpy(), name=f"Epoch{epoch}_FM0")
            experiment.log_image(t[0, 1].cpu().detach().numpy(), name=f"Epoch{epoch}_FM1")
            experiment.log_image(t[0, 2].cpu().detach().numpy(), name=f"Epoch{epoch}_FM2")
            experiment.log_image(t[0, 3].cpu().detach().numpy(), name=f"Epoch{epoch}_FM3")
            experiment.log_image(t[0, 4].cpu().detach().numpy(), name=f"Epoch{epoch}_FM4")
            experiment.log_image(t[0, 5].cpu().detach().numpy(), name=f"Epoch{epoch}_FM5")

            # Compute the error
            loss = criterion(output, labels)
            temporal_loss_train.append(loss.item())
            
            # Backpropagation (computing the gradients for each weight)
            loss.backward()
            
            # Update the weights
            optimizer.step()

        # Now, after each epoch, we have to see how the model is performing on the validation set #
        # Before evaluation we have to explicitly call .eval() method
        model.eval()
        temporal_loss_valid = []
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            output = model(images)

            # Compute the error
            loss = criterion(output, labels)
            temporal_loss_valid.append(loss.item())
        
        # Compute metrics after each epoch (mean value of loss) #
        medium_epoch_loss_train = sum(temporal_loss_train)/len(temporal_loss_train)
        medium_epoch_loss_valid = sum(temporal_loss_valid)/len(temporal_loss_valid)

        errors_train.append(medium_epoch_loss_train)
        errors_validation.append(medium_epoch_loss_valid)

        print(f"Epoch {epoch}. Training loss: {medium_epoch_loss_train}. Validation loss: {medium_epoch_loss_valid}")

        # Log metrics
        experiment.log_metric("train_loss", medium_epoch_loss_train, step=epoch)
        experiment.log_metric("val_loss", medium_epoch_loss_valid, step=epoch)

        # Saving the model of the current epoch
        torch.save(model.state_dict(), os.path.join(exp_path, f"Epoch{epoch}_Error{round(medium_epoch_loss_valid, 3)}"))


    # Run ... test.py


    plt.title("Learning curves")
    plt.plot(errors_train, label='Training loss')
    plt.plot(errors_validation, label='Validation loss')
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(exp_path, "losses.png"))