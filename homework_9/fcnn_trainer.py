from torch import nn
from utils import set_seed
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


class FCNNTrainer:
    def __init__(self, lr: int, model: nn.Module, device):
        set_seed(9)

        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.in_dim, self.out_dim = 28 * 28, 10
        self.model = model
        self.optimizer_name = None
        self.optimizer = None
        self.lr = lr
        self.epochs = None
        self.train_loader = None
        self.test_loader = None
        self.train_metrics = []
        self.test_metrics = []
        self.models_dir = "./models_states"


    def set_optimizer(self, optimizer_name: str, optimizer_func):
        self.optimizer = optimizer_func(self.model.parameters(), lr=self.lr)
        self.optimizer_name = optimizer_name

    def set_epochs(self, epochs: int):
        self.epochs = epochs

    def prepare_loaders(self, train_dataset, test_dataset) -> dict:
        train_loader = DataLoader(dataset = train_dataset, batch_size = 1000, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = 1000, shuffle = True)

        self.train_loader = train_loader
        self.test_loader = test_loader

        return {
            "train": train_loader,
            "test": test_loader
        }

    def __optimize_weights(self) -> float:
        for i, (x_tr, y_tr) in enumerate(self.train_loader):
            # Pass data to GPU's cores
            x_tr, y_tr = x_tr.to(self.device), y_tr.to(self.device)
            # Compute predictions
            outputs = self.model(x_tr)
            # Compute Loss for predictions
            loss = self.criterion(outputs, y_tr)
            # Remove previous gradients
            self.optimizer.zero_grad()
            # Compute current gradients
            loss.backward()
            # Update parameters
            self.optimizer.step()
            # Save training Loss
            return loss.item()

    def __count_test_loss(self) -> float:
        with torch.no_grad():
            for x_ts, y_ts in self.test_loader:
                x_ts, y_ts = x_ts.to(self.device), y_ts.to(self.device)
                # Compute predictions
                outputs = self.model(x_ts)
                # Compute Loss for predictions
                loss = self.criterion(outputs, y_ts)
                # Save validation Loss
                return loss.item()

    def epoch_train(self, epoch_val: int, prev_epochs: int):

        for epoch in range(epoch_val):
            train_loss = []
            self.model.train()

            train_loss.append(self.__optimize_weights())

            # Count train loss
            train_mean_loss = sum(train_loss) / len(train_loss)
            self.train_metrics.append(train_mean_loss)

            # Count test loss
            self.model.eval()
            test_loss = []
            test_loss.append(self.__count_test_loss())

            # Get mean of  Validation Loss per epoch
            test_mean_loss = sum(test_loss) / len(test_loss)
            self.test_metrics.append(test_mean_loss)

            self.__save_parameters(
                epoch=epoch,
                prev_epochs=prev_epochs,
                model_name=f"{self.model.__class__.__name__}_{self.optimizer_name}"
            )

            print(f'Train Epoch: {epoch} \t [Train Loss]: {train_mean_loss:.6f} \t [Test Loss]: {test_mean_loss:.6f}')

    # def full_train(self, epochs_list: list, optimizers_list: list):
    #     for optimizer_name, optimizer_func in optimizers_list.items():
    #         model_name = f'LogisticRegression_{optimizer_name}'
    #         self.__set_optimizer(optimizer_func)

    #         # Start training
    #         for ind, epoch in enumerate(epochs_list):
    #             self.__epoch_train(epoch_val=epoch)


    def plot_metrics(self, epochs: int):
        plt.figure(figsize=(10, 5))
        plt.plot(range(epochs), self.train_metrics, label='Training Loss')
        plt.plot(range(epochs), self.test_metrics, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Cross')
        plt.title(f"Training and Validation CEL over Epochs via {self.optimizer_name}")
        plt.legend()
        plt.grid(True)
        plt.show()

    def __save_parameters(self, epoch: int, prev_epochs: int, model_name: str):
        save_path = f'{self.models_dir}/{model_name}/epoch_{epoch + prev_epochs + 1}.pth'

        if not os.path.exists(f'{self.models_dir}/{model_name}'):
            os.makedirs(f'{self.models_dir}/{model_name}')

        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_metrics[epoch + prev_epochs],
            'test_loss': self.test_metrics[epoch + prev_epochs],
        }, save_path)

        if not os.path.exists(save_path):
            raise IOError(f"Failed to save the model at {save_path}")
        
    def load_model_state(self, model: nn.Module, optimizer, optimizer_name: str, epoch: int):
        checkpoint_path = f'{self.models_dir}/LogisticRegression_{optimizer_name}/epoch_{epoch}.pth'
    
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No such file or directory: '{checkpoint_path}'")

        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['test_loss']
    
    def clear_main_values(self, model, device):
        self.model = model
        self.optimizer_name = None
        self.optimizer = None
        self.epochs = None
        self.train_metrics = []
        self.test_metrics = []
