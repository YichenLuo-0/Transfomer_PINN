import sys

import numpy as np
import torch
from torch import nn
from torch.optim import LBFGS

from cases.triangle.triangle import Triangle
from loss_func import PinnLoss
from model.neural_network import PinnsFormer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 弹性体的尺寸
l = 2.0

# 杨氏模量和泊松比：E=201GPa, nu=0.3
e = 20.1
nu = 0.3


def random_choice(dataset_num, batch_size):
    indices = np.zeros(batch_size)
    batch_gird = dataset_num / batch_size
    for i in range(batch_size):
        begin = batch_gird * i
        end = batch_gird * (i + 1)
        indices[i] = np.random.randint(begin, end)
    return indices


def main():
    # Mesh size of the elastic body
    nx = 50
    ny = 50

    # Training parameters
    epochs = 600
    batch_size = 10

    # Initialize the elastic body
    elastic_body = Triangle(e, nu, l, fea_path="cases/triangle/file.rst")
    x, y = elastic_body.geometry(nx, ny)
    bc_dims = elastic_body.get_bc_dims()
    dataset_num = elastic_body.get_num_data()

    print("Sequence length: ", x.shape[0])

    # Initialize the boundary conditions and ground truth lists
    bc = []
    gt = []

    # Generate boundary conditions and ground truth
    for i in range(dataset_num):
        bc_ = elastic_body.boundary_conditions(x, y, i)
        gt_ = elastic_body.get_ground_truth(x, y, i)
        bc.append(bc_)
        gt.append(gt_)

    # Convert the BC to tensors
    bc = torch.tensor(bc, dtype=torch.float32)
    gt = torch.tensor(gt, dtype=torch.float32)

    # Define the coordinates tensor
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    x_copy = x.repeat(dataset_num, 1, 1)
    y_copy = y.repeat(dataset_num, 1, 1)

    # Initialize the network and optimizer
    pinn = PinnsFormer(
        bc_dims=bc_dims,
        d_output=5,
        d_emb=32,
        num_expand=16,
        patch_seq_len=256,
        depth=16,
        num_layers=4,
        num_heads=2
    ).to(device)

    # Optimizer and loss function
    optimizer = LBFGS(pinn.parameters(), lr=0.5, line_search_fn='strong_wolfe')
    loss_func = PinnLoss(e, nu)

    # Training loop
    for epoch in range(epochs):
        # get the random indices of the batch
        indices = random_choice(dataset_num, batch_size)

        # get the batch data
        x_batch = x_copy[indices].to(device)
        y_batch = y_copy[indices].to(device)
        bc_batch = bc[indices].to(device)
        gt_batch = gt[indices].to(device)

        # Reset the gradients of coordinates
        x_batch.requires_grad_(True)
        y_batch.requires_grad_(True)

        def closure():
            # Forward pass
            pred = pinn(x_batch, y_batch, bc_batch)

            # Calculate the loss
            data_loss, pde_loss = loss_func(x_batch, y_batch, bc_batch, pred, gt_batch)
            loss = data_loss + pde_loss

            # Print the loss
            print("Epoch: {}, Loss: {:.4f}, Data Loss: {:.4f}, PDE Loss: {:.4f}".format(
                epoch,
                loss.cpu().detach().numpy(),
                data_loss.cpu().detach().numpy(),
                pde_loss.cpu().detach().numpy()
            ))
            # sys.stdout.flush()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

    # Save the model
    torch.save(pinn, "cases/triangle/pinn.pth")


if __name__ == "__main__":
    main()
