import matplotlib.pyplot as plt
import numpy as np
import torch

from cases.triangle.triangle import Triangle

# 弹性体的尺寸
l = 2.0

# 杨氏模量和泊松比：E=201GPa, nu=0.3
e = 20.1
nu = 0.3


def generate_fig(x, y, sigma_x, sigma_y, tau_xy, title):
    # 计算冯米塞斯等效应力
    sigma_v = []
    for i in range(len(x)):
        sigma_t = np.mat([[sigma_x[i], tau_xy[i]], [tau_xy[i], sigma_y[i]]])
        lam, _ = np.linalg.eig(sigma_t)
        sigma_v_ = np.sqrt((lam[0] ** 2) + (lam[1] ** 2) - (lam[0] * lam[1]))
        sigma_v.append(sigma_v_)
    sigma_v = np.array(sigma_v)

    # 可视化结果
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    # 绘制σ_x应力分布
    sc = ax[0][0].scatter(x, y, c=sigma_x, cmap='viridis')
    fig.colorbar(sc, ax=ax[0][0], label='σ_x')
    ax[0][0].set_title('σ_x Stress Distribution')

    # 绘制σ_y应力分布
    sc = ax[0][1].scatter(x, y, c=sigma_y, cmap='viridis')
    fig.colorbar(sc, ax=ax[0][1], label='σ_y')
    ax[0][1].set_title('σ_y Stress Distribution')

    # 绘制τ_xy应力分布
    sc = ax[1][0].scatter(x, y, c=tau_xy, cmap='viridis')
    fig.colorbar(sc, ax=ax[1][0], label='τ_xy')
    ax[1][0].set_title('τ_xy Stress Distribution')

    # 绘制σ_v应力分布
    sc = ax[1][1].scatter(x, y, c=sigma_v, cmap='viridis')
    fig.colorbar(sc, ax=ax[1][1], label='σ_v')
    ax[1][1].set_title('σ_v Stress Distribution')

    # 设置标题
    fig.suptitle(title)

    # Print the figure
    plt.show()


def main():
    # Mesh size of the elastic body
    nx = 40
    ny = 40

    index = 10

    # Initialize the elastic body and generate the boundary conditions and ground truth
    elastic_body = Triangle(e, nu, l, fea_path="cases/triangle/file.rst")
    x, y = elastic_body.geometry(nx, ny)
    bc = elastic_body.boundary_conditions(x, y, index)
    gt = elastic_body.get_ground_truth(x, y, index)

    u_gt = gt[:, 0:1].reshape(-1)
    v_gt = gt[:, 1:2].reshape(-1)
    sigma_x_gt = gt[:, 2:3].reshape(-1)
    sigma_y_gt = gt[:, 3:4].reshape(-1)
    tau_xy_gt = gt[:, 4:5].reshape(-1)

    # Convert the data to tensors
    x = torch.tensor(x, dtype=torch.float32).view(-1, 1).unsqueeze(0).requires_grad_(True)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1).unsqueeze(0).requires_grad_(True)
    coord = torch.cat([x, y], dim=-1)
    bc = torch.tensor(bc, dtype=torch.float32).unsqueeze(0)

    # Use the trained model to predict the stress distribution
    # pinn = PinnsFormer(d_model=64, d_hidden=64, n=4, heads=2, e=e, nu=nu)
    pinn = torch.load("cases/triangle/pinn.pth").cpu()
    pred = pinn(coord, bc, coord)

    u = pred[:, :, 0:1]
    v = pred[:, :, 1:2]
    sigma_x = pred[:, :, 2:3]
    sigma_y = pred[:, :, 3:4]
    tau_xy = pred[:, :, 4:5]

    # Visualize the results
    x = x.view(-1).detach().numpy()
    y = y.view(-1).detach().numpy()
    u = u.view(-1).detach().numpy()
    v = v.view(-1).detach().numpy()
    sigma_x = sigma_x.view(-1).detach().numpy()
    sigma_y = sigma_y.view(-1).detach().numpy()
    tau_xy = tau_xy.view(-1).detach().numpy()

    # Calculate the error
    err_sigma_x = np.abs(sigma_x - sigma_x_gt)
    err_sigma_y = np.abs(sigma_y - sigma_y_gt)
    err_tau_xy = np.abs(tau_xy - tau_xy_gt)

    # Plot the results
    generate_fig(x, y, sigma_x, sigma_y, tau_xy, "Predicted Stress Distribution")
    generate_fig(x, y, sigma_x_gt, sigma_y_gt, tau_xy_gt, "Ground Truth Stress Distribution")
    generate_fig(x, y, err_sigma_x, err_sigma_y, err_tau_xy, "Error Stress Distribution")


if __name__ == '__main__':
    main()
