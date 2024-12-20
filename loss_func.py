import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class PinnLoss(_Loss):
    def __init__(self, e, nu):
        super(PinnLoss, self).__init__()
        self.e = e
        self.nu = nu
        self.G = e / (2 * (1 + nu))

        self.loss = nn.MSELoss(reduction='mean')

    def __call__(self, x, y, bc, pred, gt):
        # Predictions
        u = pred[:, :, 0:1]
        v = pred[:, :, 1:2]
        sigma_x = pred[:, :, 2:3]
        sigma_y = pred[:, :, 3:4]
        tau_xy = pred[:, :, 4:5]

        # Ground truth
        u_gt = gt[:, :, 0:1]
        v_gt = gt[:, :, 1:2]
        sigma_x_gt = gt[:, :, 2:3]
        sigma_y_gt = gt[:, :, 3:4]
        tau_xy_gt = gt[:, :, 4:5]

        # Calculate the loss
        data_loss = self.data_loss(u, v, sigma_x, sigma_y, tau_xy, u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt)
        # data_loss = torch.tensor(0.0)
        # pde_loss = self.pde_loss(x, y, sigma_x, sigma_y, tau_xy)
        pde_loss = torch.tensor(0.0)
        return data_loss, pde_loss

    def data_loss(self, u, v, sigma_x, sigma_y, tau_xy, u_gt, v_gt, sigma_x_gt, sigma_y_gt, tau_xy_gt):
        # Displacement loss
        loss_u = self.loss(u, u_gt)
        loss_v = self.loss(v, v_gt)

        # Stress loss
        loss_sigma_x = self.loss(sigma_x, sigma_x_gt)
        loss_sigma_y = self.loss(sigma_y, sigma_y_gt)
        loss_tau_xy = self.loss(tau_xy, tau_xy_gt)
        return loss_u + loss_v + loss_sigma_x + loss_sigma_y + loss_tau_xy

    def pde_loss(self, x, y, sigma_x, sigma_y, tau_xy):
        # Calculate the loss for the equilibrium equation and strain compatibility equation
        eq1, eq2 = self.cauchy_equilibrium(x, y, sigma_x, sigma_y, tau_xy)
        loss_eq = self.loss(eq1, torch.zeros_like(eq1)) + self.loss(eq2, torch.zeros_like(eq2))
        return loss_eq * self.e

    def bc_loss(self, u, v, sigma_x, sigma_y, tau_xy, bc):
        bc1, bc2 = self.disp_boundary(u, v, bc)
        bc3, bc4 = self.force_boundary(sigma_x, sigma_y, tau_xy, bc)

        loss_bc = (self.loss(bc1, torch.zeros_like(bc1)) + self.loss(bc2, torch.zeros_like(bc2)) +
                   self.loss(bc3, torch.zeros_like(bc3)) + self.loss(bc4, torch.zeros_like(bc4)))
        return loss_bc

    def cauchy_equilibrium(self, x, y, sigma_x, sigma_y, tau_xy):
        grad_outputs = torch.ones_like(x)
        dsigma_x_dx = torch.autograd.grad(sigma_x, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dsigma_y_dy = torch.autograd.grad(sigma_y, y, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dx = torch.autograd.grad(tau_xy, x, grad_outputs=grad_outputs, create_graph=True)[0]
        dtau_xy_dy = torch.autograd.grad(tau_xy, y, grad_outputs=grad_outputs, create_graph=True)[0]

        # dσ_x/dx + dτ_xy/dy = 0
        # dτ_xy/dx + dσ_y/dy = 0
        eq1 = dsigma_x_dx + dtau_xy_dy
        eq2 = dtau_xy_dx + dsigma_y_dy
        return eq1, eq2

    def geometric(self, x, y, u, v):
        # Calculate the derivatives of the displacement field
        du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        du_dy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]
        dv_dx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(x), create_graph=True)[0]
        dv_dy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(y), create_graph=True)[0]

        # ε_x = du/dx
        # ε_y = dv/dy
        # γ_xy = du/dy + dv/dx
        epsilon_x = du_dx
        epsilon_y = dv_dy
        gamma_xy = du_dy + dv_dx
        return epsilon_x, epsilon_y, gamma_xy

    def physical(self, sigma_x, sigma_y, tau_xy):
        # ε_x = (σ_x - ν * σ_y) / E
        # ε_y = (σ_y - ν * σ_x) / E
        # γ_xy = τ_xy / G
        epsilon_x = (sigma_x - self.nu * sigma_y) / self.e
        epsilon_y = (sigma_y - self.nu * sigma_x) / self.e
        gamma_xy = tau_xy / self.G
        return epsilon_x, epsilon_y, gamma_xy

    def force_boundary(self, sigma_x, sigma_y, tau_xy, bc):
        l = bc[:, :, 1:2]
        m = bc[:, :, 2:3]
        fx = bc[:, :, 3:4]
        fy = bc[:, :, 4:5]

        # Filter out the force boundary points
        mask = bc[:, :, 0:1] == 1
        # Calculate loss only for the boundary points
        sigma_x_bc = sigma_x[mask]
        sigma_y_bc = sigma_y[mask]
        tau_xy_bc = tau_xy[mask]
        l_bc = l[mask]
        m_bc = m[mask]
        fx_bc = fx[mask]
        fy_bc = fy[mask]

        # (l * σ_x) + (m * τ_xy) = fx
        # (l * τ_xy) + (m * σ_y) = fy
        bc1 = (l_bc * sigma_x_bc) + (m_bc * tau_xy_bc) - fx_bc
        bc2 = (l_bc * tau_xy_bc) + (m_bc * sigma_y_bc) - fy_bc
        return bc1, bc2

    def disp_boundary(self, u, v, bc):
        us = bc[:, :, 5:6]
        vs = bc[:, :, 6:7]

        # Filter out the displacement boundary points
        mask = bc[:, :, 0:1] == 2
        # Calculate loss only for the boundary points
        u_bc = u[mask]
        v_bc = v[mask]
        us_bc = us[mask]
        vs_bc = vs[mask]

        # u = us
        # v = vs
        # Since the displacement boundary values are very small,
        # we need to multiply them by the Young's modulus to increase their influence
        bc1 = (u_bc - us_bc) * self.e
        bc2 = (v_bc - vs_bc) * self.e
        return bc1, bc2
