from abc import abstractmethod

import ansys.mapdl.reader as pymapdl
import numpy as np
from scipy.spatial import cKDTree


class ElasticBody:
    def __init__(self, e, nu, x_min, x_max, y_min, y_max, fea_path):
        self.e = e
        self.nu = nu
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        self.num_data, self.nodes, self.disp, self.stress = self.load_fea_result(fea_path)
        self.kdtree = self.build_kdtree(self.nodes)

    def load_fea_result(self, fea_path):
        # Load the finite element analysis results
        rst = pymapdl.read_binary(fea_path)

        # Get the nodes and ground truth
        nodes = np.array(rst.grid.points[:, :2])
        num_data = rst.time_values.shape[0]

        nodes_all = []
        disp_all = []
        stress_all = []

        for i in range(num_data):
            # Get the displacement and stress
            disp = rst.nodal_displacement(i)[1]
            stress = rst.nodal_stress(i)[1]
            stress = stress[:, [0, 1, 3]]

            # Remove the nan values
            not_nan = ~np.isnan(stress[:, 0]) & ~np.isnan(stress[:, 1]) & ~np.isnan(stress[:, 2])
            nodes_all.append(nodes[not_nan])
            disp_all.append(disp[not_nan])
            stress_all.append(stress[not_nan])

        # Concatenate the displacement and stress
        return num_data, nodes_all, disp_all, stress_all

    def build_kdtree(self, nodes):
        kdtrees = []
        for node in nodes:
            kdtrees.append(cKDTree(node))
        return kdtrees

    def geometry(self, nx, ny):
        x, y = np.meshgrid(np.linspace(self.x_min, self.x_max, nx), np.linspace(self.y_min, self.y_max, ny))
        x = x.flatten()
        y = y.flatten()
        mask = self.geo_filter(x, y)
        return x[mask], y[mask]

    def boundary_conditions(self, x, y, index):
        bc = np.zeros([x.shape[0], 7])

        # generate boundary conditions
        # Boundary condition encoding format: [mask, l, m, tx, ty, u, v]
        # mask: 0-internal point, 1-force boundary, 2-displacement boundary
        # l, m: For force boundary, l, m are the components of the normal vector
        # tx, ty: For force boundary, tx, ty are the components of the force
        # u, v: For displacement boundary, u, v are the components of the displacement
        for i in range(x.shape[0]):
            x_idx = x[i]
            y_idx = y[i]
            bc[i] = self.bc_filter(x_idx, y_idx, index)
        return bc

    def get_bc_dims(self):
        # [force_dim, displacement_dim]
        return [4, 2]

    def get_num_data(self):
        return self.num_data

    def get_ground_truth(self, x, y, index):
        if index >= self.num_data:
            raise ValueError("The index is out of range!")

        nodes = np.array([x, y]).T
        distances, indices = self.kdtree[index].query(nodes)
        disp = self.disp[index][indices] * 10e9
        stress = self.stress[index][indices]
        return np.concatenate([disp, stress], axis=-1)

    @abstractmethod
    def geo_filter(self, x, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def bc_filter(self, x, y, index):
        raise NotImplementedError("This method should be overridden by subclasses.")
