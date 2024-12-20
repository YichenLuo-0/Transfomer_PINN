import numpy as np
from overrides import overrides

from cases.elastic_body import ElasticBody


class Triangle(ElasticBody):
    def __init__(self, e, nu, l, fea_path=None):
        super().__init__(e, nu, 0, l, 0, l, fea_path)
        self.l = l
        self.q0 = [(i / 10) + 10.1 for i in range(self.num_data)]

    @overrides
    def geo_filter(self, x, y):
        return y <= x

    @overrides
    def bc_filter(self, x, y, index):
        # Face AB，受到向下的分布载荷
        if y == 0:
            qx = -x / self.l * self.q0[index]
            bc = np.array([1, 0, -1, 0, qx, 0, 0])

        # Face AC，不受力
        elif y == x:
            l_ = -1 / np.sqrt(2)
            m_ = 1 / np.sqrt(2)
            bc = np.array([1, l_, m_, 0, 0, 0, 0])

        # Face BC，固定在墙上
        elif x == self.l:
            bc = np.array([2, 0, 0, 0, 0, 0, 0])

        # 其他内部点无边界条件
        else:
            bc = np.array([0, 0, 0, 0, 0, 0, 0])
        return bc
