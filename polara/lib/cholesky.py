class CholeskyFactor:
    def __init__(self, factor):
        self._factor = factor
        self._L = None
        self._transposed = False

    @property
    def L(self):
        if self._L is None:
            self._L = self._factor.L()
        return self._L

    @property
    def T(self):
        self._transposed = True
        return self

    def dot(self, v):
        if self._transposed:
            self._transposed = False
            return self.L.T.dot(self._factor.apply_P(v))
        else:
            return self._factor.apply_Pt(self.L.dot(v))

    def solve(self, y):
        x = self._factor
        if self._transposed:
            self._transposed = False
            return x.apply_Pt(x.solve_Lt(y, use_LDLt_decomposition=False))
        else:
            raise NotImplementedError

    def update_inplace(self, A, beta):
        self._factor.cholesky_inplace(A, beta=beta)
        self._L = None
