

class BasisFunctions():
    """Abstract class for basis functions that can be used to "lift" the state values.
    
    Override construct_basis
    """

    def __init__(self, n, Nlift):
        self.n = n
        self.Nlift = Nlift
        self.Lambda = None
        self.basis = None

    def lift(self, q, t):
        return self.basis(q, t)

    def construct_basis(self, ub=None, lb=None):
        pass