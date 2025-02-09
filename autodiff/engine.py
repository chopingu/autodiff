import math


class Scalar:
    """
    Scalar class containing a single scalar value,
    implementing forward and backward propagation.

    value       - scalar value
    grad        - gradient of root tensor with respect to this tensor
    _backward   - local gradient propagation
    _ancestors  - set of ancestor tencors in computational DAG
    """

    def __init__(self, value, _ancestors=()):
        self.value = value
        self.grad = 0
        self._backward = lambda: None
        self._ancestors = set(_ancestors)

    def __repr__(self):
        return f"{self.value}"

    def __add__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        res = Scalar(self.value + other.value, (self, other))

        def _backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = _backward

        return res

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            other = Scalar(other)

        res = Scalar(self.value * other.value, (self, other))

        def _backward():
            self.grad += other.value * res.grad
            other.grad += self.value * res.grad

        res._backward = _backward

        return res

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        return self * other**-1

    def __rdiv__(self, other):
        return other * self**-1

    def __pow__(self, power):
        res = Scalar(self.value**power, (self,))

        def _backward():
            self.grad += (power * self.value**(power - 1)) * res.grad

        res._backward = _backward

        return res

    def tanh(self):
        x = self.value
        y = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        res = Scalar(y, (self,))

        def _backward():
            self.grad += (1 - y**2) * res.grad

        res._backward = _backward

        return res

    def relu(self):
        res = Scalar(0 if self.value < 0 else self.value, (self,))

        def _backward():
            self.grad += (res.value > 0) * res.grad

        res._backward = _backward

        return res

    def gelu(self):
        x = self.value
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)
        y = 0.5 * x * (1 + (math.exp(2 * inner) - 1) / (math.exp(2 * inner) + 1))
        res = Scalar(y, (self,))

        def _backward():
            sech_inner = 2.0 / (math.exp(inner) + math.exp(-inner))
            self.grad += (0.5 * (1 + math.tanh(inner)) + 0.5 * x * sech_inner * sech_inner *
                          math.sqrt(2.0 / math.pi) * (1 + 3 * 0.044715 * x * x)) * res.grad

        res._backward = _backward

        return res

    def backward(self):
        toposort = []
        visited = set()

        def dfs(u):
            if u in visited:
                return

            visited.add(u)
            for v in u._ancestors:
                dfs(v)

            toposort.append(u)

        dfs(self)

        self.grad = 1
        for node in reversed(toposort):
            node._backward()
