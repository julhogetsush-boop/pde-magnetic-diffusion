"""Finite differences and reference solutions for the conducting half-slab."""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu, spsolve


class Grid:
    def __init__(self, intervals, boundary="DN"):
        if intervals < 2 or boundary not in ("DD", "DN"):
            raise ValueError("Use at least two intervals and boundary DD or DN")
        self.n = intervals
        self.boundary = boundary
        self.h = 1 / intervals
        size = intervals if boundary == "DN" else intervals - 1
        self.x = self.h * np.arange(1, size + 1)
        lower = np.ones(size - 1)
        upper = np.ones(size - 1)
        self.weights = np.ones(size)
        if boundary == "DN":
            lower[-1] = 2
            self.weights[-1] = 0.5
        self.A = diags((lower, -2 * np.ones(size), upper), (-1, 0, 1), format="csc") / self.h**2
        self.identity = eye(size, format="csc")
        self.q = np.zeros(size)
        self.q[0] = 1 / self.h**2

    def mode(self, index):
        wave = index * np.pi if self.boundary == "DD" else (index - 0.5) * np.pi
        eigenvalue = -4 / self.h**2 * np.sin(wave * self.h / 2)**2
        return wave, eigenvalue, np.sin(wave * self.x)

    def norm(self, values):
        return np.sqrt(self.h * np.dot(self.weights, abs(values)**2))

    def full(self, values, surface=0):
        if self.boundary == "DD":
            return np.r_[surface, values, 0]
        return np.r_[surface, values]


def stepper(grid, k, method, allow_growth=False):
    if not np.isfinite(k) or k <= 0:
        raise ValueError("Time step must be positive and finite")
    if method == "FTCS":
        _, most_negative, _ = grid.mode(len(grid.x))
        if -k * most_negative > 2 + 1e-13 and not allow_growth:
            raise ValueError("FTCS step exceeds the finite-grid stability limit")

        def advance(u, left, right):
            updated = u + k * (grid.A @ u)
            updated[0] += k * grid.q[0] * left
            return updated

        return advance
    if method == "CN":
        half = k * grid.A / 2
        lhs = splu(grid.identity - half)
        rhs = grid.identity + half

        def advance(u, left, right):
            vector = rhs @ u
            vector[0] += k * grid.q[0] * (left + right) / 2
            return lhs.solve(vector)

        return advance
    raise ValueError(f"Unknown method {method}")


def integrate(grid, initial, boundary, final_time, steps, method):
    k = final_time / steps
    advance = stepper(grid, k, method)
    u = initial.copy()
    left = boundary(0)
    for m in range(1, steps + 1):
        right = boundary(m * k)
        u = advance(u, left, right)
        left = right
    return u


def multiplier(method, z):
    return 1 + z if method == "FTCS" else (1 + z / 2) / (1 - z / 2)


def transfer(depth, omega):
    z = np.sqrt(1j * np.asarray(omega))
    return np.cosh((1 - depth) * z) / np.cosh(z)


def transient_reference(depth, t, omega, modes=96):
    """Continuous sine-series solution for zero initial field at positive time."""
    if t <= 0:
        raise ValueError("The transient reference expects positive time")
    wave = (np.arange(modes) + 0.5) * np.pi
    coefficient = 2 * omega * wave * np.exp(-wave**2 * t) / (wave**4 + omega**2)
    periodic = np.imag(transfer(depth, omega) * np.exp(1j * omega * t))
    return periodic + np.sin(np.outer(depth, wave)) @ coefficient


def harmonic_grid(grid, omega, method="SD", k=None):
    if method == "SD":
        symbol = 1j * omega
    elif method == "FTCS":
        symbol = np.expm1(1j * omega * k) / k
    elif method == "CN":
        symbol = 2j * np.tan(omega * k / 2) / k
    else:
        raise ValueError(f"Unknown harmonic method {method}")
    return spsolve(symbol * grid.identity - grid.A, grid.q)


def period_fit(grid, omega, points, method, tolerance=1e-8):
    """Fit the last cycle after the same-phase whole-grid difference settles."""
    k = 2 * np.pi / (points * omega)
    advance = stepper(grid, k, method)
    angle = np.arange(points + 1) * (2 * np.pi / points)
    forcing = np.sin(angle)
    forcing[-1] = 0.0
    u = np.zeros(len(grid.x))
    trace = np.empty(points)
    mid = grid.n // 2 - 1
    for cycle in range(1, 31):
        previous = u.copy()
        for j in range(points):
            trace[j] = u[mid]
            u = advance(u, forcing[j], forcing[j + 1])
        drift = grid.norm(u - previous)
        if cycle >= 3 and drift <= tolerance:
            coefficients = np.linalg.lstsq(
                np.column_stack((forcing[:-1], np.cos(angle[:-1]), np.ones(points))),
                trace, rcond=None,
            )[0]
            return coefficients[0] + 1j * coefficients[1], cycle, drift
    raise RuntimeError(f"Periodic response did not settle at Omega={omega}")
