"""Tiny collection of 3D path generators.

Edit SELECT / PARAMS at the bottom and run this file to write trajectory.csv.
"""

import math
import numpy as np

# Analytic / geometric (return (N,3))

def line(n=500, length=1.0):
    t = np.linspace(0, length, n)
    return np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1).astype(float)

def circle(n=1000, R=1.0):
    t = np.linspace(0, 2 * math.pi, n)
    return np.stack([R * np.cos(t), R * np.sin(t), np.zeros_like(t)], axis=1).astype(float)

def helix(n=1500, R=1.0, pitch=0.25, turns=3.0):
    t = np.linspace(0, 2 * math.pi * turns, n)
    return np.stack([R * np.cos(t), R * np.sin(t), pitch * t], axis=1).astype(float)

def flattened_helix(n=1500, R=1.0, eps=0.02, pitch=0.2, turns=4.0):
    t = np.linspace(0, 2 * math.pi * turns, n)
    return np.stack([R * np.cos(t), eps * R * np.sin(t), pitch * t], axis=1).astype(float)

def lissajous(n=2000, a=3, b=2, delta=math.pi/2):
    t = np.linspace(0, 2 * math.pi, n)
    return np.stack([np.sin(a * t + delta), np.sin(b * t), np.zeros_like(t)], axis=1).astype(float)

def torus_knot(n=3000, p=2, q=3, R=2.0, r=0.6):
    t = np.linspace(0, 2 * math.pi, n)
    x = (R + r * np.cos(p * t)) * np.cos(q * t)
    y = (R + r * np.cos(p * t)) * np.sin(q * t)
    z = r * np.sin(p * t)
    return np.stack([x, y, z], axis=1).astype(float)

def drift_micro(n=2000, T=10.0, A=1e-6, omega=40.0):
    t = np.linspace(0, T, n)
    return np.stack([t, A * np.sin(omega * t), np.zeros_like(t)], axis=1).astype(float)

def exb_cycloid(n=2500, T=20.0, E=0.3, B=1.0):
    t = np.linspace(0, T, n)
    omega_c = B
    x = (E/B) * t - (E/(B**2)) * np.sin(omega_c * t)
    y = (E/(B**2)) * (1 - np.cos(omega_c * t))
    z = np.zeros_like(t)
    return np.stack([x, y, z], axis=1).astype(float)

# ODE-based via a small RK4

def integrate_ode(deriv, y0, t0, t1, dt):
    """Simple RK4 integrator returning (N,3). Assumes y has length 3."""
    y = np.asarray(y0, dtype=float).copy()
    t_values = np.arange(t0, t1 + dt * 0.5, dt)
    out = np.zeros((len(t_values), 3))
    for i, t in enumerate(t_values):
        out[i] = y
        k1 = deriv(t, y)
        k2 = deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = deriv(t + dt, y + dt * k3)
        y = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return out

def lorenz_deriv_factory(sigma=10.0, beta=8/3, rho=28.0):
    def f(t, y):
        x, yv, z = y
        return np.array([
            sigma * (yv - x),
            x * (rho - z) - yv,
            x * yv - beta * z,
        ])
    return f

def rossler_deriv_factory(a=0.2, b=0.2, c=5.7):
    def f(t, y):
        x, yv, z = y
        return np.array([
            -(yv + z),
            x + a * yv,
            b + z * (x - c),
        ])
    return f

def lorenz(n=8000, dt=0.005):
    deriv = lorenz_deriv_factory()
    T = n * dt
    return integrate_ode(deriv, np.array([1.0, 1.0, 1.0]), 0.0, T, dt)

def rossler(n=8000, dt=0.01):
    deriv = rossler_deriv_factory()
    T = n * dt
    return integrate_ode(deriv, np.array([0.0, 1.0, 0.0]), 0.0, T, dt)

# Selection (edit here)

SELECT = "lissajous"  # change to a key in GENERATORS
PARAMS = {}
OUTPUT_CSV = "trajectory.csv"

GENERATORS = {
    "line": line,
    "circle": circle,
    "helix": helix,
    "flattened_helix": flattened_helix,
    "lissajous": lissajous,
    "torus_knot": torus_knot,
    "drift_micro": drift_micro,
    "exb_cycloid": exb_cycloid,
    "lorenz": lorenz,
    "rossler": rossler,
}

def build():
    if SELECT not in GENERATORS:
        raise ValueError(f"Unknown generator '{SELECT}'. Available: {list(GENERATORS)}")
    return GENERATORS[SELECT](**PARAMS)

if __name__ == "__main__":
    pts = build()
    if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] != 3:
        raise RuntimeError("Generator did not return an (N,3) array")
    np.savetxt(OUTPUT_CSV, pts, delimiter=",")
    print(f"Saved {len(pts)} points to {OUTPUT_CSV} using '{SELECT}'")
