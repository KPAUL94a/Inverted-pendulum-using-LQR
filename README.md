## Inverted Pendulum LQR Controller

This repository features a Python simulation of a cart-pole inverted pendulum system stabilized by a Linear Quadratic Regulator (LQR). The core physics model, control logic, and animation pipeline are entirely implemented within `LQR.py`.

### 1. System Dynamics

The non-linear dynamics track a state vector defined as $x = [x, \dot{x}, \theta, \dot{\theta}]^T$, representing the cart's position and velocity alongside the pole's angle and angular velocity. The physical properties are configured as follows:

| Parameter | Value | Unit |
| --- | --- | --- |
| **Cart Mass (M)** | 1.0 | kg

 |
| **Pole Mass (m)** | 0.1 | kg

 |
| **Pole Length (l)** | 0.5 | m

 |
| **Gravity (g)** | 9.81 | m/s²

 |

To enable linear control, `LQR.py` employs a numerical method to linearize the system around its upright equilibrium state, yielding the continuous-time state-space matrices $A$ and $B$.

---

### 2. LQR Implementation

The LQR strategy calculates an optimal feedback matrix $K$ by minimizing the quadratic cost function:


$$J = \int_{0}^{\infty} (x^T Q x + u^T R u) dt$$

* **State Cost Matrix (Q):** Set to `[1.0, 1.0, 10.0, 10.0]` on the diagonal to aggressively penalize angular deviations.


* **Control Cost (R):** Set to `[[0.05]]` to balance actuation effort.


* **Feedback Gain (K):** Computed using `scipy.linalg.solve_continuous_are` to solve the continuous Algebraic Riccati Equation.


* **Actuation:** The controller calculates a restoring force $F = -K(x - x_{target})$, clipped between -10 and 10 to simulate real-world motor limits.



---

### 3. Simulation & Visualization

The simulation initializes the pendulum at a highly unstable starting angle of $\pi/3$ radians. The system's closed-loop trajectory is numerically integrated over an 8-second timeframe using `scipy.integrate.solve_ivp` with a maximum step size of 0.02s. Finally, `matplotlib.animation` renders the dynamic cart-pole response, exporting the visual output directly to a file named `inverted_pendulum.gif`.    
