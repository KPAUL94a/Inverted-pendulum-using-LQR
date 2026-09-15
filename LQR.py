import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are

# PHYSICAL VALUES

M = 1.0
m = 0.1
l = 0.5
g = 9.81

state_eq = np.array([0.0, 0.0, 0.0, 0.0])


# GETTING x_dot, x_ddot, theta_dot, theta_ddot
def dynamics(state, F):
    x, x_dot, theta, theta_dot = state

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    total_mass = M + m
    temp = (F + m * l * theta_dot**2 * sin_t) / total_mass
    theta_ddot = (g * sin_t - cos_t * temp) / (
            l * (4.0 / 3.0 - m * cos_t**2 / total_mass)
        )
    x_ddot = temp - (m * l * theta_ddot * cos_t) / total_mass

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

state = np.array([0.0, 0.0, 0.0, 0.0])

#    Numerically linearize about the upright equilibrium
#    (state = 0, F = 0) to get dx/dt ~= A x + B F
#    Trying to get how system is being affected from changing of state vector and force (A, B)

def linearize(state_eq, F_eqm = 0, eps = 1e-5):
    n = len(state_eq)
    A = np.zeros((n, n))

    for i in range(n):
        s_plus, s_minus = state_eq.copy(), state_eq.copy()
        s_plus[i] += eps
        s_minus[i] -= eps

        A[:, i] = (dynamics(s_plus, F_eqm) - dynamics(s_minus, F_eqm))/(2*eps)

    B = (dynamics(state_eq, F_eqm + eps) - dynamics(state_eq, F_eqm - eps))/(2*eps)

    return A, B.reshape(-1, 1)


A, B = linearize(state_eq)  # From it we get A, B

## Now we will try to get LQR and control feedback gain K

Q = np.diag([1.0, 1.0, 10.0, 10.0])  # State cost matrix weight of 1, 1, 10, 10
R = np.array([[0.05]])    # Control cost matrix

# Solving Riccati equation to get the P(optimal cost) and K(Deriving feedback Gain)

P = solve_continuous_are(A, B, Q, R)
K = (np.linalg.inv(R) @ B.T @ P).flatten()

target = np.array([0, 0, 0, 0])  #It should be our target to make the rod straight

# Thsi will provide the force to keet the rod straight
def controller(state):
    F = -K @ (state - target)
    return np.clip(F, -10, 10)

#This will calculate dynamics of system at different time_stamps
def closed_loop(t, state):
    return dynamics(state, controller(state))

#### SIMULATING

state0 = np.array([0.0, 0.0, np.pi/3, 0.0])
t_span = (0, 8)
t_eval = np.linspace(*t_span, 400)

sol = solve_ivp(closed_loop, t_span, state0, t_eval=t_eval, max_step=0.02)
print(sol)
x_t, theta_t = sol.y[0], sol.y[2]




fig, ax = plt.subplots(2, figsize=(12, 12))
ax[0].set_xlim(0, 20)
ax[0].set_ylim(-0.3, 1.2)
ax[0].set_aspect("equal")
ax[0].grid(alpha=0.3)
ax[0].set_title("Inverted Pendulum (Cart-Pole) -- LQR Balancing")
ax[0].axhline(0, color="black", lw=1)

cart_w, cart_h = 0.3, 0.15
cart_patch = plt.Rectangle((0, 0), cart_w, cart_h, fc="#3B82F6", ec="black", zorder=3)
ax[0].add_patch(cart_patch)
pole_line, = ax[0].plot([], [], lw=4, c="#EF4444", zorder=2)
bob = plt.Circle((0, 0), 0.045, fc="#111827", zorder=4)
ax[0].add_patch(bob)
time_text = ax[0].text(0.02, 0.92, "", transform=ax[0].transAxes)


def init():
    cart_patch.set_xy((-cart_w / 2, -cart_h / 2))
    pole_line.set_data([], [])
    return cart_patch, pole_line, bob, time_text


def update(i):
    xc, th = x_t[i], theta_t[i]
    cart_patch.set_xy((xc - cart_w / 2, -cart_h / 2))
    pivot = (xc, cart_h / 2)
    tip = (xc + 2 * l * np.sin(th), cart_h / 2 + 2 * l * np.cos(th))
    pole_line.set_data([pivot[0], tip[0]], [pivot[1], tip[1]])
    bob.center = tip
    time_text.set_text(f"t = {sol.t[i]:.2f}s")
    return cart_patch, pole_line, bob, time_text

time = np.linspace(0, 8, 400)
ax[1].plot(time, theta_t)


ani = animation.FuncAnimation(
    fig, update, frames=len(sol.t), init_func=init, interval=20, blit=True
)

ani.save("inverted_pendulum_test_9.gif", writer="pillow", fps=30)
print("Saved inverted_pendulum_test.gif")
plt.close(fig)