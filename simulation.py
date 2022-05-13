import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sb
import tkinter as tk
import threading


class DuffingOscillator(threading.Thread):
    def __init__(self, alpha, beta, gamma, delta, omega, x0, v0):
        super().__init__()
        self.stop_simulation = False

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.x0 = x0
        self.v0 = v0

    def run(self):
        # Perform calculations
        x_grid = np.linspace(-1.5, 1.5, 100)
        v_grid = v(x_grid, self.alpha, self.beta)

        t_max, t_trans = 18000, 300
        dt_per_period = 100

        # Solving the equation
        t, xs, dt, pstep = solve_duffing_equation(t_max, dt_per_period, t_trans, self.x0, self.v0, self.alpha,
                                                  self.beta, self.gamma, self.delta, self.omega)
        x, x_dot = xs.T

    @staticmethod
    def v(x, alpha, beta):
        """Potential"""
        return beta * 0.25 * x ** 4 - alpha * 0.5 * x ** 2

    @staticmethod
    def derivatives(xs, t, alpha, beta, gamma, delta, omega):
        """Returns the derivatives dx/dt and d2x/dt2"""
        x, x_dot = xs
        x_dot_dot = -beta * x ** 3 - alpha * x - delta * x_dot + gamma * np.cos(omega * t)
        return x_dot, x_dot_dot

    @staticmethod
    def solve_duffing_equation(t_max, dt_per_period, t_trans, x0, v0, alpha, beta, gamma, delta, omega):
        """
        Finds the numerical solution to the Duffing equation using a suitable time grid.
        Solves a system of ordinary differential equations using lsoda algorithm from the FORTRAN library odepack.

        :param t_max: maximum time in seconds to integrate to
        :param dt_per_period: the number of time samples to include per period of the driving motion
        :param t_trans: the initial time period of transient behaviour until the solution settles down
        :param x0: initial position
        :param v0: initial velocity
        :param alpha: controls the linear stiffness
        :param beta: controls the amount of non-linearity in the restoring force
        :param gamma: the amplitude of the periodic driving force
        :param delta: controls the amount of damping
        :param omega: the angular frequency of the periodic driving force
        :return: the time grid (after t_trans); position, velocity, xdot; dt; step - the number of array points per
        period of the driving motion
        """
        # Time point spacings and the time grid
        period = 2 * np.pi / omega
        dt = 2 * np.pi / omega / dt_per_period
        step = int(period / dt)
        t = np.arange(0, t_max, dt)

        # Initial conditions: x, xdot
        xs0 = [x0, v0]
        xs = odeint(DuffingOscillator.derivatives, xs0, t, args=(alpha, beta, gamma, delta, omega))
        idx = int(t_trans / dt)
        return t[idx:], xs[idx:], dt, step


def v(x, alpha, beta):
    """Potential"""
    return beta * 0.25 * x ** 4 - alpha * 0.5 * x ** 2


# def dv_dx(x, alpha, beta):
#     """First derivative of potential"""
#     return beta * x ** 3 - alpha * x


def derivatives(xs, t, alpha, beta, gamma, delta, omega):
    """Returns the derivatives dx/dt and d2x/dt2"""
    x, x_dot = xs
    x_dot_dot = -beta * x ** 3 - alpha * x - delta * x_dot + gamma * np.cos(omega * t)
    return x_dot, x_dot_dot

    """
    Find the numerical solution to the Duffing equation using a suitable
    time grid: tmax is the maximum time (s) to integrate to; t_trans is
    the initial time period of transient behaviour until the solution
    settles down (if it does) to some kind of periodic motion (these data
    points are dropped) and dt_per_period is the number of time samples
    (of duration dt) to include per period of the driving motion (frequency
    omega).

    Returns the time grid, t (after t_trans), position, x, and velocity,
    xdot, dt, and step, the number of array points per period of the driving
    motion.

    """


def solve_duffing_equation(t_max, dt_per_period, t_trans, x0, v0, alpha, beta, gamma, delta, omega):
    """
    Finds the numerical solution to the Duffing equation using a suitable time grid.
    :param t_max: maximum time in seconds to integrate to
    :param dt_per_period:
    :param t_trans: initial
    :param x0: initial position
    :param v0: initial velocity
    :param alpha:
    :param beta:
    :param gamma: is the amplitude of the periodic driving force
    :param delta: controls the amount of damping
    :param omega: is the angular frequency of the periodic driving force
    :return:
    """
    # Time point spacings and the time grid
    period = 2 * np.pi / omega
    dt = 2 * np.pi / omega / dt_per_period
    step = int(period / dt)
    t = np.arange(0, t_max, dt)

    # Initial conditions: x, xdot
    xs0 = [x0, v0]
    xs = odeint(derivatives, xs0, t, args=(alpha, beta, gamma, delta, omega))
    idx = int(t_trans / dt)
    return t[idx:], xs[idx:], dt, step


class Gui:
    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry('300x420')
        self.root.resizable(False, False)
        self.root.title('Duffing Oscillator')

        self.alpha_label = None
        self.beta_label = None
        self.gamma_label = None
        self.delta_label = None
        self.omega_label = None
        self.initial_position_label = None
        self.initial_velocity_label = None

        self.alpha_entry = None
        self.beta_entry = None
        self.gamma_entry = None
        self.delta_entry = None
        self.omega_entry = None
        self.initial_position_entry = None
        self.initial_velocity_entry = None

        self.info_label = None
        self.start_button = None
        self.stop_button = None

        self.is_simulation_running = None

        self.create_widgets()

        self.root.mainloop()

    def create_widgets(self):
        self.alpha_label = tk.Label(self.root, text='alpha')
        self.beta_label = tk.Label(self.root, text='beta')
        self.gamma_label = tk.Label(self.root, text='gamma')
        self.delta_label = tk.Label(self.root, text='delta')
        self.omega_label = tk.Label(self.root, text='omega')
        self.initial_position_label = tk.Label(self.root, text='initial position')
        self.initial_velocity_label = tk.Label(self.root, text='initial velocity')

        def validate_input(p):
            if p == '':
                return True
            try:
                float(p)
                return True
            except ValueError:
                return False

        vcmd = self.root.register(validate_input)

        self.alpha_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.beta_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.gamma_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.delta_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.omega_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.initial_position_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.initial_velocity_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))

        self.info_label = tk.Label(self.root, text='')
        self.start_button = tk.Button(self.root, text='Start', command=self.on_start_btn_clicked)
        self.stop_button = tk.Button(self.root, text='Stop', command=self.on_stop_btn_clicked)
        self.stop_button['state'] = tk.DISABLED

        self.alpha_label.grid(row=0, column=0, pady=10)
        self.beta_label.grid(row=1, column=0, pady=10)
        self.gamma_label.grid(row=2, column=0, pady=10)
        self.delta_label.grid(row=3, column=0, pady=10)
        self.omega_label.grid(row=4, column=0, pady=10)
        self.initial_position_label.grid(row=5, column=0, pady=10)
        self.initial_velocity_label.grid(row=6, column=0, pady=10)

        self.alpha_entry.grid(row=0, column=1, pady=10)
        self.beta_entry.grid(row=1, column=1, pady=10)
        self.gamma_entry.grid(row=2, column=1, pady=10)
        self.delta_entry.grid(row=3, column=1, pady=10)
        self.omega_entry.grid(row=4, column=1, pady=10)
        self.initial_position_entry.grid(row=5, column=1, pady=10)
        self.initial_velocity_entry.grid(row=6, column=1, pady=10)

        self.info_label.grid(row=7, column=0, columnspan=2, sticky=tk.W + tk.E, pady=10, padx=10)
        self.start_button.grid(row=8, column=0, columnspan=2, sticky=tk.W + tk.E, pady=10, padx=10)
        self.stop_button.grid(row=9, column=0, columnspan=2, sticky=tk.W + tk.E, pady=10, padx=10)

    def on_start_btn_clicked(self):
        if not self.is_simulation_running:
            self.is_simulation_running = True
            self.start_button['state'] = tk.DISABLED
            self.stop_button['state'] = tk.NORMAL
            self.info_label['text'] = 'Performing calculations...'

    def on_stop_btn_clicked(self):
        if self.is_simulation_running:
            self.is_simulation_running = False
            self.start_button['state'] = tk.NORMAL
            self.stop_button['state'] = tk.DISABLED
            self.info_label['text'] = 'Simulation stopped'


def main():
    gui_thread = threading.Thread(target=gui)
    gui_thread.start()

    alpha, beta = -1, 1

    x_grid = np.linspace(-1.5, 1.5, 100)
    v_grid = v(x_grid, alpha, beta)

    # Set up the motion for an oscillator with initial position x0 and initially at rest
    x0, v0 = 0, 0
    t_max, t_trans = 18000, 300
    omega = 1.2
    gamma, delta = 0.5, 0.3

    dt_per_period = 100

    # Solving the equation
    t, xs, dt, pstep = solve_duffing_equation(t_max, dt_per_period, t_trans, x0, v0, alpha, beta, gamma, delta, omega)
    x, x_dot = xs.T

    # Creating plots
    fig, ax = plt.subplots(nrows=2, ncols=2)

    # Potential energy
    ax1 = ax[0, 0]
    ax1.plot(x_grid, v_grid)
    # ax1.set_ylim(-0.3, 0.15)
    ln1, = ax1.plot([], [], 'mo')
    ax1.set_xlabel(r'$x\ [\mathrm{m}]$')
    ax1.set_ylabel(r'$V(x)\ [\mathrm{J}]$')
    ax1.set_title('Potential energy')
    ax1.grid()

    # Position as a function of time
    ax2 = ax[1, 0]
    ax2.set_xlabel(r'$t\ [\mathrm{s}]$')
    ax2.set_ylabel(r'$x\ [\mathrm{m}]$')
    ln2, = ax2.plot(t[:100], x[:100])
    ax2.set_ylim(np.min(x), np.max(x))
    ax2.set_title('Position as a function of time')
    ax2.grid()

    # Phase space plot
    ax3 = ax[1, 1]
    ax3.set_xlabel(r'$x\ [\mathrm{m}]$')
    ax3.set_ylabel(r'$\dot{x}\ [\mathrm{m\,s^{-1}}]$')
    ln3, = ax3.plot([], [])
    ax3.set_xlim(np.min(x), np.max(x))
    ax3.set_ylim(np.min(x_dot), np.max(x_dot))
    ax3.set_title('Phase space')
    ax3.grid()

    # Poincaré section plot
    ax4 = ax[0, 1]
    ax4.set_xlabel(r'$x\ [\mathrm{m}]$')
    ax4.set_ylabel(r'$\dot{x}\ [\mathrm{m\,s^{-1}}]$')
    ax4.scatter(x[::pstep], x_dot[::pstep], s=2, lw=0, c=sb.color_palette()[0])
    scat1 = ax4.scatter([x0], [v0], lw=0, c='m')
    plt.tight_layout()
    ax4.set_title('Poincaré section')
    ax4.grid()

    def animate(i):
        """Update the image for iteration i of the Matplotlib animation."""

        ln1.set_data(x[i], v(x[i], alpha, beta))
        ln2.set_data(t[:i + 1], x[:i + 1])
        ax2.set_xlim(t_trans, t[i])
        ln3.set_data(x[:i + 1], x_dot[:i + 1])
        if not i % pstep:
            scat1.set_offsets(xs[i])
        return

    anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=100)

    plt.show()


if __name__ == '__main__':
    Gui()
