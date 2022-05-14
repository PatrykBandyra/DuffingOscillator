import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation
import seaborn as sb
import tkinter as tk
from tkinter import ttk
import json


class DuffingOscillator:
    def __init__(self, alpha, beta, gamma, delta, omega, x0, v0, t_max, t_trans, dt_per_period, gui):
        super().__init__()
        self.stop_simulation = False

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.omega = omega
        self.x0 = x0
        self.v0 = v0

        self.t_max = t_max
        self.t_trans = t_trans
        self.dt_per_period = dt_per_period

        self.gui = gui

        self.animation_speed = 1
        self.i = 0

    def handle_close(self, event):
        self.gui.on_stop_btn_clicked()

    def run(self):
        x_grid = np.linspace(-1.5, 1.5, 100)
        v_grid = DuffingOscillator.v(x_grid, self.alpha, self.beta)

        # Solving the equation
        t, xs, dt, pstep = DuffingOscillator.solve_duffing_equation(self.alpha, self.beta, self.gamma, self.delta,
                                                                    self.omega, self.x0, self.v0, self.t_max,
                                                                    self.t_trans, self.dt_per_period)
        x, x_dot = xs.T

        # Creating plots
        fig, ax = plt.subplots(nrows=2, ncols=2)
        fig.canvas.mpl_connect('close_event', self.handle_close)

        # Potential energy
        ax1 = ax[0, 0]
        ax1.plot(x_grid, v_grid)
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
        ax4.scatter(x[::pstep], x_dot[::pstep], s=2, lw=0, color=sb.color_palette()[0])
        scat1 = ax4.scatter([self.x0], [self.v0], lw=0, color='m')
        plt.tight_layout()
        ax4.set_title('Poincaré section')
        ax4.grid()

        def animate(i):
            """Update the image for iteration i of the Matplotlib animation"""
            if not self.stop_simulation:
                self.i += 1 * self.animation_speed
                self.i = self.i % len(x)
                ln1.set_data(x[self.i],
                             DuffingOscillator.v(x[self.i], self.alpha, self.beta))
                ln2.set_data(t[:self.i + 1], x[:self.i + 1])
                ax2.set_xlim(self.t_trans, t[self.i])
                ln3.set_data(x[:self.i + 1], x_dot[:self.i + 1])
                if not self.i % pstep:
                    scat1.set_offsets(xs[self.i])
                return
            else:
                plt.close(plt.gcf())

        anim = animation.FuncAnimation(fig, animate, frames=len(x), interval=100)

        plt.show()

    @staticmethod
    def v(x, alpha, beta):
        """Potential"""
        return beta * 0.25 * x ** 4 - alpha * 0.5 * x ** 2

    @staticmethod
    def derivatives(xs, t, alpha, beta, gamma, delta, omega):
        """Returns the derivatives dx/dt and d2x/dt2"""
        x, x_dot = xs
        x_dot_dot = -beta * x ** 3 + alpha * x - delta * x_dot + gamma * np.cos(omega * t)
        return x_dot, x_dot_dot

    @staticmethod
    def solve_duffing_equation(alpha, beta, gamma, delta, omega, x0, v0, t_max, t_trans, dt_per_period):
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


class Gui:
    def __init__(self):
        self.simulation = None
        self.presets = None
        self.root = tk.Tk()
        self.root.geometry('565x655')  # width x height
        self.root.resizable(False, False)
        self.root.title('Duffing Oscillator')

        self.alpha_label = None
        self.beta_label = None
        self.gamma_label = None
        self.delta_label = None
        self.omega_label = None
        self.initial_position_label = None
        self.initial_velocity_label = None
        self.t_max_label = None
        self.t_trans_label = None
        self.dt_per_period_label = None

        self.alpha_entry = None
        self.beta_entry = None
        self.gamma_entry = None
        self.delta_entry = None
        self.omega_entry = None
        self.initial_position_entry = None
        self.initial_velocity_entry = None
        self.t_max_entry = None
        self.t_trans_entry = None
        self.dt_per_period_entry = None

        self.tree_view_info_label = None
        self.tree_frame = None
        self.tree_scroll = None
        self.tree_view = None

        self.animation_speed_label = None
        self.animation_speed_slider = None

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
        self.t_max_label = tk.Label(self.root, text='maximum time in seconds to integrate to')
        self.t_trans_label = tk.Label(self.root, text='the initial time period of transient behaviour until '
                                                      'the solution settles down')
        self.dt_per_period_label = tk.Label(self.root, text='the number of time samples to include per period '
                                                            'of the driving motion')

        def validate_input(p):
            if p == '':
                return True
            if len(p) == 1 and p[0] == '-':
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
        self.t_max_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.t_trans_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))
        self.dt_per_period_entry = tk.Entry(self.root, validate='all', validatecommand=(vcmd, '%P'))

        self.tree_view_info_label = tk.Label(self.root, text='Select preset by double click', fg='#ff0000',
                                             font=('Arial', 10))

        # Tree frame
        self.tree_frame = tk.Frame(self.root)
        self.tree_frame.grid(row=10, column=0, columnspan=2, pady=5, padx=(5, 0), sticky=tk.W + tk.E)

        # Treeview scrollbar
        self.tree_scroll = tk.Scrollbar(self.tree_frame)
        self.tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview for presets
        self.tree_view = ttk.Treeview(self.tree_frame, height=5, yscrollcommand=self.tree_scroll.set)
        self.tree_view.pack(side=tk.LEFT)
        self.tree_scroll.config(command=self.tree_view.yview)
        self.tree_view['columns'] = ('alpha', 'beta', 'gamma', 'delta', 'omega', 'init pos', 'init velo', 't max',
                                     't trans', 'dt per period')
        self.tree_view.column('#0', width=30, minwidth=20)
        self.tree_view.column('alpha', anchor=tk.W, width=50)
        self.tree_view.column('beta', anchor=tk.W, width=50)
        self.tree_view.column('gamma', anchor=tk.W, width=50)
        self.tree_view.column('delta', anchor=tk.W, width=50)
        self.tree_view.column('omega', anchor=tk.W, width=50)
        self.tree_view.column('init pos', anchor=tk.W, width=50)
        self.tree_view.column('init velo', anchor=tk.W, width=50)
        self.tree_view.column('t max', anchor=tk.W, width=50)
        self.tree_view.column('t trans', anchor=tk.W, width=50)
        self.tree_view.column('dt per period', anchor=tk.W, width=50)

        self.tree_view.heading('#0', text='No.', anchor=tk.W)
        self.tree_view.heading('alpha', text='alpha', anchor=tk.W)
        self.tree_view.heading('beta', text='beta', anchor=tk.W)
        self.tree_view.heading('gamma', text='gamma', anchor=tk.W)
        self.tree_view.heading('delta', text='delta', anchor=tk.W)
        self.tree_view.heading('omega', text='omega', anchor=tk.W)
        self.tree_view.heading('init pos', text='initial position', anchor=tk.W)
        self.tree_view.heading('init velo', text='initial velocity', anchor=tk.W)
        self.tree_view.heading('t max', text='t max', anchor=tk.W)
        self.tree_view.heading('t trans', text='t trans', anchor=tk.W)
        self.tree_view.heading('dt per period', text='dt per period', anchor=tk.W)

        # Slider
        self.animation_speed_label = tk.Label(self.root, text='Animation speed', fg='#ff0000',
                                              font=('Arial', 10))
        self.animation_speed_slider = tk.Scale(self.root, from_=1, to=20, orient=tk.HORIZONTAL, command=self.on_slide)

        # Buttons
        self.info_label = tk.Label(self.root, text='', fg='#ff0000')
        self.start_button = tk.Button(self.root, text='Start', command=self.on_start_btn_clicked)
        self.stop_button = tk.Button(self.root, text='Stop', command=self.on_stop_btn_clicked)
        self.stop_button['state'] = tk.DISABLED

        # Positioning widgets in a grid
        self.alpha_label.grid(row=0, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.beta_label.grid(row=1, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.gamma_label.grid(row=2, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.delta_label.grid(row=3, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.omega_label.grid(row=4, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.initial_position_label.grid(row=5, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.initial_velocity_label.grid(row=6, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.t_max_label.grid(row=7, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.t_trans_label.grid(row=8, column=0, pady=5, padx=(5, 0), sticky=tk.E)
        self.dt_per_period_label.grid(row=9, column=0, pady=5, padx=(5, 0), sticky=tk.E)

        self.alpha_entry.grid(row=0, column=1, pady=5, padx=(5, 0))
        self.beta_entry.grid(row=1, column=1, pady=5, padx=(5, 0))
        self.gamma_entry.grid(row=2, column=1, pady=5, padx=(5, 0))
        self.delta_entry.grid(row=3, column=1, pady=5, padx=(5, 0))
        self.omega_entry.grid(row=4, column=1, pady=5, padx=(5, 0))
        self.initial_position_entry.grid(row=5, column=1, pady=5, padx=(5, 0))
        self.initial_velocity_entry.grid(row=6, column=1, pady=5, padx=(5, 0))
        self.t_max_entry.grid(row=7, column=1, pady=5, padx=(5, 0))
        self.t_trans_entry.grid(row=8, column=1, pady=5, padx=(5, 0))
        self.dt_per_period_entry.grid(row=9, column=1, pady=5, padx=(5, 0))

        self.tree_view_info_label.grid(row=10, column=0, columnspan=2, pady=5, padx=(5, 0), sticky=tk.W + tk.E)
        self.tree_frame.grid(row=11, column=0, columnspan=2, pady=5, padx=(5, 0), sticky=tk.W + tk.E)

        self.animation_speed_label.grid(row=12, column=0, columnspan=2, pady=(5, 0), padx=(5, 0), sticky=tk.W + tk.E)
        self.animation_speed_slider.grid(row=13, column=0, columnspan=2, pady=(5, 0), padx=(5, 0), sticky=tk.W + tk.E)

        self.info_label.grid(row=14, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5, padx=(5, 0))
        self.start_button.grid(row=15, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5, padx=(5, 0))
        self.stop_button.grid(row=16, column=0, columnspan=2, sticky=tk.W + tk.E, pady=5, padx=(5, 0))

        # Loading presets
        presets = self.load_presets()
        if presets is not None:
            for index, preset in enumerate(presets):
                values = tuple(preset.values())
                self.tree_view.insert(parent='', iid=str(index), text=str(index), index='end', values=values)

        # Binding
        self.tree_view.bind('<Double-1>', self.on_preset_clicked)

    def on_slide(self, animation_speed):
        if self.simulation is not None:
            self.simulation.animation_speed = int(animation_speed)

    def on_preset_clicked(self, event):
        # Clear entries
        self.alpha_entry.delete(0, tk.END)
        self.beta_entry.delete(0, tk.END)
        self.gamma_entry.delete(0, tk.END)
        self.delta_entry.delete(0, tk.END)
        self.omega_entry.delete(0, tk.END)
        self.initial_position_entry.delete(0, tk.END)
        self.initial_velocity_entry.delete(0, tk.END)
        self.t_max_entry.delete(0, tk.END)
        self.t_trans_entry.delete(0, tk.END)
        self.dt_per_period_entry.delete(0, tk.END)

        # Grab data from tree view
        selected = self.tree_view.focus()
        values = self.tree_view.item(selected, 'values')

        # Insert preset data into entries
        self.alpha_entry.insert(0, values[0])
        self.beta_entry.insert(0, values[1])
        self.gamma_entry.insert(0, values[2])
        self.delta_entry.insert(0, values[3])
        self.omega_entry.insert(0, values[4])
        self.initial_position_entry.insert(0, values[5])
        self.initial_velocity_entry.insert(0, values[6])
        self.t_max_entry.insert(0, values[7])
        self.t_trans_entry.insert(0, values[8])
        self.dt_per_period_entry.insert(0, values[9])

    def load_presets(self):
        try:
            with open('presets.json', 'r') as file:
                data = json.load(file)
                # Validate data
                for i, preset in enumerate(data):
                    try:
                        float(preset['alpha'])
                        float(preset['beta'])
                        float(preset['gamma'])
                        float(preset['delta'])
                        float(preset['omega'])
                        float(preset['x0'])
                        float(preset['v0'])
                        float(preset['t_max'])
                        float(preset['t_trans'])
                        float(preset['dt_per_period'])
                    except ValueError:
                        self.info_label['text'] = f'Found invalid data in one of the presets'
                        del data[i]
                return data
        except FileNotFoundError:
            self.info_label['text'] = 'No presets found'
            return None

    def get_data_from_entries(self):
        try:
            alpha = float(self.alpha_entry.get())
            beta = float(self.beta_entry.get())
            gamma = float(self.gamma_entry.get())
            delta = float(self.delta_entry.get())
            omega = float(self.omega_entry.get())
            x0 = float(self.initial_position_entry.get())
            v0 = float(self.initial_velocity_entry.get())
            t_max = float(self.t_max_entry.get())
            t_trans_entry = float(self.t_trans_entry.get())
            dt_per_period = float(self.dt_per_period_entry.get())
            return alpha, beta, gamma, delta, omega, x0, v0, t_max, t_trans_entry, dt_per_period
        except ValueError:
            self.info_label['text'] = 'Please fill in all entries'
            return None

    def on_start_btn_clicked(self):
        if not self.is_simulation_running:
            data = self.get_data_from_entries()
            if data is not None:
                self.is_simulation_running = True
                self.start_button['state'] = tk.DISABLED
                self.stop_button['state'] = tk.NORMAL
                self.info_label['text'] = 'Performing calculations...'
                self.simulation = DuffingOscillator(*data, self)
                self.simulation.run()

    def on_stop_btn_clicked(self):
        if self.is_simulation_running:
            self.is_simulation_running = False
            self.start_button['state'] = tk.NORMAL
            self.stop_button['state'] = tk.DISABLED
            self.info_label['text'] = 'Stopping simulation...'
            self.simulation.stop_simulation = True


def main():
    Gui()


if __name__ == '__main__':
    main()
