"""
Author: Des De Borger

Script to simulate permanent magnet lens behaviour.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

MU_R=1.05
ETA=296548.4789
EPSILON=9.78475592*10**-7

def B_field_ring(z, R_1, R_2):
    return (z/1000)/2*(1/np.sqrt((z/1000)**2+(R_1/1000)**2)-1/np.sqrt((z/1000)**2+(R_2/1000)**2))

def B_field_z(z, R_1, R_2, d):
    return B_field_ring(z+d/2, R_1, R_2)-B_field_ring(z-d/2, R_1, R_2)

def B_field_ring_d1(z, R_1, R_2):
    r1_sq = R_1 ** 2
    r2_sq = R_2 ** 2

    term1 = 1 / np.sqrt(z ** 2 + r1_sq)
    term2 = -1 / np.sqrt(z ** 2 + r2_sq)

    inner1 = -(1 / (z ** 2 + r1_sq)) ** (3 / 2)
    inner2 = (1 / (z ** 2 + r2_sq)) ** (3 / 2)
    term3 = z**2 * (inner1 + inner2)

    return (1/2000)*(term1 + term2 + term3)

def B_field_z_d1(z, R_1, R_2, d):
    return B_field_ring_d1(z+d/2, R_1, R_2)-B_field_ring_d1(z-d/2, R_1, R_2)

def B_field_ring_d2(z, R_1, R_2):
    r1_sq = R_1 ** 2
    r2_sq = R_2 ** 2

    # First part: z * (A - B - C + D)
    A = (3 * z ** 2) / (z ** 2 + r1_sq) ** (5 / 2)
    B = 1 / (z ** 2 + r1_sq) ** (3 / 2)
    C = (3 * z ** 2) / (z ** 2 + r2_sq) ** (5 / 2)
    D = 1 / (z ** 2 + r2_sq) ** (3 / 2)

    # Second part: 2 * (-E + F)
    E = z / (z ** 2 + r1_sq) ** (3 / 2)
    F = z / (z ** 2 + r2_sq) ** (3 / 2)

    return z/2000 * (A - B - C + D) + 2 * (-E + F)

def B_field_z_d2(z, R_1, R_2, d):
    return B_field_ring_d2(z+d/2, R_1, R_2)-B_field_ring_d2(z-d/2, R_1, R_2)

def calculate_B_field(n, setup_length, lens_position, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet):
    z_eval = np.linspace(0, setup_length, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+MU_R*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    return B_r_yoke*B_field_z(z_eval-lens_position, R_1, R_2, d)

def calculate_B_field_d1(n, setup_length, lens_position, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet):
    z_eval = np.linspace(0, setup_length, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+MU_R*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    return B_r_yoke*B_field_z_d1(z_eval-lens_position, R_1, R_2, d)


def calculate_B_field_d2(n, setup_length, lens_position, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet):
    z_eval = np.linspace(0, setup_length, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+MU_R*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    return B_r_yoke*B_field_z_d2(z_eval-lens_position, R_1, R_2, d)


def plot_B_field(ax, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds=30, n=1000):
    z=np.linspace(-bounds, bounds, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+MU_R*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    B_z=B_r_yoke*B_field_z(z, R_1, R_2, d)

    ax.clear()
    ax.plot(z, B_z, linestyle='-', color='b', label='$B(z)$')
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='0')
    ax.set_xlabel('$z$ (mm)', fontsize=14)
    ax.set_ylabel('$B(z)$ (T)', fontsize=14)
    ax.set_title('Interactief $B(z)$ veld', fontsize=16)
    ax.grid()
    ax.legend()

    return

def plot_B_field(ax, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds=30, n=1000):
    z=np.linspace(-bounds, bounds, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+MU_R*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    B_z=B_r_yoke*B_field_z(z, R_1, R_2, d)

    ax.clear()
    ax.plot(z, B_z, linestyle='-', color='b', label='$B(z)$')
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='0')
    ax.set_xlabel('$z$ (mm)', fontsize=14)
    ax.set_ylabel('$B(z)$ (T)', fontsize=14)
    ax.set_title('Interactief $B(z)$ veld', fontsize=16)
    ax.grid()
    ax.legend()

    return


def plot_B_field_interactive(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds=30, n=1000):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.55)

    plot_B_field(ax, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds, n)

    # Slider axes
    ax_R1 = plt.axes([0.25, 0.45, 0.65, 0.03])
    ax_R2 = plt.axes([0.25, 0.40, 0.65, 0.03])
    ax_R1_magnet = plt.axes([0.25, 0.35, 0.65, 0.03])
    ax_R2_magnet = plt.axes([0.25, 0.30, 0.65, 0.03])
    ax_d = plt.axes([0.25, 0.25, 0.65, 0.03])
    ax_d_magnet = plt.axes([0.25, 0.20, 0.65, 0.03])
    ax_B_r_magnet = plt.axes([0.25, 0.15, 0.65, 0.03])

    # Sliders
    s_R1 = Slider(ax_R1, "$R_1$ (mm)", 0.1, 5, valinit=R_1)
    s_R2 = Slider(ax_R2, "$R_2$ (mm)", 0.15, 15, valinit=R_2)
    s_R1_magnet = Slider(ax_R1_magnet, "$R_{1, m}$ (mm)", 1, 20, valinit=R_1_magnet)
    s_R2_magnet = Slider(ax_R2_magnet, "$R_{2, m}$ (mm)", 1, 20, valinit=R_2_magnet)
    s_d = Slider(ax_d, "$d_g$ (mm)", 0.1, 6, valinit=d)
    s_d_magnet = Slider(ax_d_magnet, "$d_{m}$ (mm)", 0.1, 10, valinit=d_magnet)
    s_B_r_magnet = Slider(ax_B_r_magnet, "$B_r (T)$", 0, 2, valinit=B_r_magnet)

    # Update function
    def update(val):
        plot_B_field(ax,
                     s_R1.val,
                     s_R2.val,
                     s_R1_magnet.val,
                     s_R2_magnet.val,
                     s_d.val,
                     s_d_magnet.val,
                     s_B_r_magnet.val)
        fig.canvas.draw_idle()

    # Connect sliders to update
    s_R1.on_changed(update)
    s_R2.on_changed(update)
    s_R1_magnet.on_changed(update)
    s_R2_magnet.on_changed(update)
    s_d.on_changed(update)
    s_d_magnet.on_changed(update)
    s_B_r_magnet.on_changed(update)

    plt.show()

def BH_curve_magnet(B_max=None):
    comma_to_float = lambda s: float(s.replace(',', '.'))

    data = np.genfromtxt('NdFeB_BH_curve.csv',
                        delimiter=';',
                        converters={0: comma_to_float, 1: comma_to_float})
    H = data[:, 0]/10 # Tesla
    B = data[:, 1]/10 # Tesla

    if B_max is not None:
        B=B*B_max/B[0]
    spline = CubicSpline(H, B)

    return spline

def calculate_operating_point(spline, reluctance_correction):
    Pci = 1/(1-reluctance_correction)
    def func(x):
        return spline(x)-x*Pci

    h_op = float(fsolve(func, 1)[0])
    b_op = float(spline(h_op))
    return h_op, b_op

def plot_operating_point(spline, reluctance_correction):
    h_op, b_op = calculate_operating_point(spline, reluctance_correction)
    fig, ax = plt.subplots()
    H_eval = np.linspace(spline.x[0],spline.x[-1], 100)
    B_eval = spline(H_eval)
    ax.plot(H_eval, B_eval, color='blue', label="B-H curve")
    ax.plot(H_eval, H_eval/(1-reluctance_correction), color='red', label="Pci")
    ax.plot(h_op, b_op, 'o', markersize=6, color='red', label="Operating point")
    ax.set_xlabel('H (T)', fontsize=14)
    ax.set_ylabel('B (T)', fontsize=14)
    ax.set_ylim(0, np.max(B_eval)+0.1)
    ax.set_title('Operating point', fontsize=16)
    ax.grid()
    ax.legend()

    plt.show()

def make_lens_interactive():

    # ── Magic constants ───────────────────────────────────────────────────────
    R_1 = 0.8
    R_2 = 3.25
    R_1_magnet=4.5
    R_2_magnet=6
    d = 0.8
    d_magnet=2
    B_r_magnet_theoretical=1.17
    leak_factor=1
    B_r_magnet=B_r_magnet_theoretical*leak_factor
    T = 30*10**3
    SETUP_LENGTH          = 228
    LENS_POSITION         = 114
    BOUNDS                = 30
    N                     = 1000

    # ── Stage 1 : interactive B-field plot ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Adjust sliders, then press  Enter  or close the window\n"
        "to simulate lens properties",
        fontsize=11,
    )
    plt.subplots_adjust(left=0.25, bottom=0.55)

    plot_B_field(ax, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, BOUNDS, N)

    SL, SW, SH = 0.25, 0.65, 0.03
    ax_R1        = fig.add_axes([SL, 0.45, SW, SH])
    ax_R2        = fig.add_axes([SL, 0.40, SW, SH])
    ax_R1_magnet = fig.add_axes([SL, 0.35, SW, SH])
    ax_R2_magnet = fig.add_axes([SL, 0.30, SW, SH])
    ax_d         = fig.add_axes([SL, 0.25, SW, SH])
    ax_d_magnet  = fig.add_axes([SL, 0.20, SW, SH])
    ax_B_r       = fig.add_axes([SL, 0.15, SW, SH])

    s_R1        = Slider(ax_R1,        r"$R_1$ (mm)",       0.10, 5.0,  valinit=R_1)
    s_R2        = Slider(ax_R2,        r"$R_2$ (mm)",       0.15, 15.0, valinit=R_2)
    s_R1_magnet = Slider(ax_R1_magnet, r"$R_{1,m}$ (mm)",   1.0,  20.0, valinit=R_1_magnet)
    s_R2_magnet = Slider(ax_R2_magnet, r"$R_{2,m}$ (mm)",   1.0,  20.0, valinit=R_2_magnet)
    s_d         = Slider(ax_d,         r"$d_g$ (mm)",       0.10, 6.0,  valinit=d)
    s_d_magnet  = Slider(ax_d_magnet,  r"$d_m$ (mm)",       0.10, 10.0, valinit=d_magnet)
    s_B_r       = Slider(ax_B_r,       r"$B_r$ (T)",        0.0,  2.0,  valinit=B_r_magnet)

    def update(_val):
        plot_B_field(
            ax,
            s_R1.val, s_R2.val,
            s_R1_magnet.val, s_R2_magnet.val,
            s_d.val, s_d_magnet.val,
            s_B_r.val,
            BOUNDS, N,
        )
        fig.canvas.draw_idle()

    for s in (s_R1, s_R2, s_R1_magnet, s_R2_magnet, s_d, s_d_magnet, s_B_r):
        s.on_changed(update)

    # Capture slider values when the user is done (Enter or window close)
    captured = {}

    def _capture():
        captured["R_1"]        = s_R1.val
        captured["R_2"]        = s_R2.val
        captured["R_1_magnet"] = s_R1_magnet.val
        captured["R_2_magnet"] = s_R2_magnet.val
        captured["d"]          = s_d.val
        captured["d_magnet"]   = s_d_magnet.val
        captured["B_r_magnet"] = s_B_r.val

    def on_key(event):
        if event.key == "enter":
            _capture()
            plt.close(fig)

    def on_close(_event):
        if not captured:   # only fill if Enter wasn't pressed already
            _capture()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event",     on_close)

    plt.show()   # blocks here until the window is closed

    if not captured:
        return   # user closed without interacting

    # ── Stage 2 : build lens and show properties ──────────────────────────────
    print("Simulating lens — please wait …")

    lens = Lens(
        captured["R_1"],
        captured["R_2"],
        captured["R_1_magnet"],
        captured["R_2_magnet"],
        captured["d"],
        captured["d_magnet"],
        captured["B_r_magnet"],
        B_r_magnet_theoretical,
        T,
        setup_length=SETUP_LENGTH,
        lens_position=LENS_POSITION,
        n=N,
    )

    lens.calculate_B_field()
    lens.calculate_lens_properties()
    lens.display_properties()


class Lens:
    def __init__(self, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T, setup_length=228, lens_position=114, n=10000):

        self.dx = setup_length/(n-1)
        self.R_1 = R_1
        self.R_2 = R_2
        self.R_1_magnet = R_1_magnet
        self.R_2_magnet = R_2_magnet
        self.d = d
        self.d_magnet = d_magnet
        self.B_r_magnet = B_r_magnet
        self.T = T
        self.setup_length=setup_length
        self.lens_position=lens_position
        self.n=n
        self.z_eval_full=np.linspace(0, setup_length, n)
        self.update_T()
        self.update_B_r_yoke()



        self.mesh_list=[]

    def update_B_r_yoke(self):

        self.BH_curve = BH_curve_magnet(self.B_r_magnet)

        self.A_magnet=(self.R_2_magnet**2-self.R_1_magnet**2)
        self.A_gap=(self.R_2**2-self.R_1**2)
        self.reluctance_correction = (1+(self.A_magnet*self.d)/(self.A_gap*self.d_magnet))**-1
        self.H_op, self.B_op = calculate_operating_point(self.BH_curve, self.reluctance_correction)
        self.B_r_yoke=self.A_magnet/self.A_gap*(self.B_op-self.H_op)
        self.Pci = 1+(self.A_gap*self.d_magnet)/(self.A_magnet*self.d)

        self.calculate_B_field()

    def update_T(self):
        self.T_rel = self.T*(1+EPSILON*self.T)


    @staticmethod
    def angle_to_slope(angle):
        return np.tan(angle)

    def setup_parameters(self, object_pos, object_height, lens_pos):
        self.lens_position = lens_pos
        self.object_height = object_height
        self.object_plane = object_pos
        self.object_plane_rel = object_pos-lens_pos

        self.z_eval = np.linspace(self.object_plane, self.setup_length, int(self.n*(1-self.object_plane/self.setup_length)))

        self.calculate_B_field()
        self.calculate_lens_properties()
        self.calculate_image_properties()

    def calculate_B_field(self):
        self.B_field = calculate_B_field(self.n, self.setup_length, self.lens_position, self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet)
        self.B_field_d1 = calculate_B_field_d1(self.n, self.setup_length, self.lens_position, self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet)
        self.B_field_d2 = calculate_B_field_d2(self.n, self.setup_length, self.lens_position, self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet)

    def plot_B_field(self):
        fig, ax = plt.subplots()
        ax.plot(self.z_eval_full, self.B_field, linestyle='-', color='b', label='$B_z$ veld')
        ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='$B_z$=0')
        ax.set_xlabel('z (mm)')
        ax.set_ylabel('$B_z$ (Tesla)')
        ax.grid()
        ax.legend()
        plt.show()

    def ray_trace(self, initial_value, initial_slope, object_plane=None):
        def B(z): return self.B_r_yoke*B_field_z(z-self.lens_position, self.R_1, self.R_2, self.d)

        def paraxialequations(z, r):
            x1, x2 = r
            dx1dz = x2
            dx2dz = - (ETA*B(z))**2/(4*self.T_rel)*x1/10**6 # Conversion to mm
            return [dx1dz, dx2dz]
        if object_plane is not None:
            z_eval = np.linspace(object_plane, self.setup_length, int(self.n*(1-object_plane/self.setup_length)))
            sol=solve_ivp(paraxialequations, (object_plane, self.setup_length), np.array([initial_value, initial_slope]), t_eval=z_eval, method='DOP853')
        else:
            z_eval = self.z_eval_full
            sol=solve_ivp(paraxialequations, (0, self.setup_length), np.array([initial_value, initial_slope]), t_eval=self.z_eval_full, method='DOP853')

        return np.array(sol.y[0]), z_eval

    def calculate_GH(self, object_plane):
        self.G, self.ray_trace_z=self.ray_trace(1, 0, object_plane)
        self.H, _=self.ray_trace(0, 1, object_plane)

    def calculate_aberration_coeff(self):
        self.update_T()
        self.update_B_r_yoke()
        self.calculate_B_field()
        self.calculate_GH(0)

        first_term = 3/(8*self.f)
        integral1 = 4*ETA**2/self.T_rel*self.B_field**4
        integral2 = 5*self.B_field_d1**2
        integral3 = -self.B_field*self.B_field_d2
        self.D=first_term + ETA**2/(48*self.T_rel)*np.sum((integral1 + integral2 + integral3)*self.G**3 * self.H*10**-12)*self.dx
        self.C_M=-ETA**2/(4*self.T_rel)*np.sum(self.B_field**2 * self.G * self.H*10**-6)*self.dx
        self.C_theta=ETA/(4*np.sqrt(self.T_rel))*np.sum(self.B_field)*self.dx

    def calculate_lens_properties(self):
        self.G, _=self.ray_trace(1, 0)
        self.H, _=self.ray_trace(0, 1)

        self.Gi=(self.G[-1]-self.G[-2])/self.dx
        self.f=-1/self.Gi

        self.y_intercept = self.G[-1] - self.Gi * self.setup_length
        Z_Fi_abs = -self.y_intercept / self.Gi
        self.Z_Fi = Z_Fi_abs - self.lens_position   # relative to lens centre
        self.Z_Pi = self.Z_Fi - self.f
        self.Z_Fo = -self.Z_Fi
        self.Z_Po = -self.Z_Pi

        self.asymptotic_image_ray = self.Gi*self.z_eval_full+self.y_intercept

    def calculate_image_properties(self):
            self.image_plane_rel = self.Z_Pi + (self.object_plane_rel - self.Z_Po) * self.f / (self.object_plane_rel - self.Z_Po+self.f)
            self.image_plane = self.image_plane_rel + self.lens_position
            self.M = -(self.image_plane_rel - self.Z_Pi) / (self.object_plane_rel - self.Z_Po)
            print(f"Object plane: {self.object_plane_rel} mm, Image plane: {self.image_plane_rel} mm, Magnification: {self.M}")

    def display_properties(self, limits = None, output_path=None, dpi=100):
        print(f"Maximum flux density = {self.B_r_yoke:.6f} T")
        print(f"Pci = {self.Pci:.6f}")
        print(f"Z_Fi = {self.Z_Fi:.6f} mm")
        print(f"Z_Pi = {self.Z_Pi:.6f} mm")
        print(f"f = {self.f:.6f} mm")

        fig, ax=plt.subplots()

        ax.plot(self.z_eval_full, self.G, color='black', linewidth=2, label='Baan elektron')
        ax.plot(self.z_eval_full, self.asymptotic_image_ray, color='red', linewidth=1, label='Beeld asymtotische bundel')
        ax.plot(self.z_eval_full, np.ones(self.n), color='blue', linewidth=1, label='Object asymtotische bundel')
        ax.plot((self.lens_position + self.Z_Fi)*np.ones(2), [-5, 5], linestyle='--', label='Achterste focaalvlak')
        ax.plot((self.lens_position + self.Z_Pi)*np.ones(2), [-5, 5], linestyle='--', label='Beeld hoofdvlak')

        ax.plot(self.z_eval_full, self.B_field/np.max(self.B_field)*3, linewidth=0.5, linestyle='--', color='red', label='$B(z)$')
        ax.set_ylim(-5, 5)
        ax.set_xlabel("$z$ (mm)", fontsize=14)
        ax.set_ylabel("$r$ (mm)", fontsize=14)
        ax.set_title("Lens eigenschappen", fontsize=16)
        ax.grid()
        ax.legend()
        plt.tight_layout()

        if limits:
            ax.set_xlim(limits[0], limits[1])

        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

    def plot_trajectories(self, initial_values, limits = None):
        fig, ax = plt.subplots()
        ### PLot trajectories ##
        for i in range(0,np.shape(initial_values)[0]):
            traj=initial_values[i, 0]*self.G+initial_values[i, 1]*self.H
            ax.plot(self.z_eval_full, traj, color='blue', linewidth=1)
        ax.plot(self.z_eval_full, self.B_field/np.max(self.B_field)*3, color='red')
        ax.grid()
        if limits:
            ax.set_xlim(limits[0], limits[1])
        plt.show()

    def plot_setup(self, initial_values=None, limits = None, report=False):
        self.calculate_GH(self.object_plane)
        fig, ax = plt.subplots()
        if initial_values is not None:
            for i in range(0,np.shape(initial_values)[0]):
                traj=initial_values[i, 0]*self.G+self.angle_to_slope(initial_values[i, 1])*self.H
                ax.plot(self.z_eval, traj, color='blue', linewidth=1)
        ax.plot(self.z_eval_full, self.B_field/np.max(self.B_field)*3, color='red', label='$B(z)$')

        ax.plot((self.Z_Po+self.lens_position)*np.ones(2), [-5, 5], linestyle='--', label='Object hoofdvlak')
        ax.plot((self.Z_Fi+self.lens_position)*np.ones(2), [-5, 5], linestyle='--', label='Achterste focaalvlak')
        ax.plot((-self.Z_Fi+self.lens_position)*np.ones(2), [-5, 5], linestyle='--', label='Voorste focaalvlak')
        ax.plot((self.object_plane)*np.ones(2), [-5, 5], linestyle='--', label='Objectvlak')
        ax.plot((self.image_plane)*np.ones(2), [-5, 5], linestyle='--', label='Beeldvlak')
        if limits:
            ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(auto=True)
        ax.set_xlabel("z (mm)", fontsize=14)
        ax.set_ylabel("r (mm)", fontsize=14)
        ax.set_title("Stralendiagram", fontsize=16)
        ax.grid()
        ax.legend()
        plt.tight_layout()
        if report==True:
            return ax
        else:
            plt.show()

    def discretize_ray(self, ray, z_eval, bins, range_x, range_y):
        hist, _, _ =np.histogram2d(ray, z_eval, bins, [range_x, range_y])
        return (hist>0).astype(int) # Return binary histogram

    def random_ray(self, object_height, opening_angle):
        initial_value = np.random.uniform(-object_height, object_height)
        initial_slope = np.random.uniform(-self.angle_to_slope(opening_angle), self.angle_to_slope(opening_angle))
        return initial_value*self.G + initial_slope*self.H

    def no_collision_ray(self, object_height, opening_angle):
        ray = self.random_ray(object_height, opening_angle)
        collision = False
        for mesh in self.mesh_list:
            if mesh.check_collision(ray, self.ray_trace_z):
                collision = True
        if collision==True:
            return self.no_collision_ray(object_height, opening_angle)
        else:
            return ray

    def monte_carlo(self, object_height, opening_angle, camera_pos=None, pixel_size=55*10**-3, pixel_count=256, voxel_length=0.1, output_path=None):
        r_range = [-pixel_count/2 * pixel_size, pixel_count/2 * pixel_size]
        z_bins = int(self.setup_length / voxel_length)
        histogram = np.zeros((pixel_count, z_bins))
        self.calculate_GH(self.object_plane)

        # Default camera to end of setup if not specified
        if camera_pos is None:
            camera_pos = self.setup_length

        plt.ion()
        fig, (ax_hist, ax_cam) = plt.subplots(1, 2, figsize=(14, 5))

        # --- Histogram plot ---
        z_start = self.z_eval_full[0]
        z_end = self.z_eval_full[-1]
        histogram_plot = ax_hist.imshow(
            histogram, origin='lower', aspect='auto',
            extent=[z_start, z_end, r_range[0], r_range[1]]
        )

        lens_z_start = self.lens_position - self.d / 2
        lens_z_end = self.lens_position + self.d / 2
        ax_hist.axvspan(lens_z_start, lens_z_end, alpha=0.3, color='cyan', label='Lens')

        for i, mesh in enumerate(self.mesh_list):
            ax_hist.axvline(x=mesh.pos, color='red', linestyle='--',
                            linewidth=1.5, label=f'Mesh {i+1} (period={mesh.line_dist}mm)')

        # Draw camera position on histogram
        ax_hist.axvline(x=camera_pos, color='yellow', linestyle='-',
                        linewidth=1.5, label=f'Camera (z={camera_pos}mm)')

        ax_hist.set_xlabel('z (mm)')
        ax_hist.set_ylabel('r (mm)')
        ax_hist.set_title('Ray Monte Carlo Simulation')
        ax_hist.legend(loc='upper right', fontsize=8)

        # --- Camera image plot ---
        camera_image = self.camera_image(camera_pos, histogram)
        camera_plot = ax_cam.imshow(
            camera_image, origin='lower', aspect='equal', cmap='gist_gray'
        )
        ax_cam.set_xlabel('x (pixels)')
        ax_cam.set_ylabel('y (pixels)')
        ax_cam.set_title(f'Camera image at z = {camera_pos} mm')

        plt.tight_layout()

        i = 0
        try:
            while True:
                ray = self.no_collision_ray(object_height, opening_angle)
                histogram += self.discretize_ray(ray, self.ray_trace_z, [pixel_count, z_bins], r_range, [0, self.setup_length])
                if i % 100 == 0:
                    histogram_plot.set_data(histogram)
                    histogram_plot.set_clim(vmin=0, vmax=histogram.max())

                    camera_image = self.camera_image(camera_pos, histogram, voxel_length)
                    camera_plot.set_data(camera_image)
                    camera_plot.set_clim(vmin=0, vmax=camera_image.max())

                    plt.draw()
                    plt.pause(0.001)
                i += 1
        except KeyboardInterrupt:
            print("Simulation ended.")
            if output_path is not None:
                plt.savefig(output_path + "_plot.png", dpi=360)
                settings = {
                    "lens position": self.lens_position,
                    "mesh position": self.mesh_list[0].pos,
                    "mesh line distance": self.mesh_list[0].line_dist,
                    "mesh line thickness": self.mesh_list[0].line_thickness,
                    "object position": self.object_plane,
                    "object height": object_height,
                    "opening angle": opening_angle,
                    "camera position": camera_pos,
                    "pixel size": pixel_size,
                    "pixel count": pixel_count,
                    "voxel length": voxel_length,
                }

                with open(output_path + "_settings.txt", "w") as f:
                    for key, value in settings.items():
                        f.write(f"{key}: {value}\n")

    def variable_R_1(self, R_1_min, R_1_max, R_1_n, output_path=None, dpi=100):
        original_R_1 = self.R_1
        def return_properties(R_1):
            self.R_1 = R_1
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            return self.Z_Fi, self.Z_Pi, self.f, self.Pci

        R_1_eval=np.linspace(R_1_min, R_1_max, R_1_n)
        results = np.array([return_properties(R_1) for R_1 in R_1_eval])
        Z_Fi_values, Z_Pi_values, f_values, Pci_values = results.T
        _, ax=plt.subplots()

        _, ax1 = plt.subplots()

        ax1.plot(R_1_eval, f_values, color='red', label='$f$')
        ax1.plot(R_1_eval, Z_Pi_values, color='blue', label='$z_{Pi}$')
        ax1.plot(R_1_eval, Z_Fi_values, color='green', label='$z_{Fi}$')
        ax1.set_xlabel("$R_1$ (mm)", fontsize=14)
        ax1.set_ylabel("Lenseigenschap (mm)", fontsize=14)
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(R_1_eval, Pci_values, color='orange', linestyle='--', label='$P_{ci}$')
        ax2.set_ylabel('$P_{ci}$', fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Lenseigenschappen voor variable $R_1$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_R_1)

    def variable_R_1_ab(self, R_1_min, R_1_max, R_1_n, output_path=None, dpi=100):
        plt.rcParams['text.usetex'] = True
        original_R_1 = self.R_1
        def return_properties(R_1):
            self.R_1 = R_1
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            self.calculate_aberration_coeff()
            return self.D, self.C_M, self.C_theta

        R_1_eval=np.linspace(R_1_min, R_1_max, R_1_n)
        results = np.array([return_properties(R_1) for R_1 in R_1_eval])
        D_values, C_M_values, _ = results.T
        _, ax1 = plt.subplots()

        ax1.plot(R_1_eval, C_M_values, color='blue', label='$C_M$')
        ax1.set_xlabel("$R_1$ (mm)", fontsize=14)
        ax1.set_ylabel("$C_M$ (dimensieloos)", fontsize=14)
        ax1.grid()
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(R_1_eval, D_values, color='red', label='$D$')
        ax2.set_ylabel("$D$ ($mm^{-2}$)", fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Aberratiecoëfficienten voor variable $R_1$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_R_1)


    def variable_R_2(self, R_2_min, R_2_max, R_2_n, output_path=None, dpi=100):
        original_R_2 = self.R_2
        def return_properties(R_2):
            self.R_2 = R_2
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            return self.Z_Fi, self.Z_Pi, self.f, self.Pci

        R_2_eval=np.linspace(R_2_min, R_2_max, R_2_n)
        results = np.array([return_properties(R_2) for R_2 in R_2_eval])
        Z_Fi_values, Z_Pi_values, f_values, Pci_values = results.T
        _, ax=plt.subplots()

        _, ax1 = plt.subplots()

        ax1.plot(R_2_eval, f_values, color='red', label='$f$')
        ax1.plot(R_2_eval, Z_Pi_values, color='blue', label='$z_{Pi}$')
        ax1.plot(R_2_eval, Z_Fi_values, color='green', label='$z_{Fi}$')
        ax1.set_xlabel("$R_2$ (mm)", fontsize=14)
        ax1.set_ylabel("Lenseigenschap (mm)", fontsize=14)
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(R_2_eval, Pci_values, color='orange', linestyle='--', label='$P_{ci}$')
        ax2.set_ylabel('$P_{ci}$', fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Lenseigenschappen voor variable $R_2$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_R_2)

    def variable_R_2_ab(self, R_2_min, R_2_max, R_2_n, output_path=None, dpi=100):
        plt.rcParams['text.usetex'] = True
        original_R_2 = self.R_2
        def return_properties(R_2):
            self.R_2 = R_2
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            self.calculate_aberration_coeff()
            return self.D, self.C_M, self.C_theta

        R_2_eval=np.linspace(R_2_min, R_2_max, R_2_n)
        results = np.array([return_properties(R_2) for R_2 in R_2_eval])
        D_values, C_M_values, _ = results.T
        _, ax1 = plt.subplots()

        ax1.plot(R_2_eval, C_M_values, color='blue', label='$C_M$')
        ax1.set_xlabel("$R_2$ (mm)", fontsize=14)
        ax1.set_ylabel("$C_M$ (dimensieloos)", fontsize=14)
        ax1.grid()
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(R_2_eval, D_values, color='red', label='$D$')
        ax2.set_ylabel("$D$ ($mm^{-2}$)", fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Aberratiecoëfficienten voor variable $R_2$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_R_2)

    def variable_d(self, d_min, d_max, d_n, output_path=None, dpi=100):
        original_d = self.d
        def return_properties(d):
            self.d = d
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            return self.Z_Fi, self.Z_Pi, self.f, self.Pci

        d_eval = np.linspace(d_min, d_max, d_n)
        results = np.array([return_properties(d) for d in d_eval])
        Z_Fi_values, Z_Pi_values, f_values, Pci_values = results.T

        _, ax1 = plt.subplots()

        ax1.plot(d_eval, f_values, color='red', label='$f$')
        ax1.plot(d_eval, Z_Pi_values, color='blue', label='$z_{Pi}$')
        ax1.plot(d_eval, Z_Fi_values, color='green', label='$z_{Fi}$')
        ax1.set_xlabel("$d$ (mm)", fontsize=14)
        ax1.set_ylabel("Lenseigenschap (mm)", fontsize=14)
        ax1.grid()

        ax2 = ax1.twinx()
        ax2.plot(d_eval, Pci_values, color='orange', linestyle='--', label='$P_{ci}$')
        ax2.set_ylabel('$P_{ci}$', fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Lenseigenschappen voor variable $d$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_d)
    
    def variable_d_ab(self, d_min, d_max, d_n, output_path=None, dpi=100):
        plt.rcParams['text.usetex'] = True
        original_d = self.d
        def return_properties(d):
            self.d = d
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            self.calculate_aberration_coeff()
            return self.D, self.C_M, self.C_theta

        d_eval=np.linspace(d_min, d_max, d_n)
        results = np.array([return_properties(d) for d in d_eval])
        D_values, C_M_values, _ = results.T
        _, ax1 = plt.subplots()

        ax1.plot(d_eval, C_M_values, color='blue', label='$C_M$')
        ax1.set_xlabel("$d$ (mm)", fontsize=14)
        ax1.set_ylabel("$C_M$ (dimensieloos)", fontsize=14)
        ax1.grid()
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(d_eval, D_values, color='red', label='$D$')
        ax2.set_ylabel("$D$ ($mm^{-2}$)", fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Aberratiecoëfficienten voor variable $d$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_d)

    def variable_B_r(self, B_r_min, B_r_max, B_r_n, output_path=None, dpi=100):
        original_B_r = self.B_r_magnet
        def return_properties(B_r):
            self.B_r_magnet = B_r
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            return self.Z_Fi, self.Z_Pi, self.f, self.reluctance_correction

        B_r_eval=np.linspace(B_r_min, B_r_max, B_r_n)
        results = np.array([return_properties(B_r) for B_r in B_r_eval])
        Z_Fi_values, Z_Pi_values, f_values, reluctance_correction_values = results.T
        _, ax=plt.subplots()

        _, ax1 = plt.subplots()

        ax1.plot(B_r_eval, f_values, color='red', label='$f$')
        ax1.plot(B_r_eval, Z_Pi_values, color='blue', label='$z_{Pi}$')
        ax1.plot(B_r_eval, Z_Fi_values, color='green', label='$z_{Fi}$')
        ax1.set_xlabel("$B_r$ (mm)", fontsize=14)
        ax1.set_ylabel("Lenseigenschap (mm)", fontsize=14)
        ax1.grid()

        ax1.legend()

        ax1.set_title(f"Lenseigenschappen voor variable $B_r$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_B_r)

    def variable_B_r_ab(self, B_r_min, B_r_max, B_r_n, output_path=None, dpi=100):
        plt.rcParams['text.usetex'] = True
        original_B_r = self.B_r_magnet
        def return_properties(B_r):
            self.B_r_magnet = B_r
            self.update_B_r_yoke()
            self.calculate_lens_properties()
            self.calculate_aberration_coeff()
            return self.D, self.C_M, self.C_theta

        B_r_eval=np.linspace(B_r_min, B_r_max, B_r_n)
        results = np.array([return_properties(B_r) for B_r in B_r_eval])
        D_values, C_M_values, _ = results.T
        _, ax1 = plt.subplots()

        ax1.plot(B_r_eval, C_M_values, color='blue', label='$C_M$')
        ax1.set_xlabel("$B_r$ (mm)", fontsize=14)
        ax1.set_ylabel("$C_M$ (dimensieloos)", fontsize=14)
        ax1.grid()
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(B_r_eval, D_values, color='red', label='$D$')
        ax2.set_ylabel("$D$ ($mm^{-2}$)", fontsize=14)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.set_title(f"Aberratiecoëfficienten voor variable $B_r$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_B_r)

    def variable_T(self, T_min, T_max, T_n, output_path=None, dpi=100):
        original_T = self.T*10**-3
        def return_properties(T):
            self.T = T*10**3
            self.update_T()
            self.calculate_lens_properties()
            return self.Z_Fi, self.Z_Pi, self.f

        T_eval=np.linspace(T_min, T_max, T_n)
        results = np.array([return_properties(T) for T in T_eval])
        Z_Fi_values, Z_Pi_values, f_values = results.T
        _, ax=plt.subplots()

        _, ax1 = plt.subplots()

        ax1.plot(T_eval, f_values, color='red', label='$f$')
        ax1.plot(T_eval, Z_Pi_values, color='blue', label='$z_{Pi}$')
        ax1.plot(T_eval, Z_Fi_values, color='green', label='$z_{Fi}$')
        ax1.set_xlabel("$T$ (KeV)", fontsize=14)
        ax1.set_ylabel("Lenseigenschap (mm)", fontsize=14)
        ax1.grid()

        ax1.legend()

        ax1.set_title(f"Lenseigenschappen voor variable $T$", fontsize=16)

        plt.tight_layout()
        if output_path is not None:
            plt.savefig(output_path, dpi=dpi)
        else:
            plt.show()

        return_properties(original_T)

    def add_mesh(self, mesh):
        self.mesh_list.append(mesh)

    def camera_image(self, pos, histogram, voxel_length=0.1):
        z_bins = histogram.shape[1]
        z_axis = np.linspace(0, self.setup_length, z_bins)
        index = np.argmin(np.abs(z_axis - pos))
        profile = histogram[:, index]
        return np.outer(profile, profile)

class Mesh:
    def __init__(self, pos, line_dist, line_thickness, line_count=100, line_offset=0):
        self.pos=pos
        self.line_dist=line_dist
        self.line_thickness=line_thickness
        self.line_count=line_count
        self.line_offset=line_offset

    def mesh_array(self, pixel_size, pixel_count):
        """Make a discretized array of the mesh"""
        mesh_array=np.zeros(pixel_count)
        for i in 1/2*np.linspace(-pixel_count, pixel_count):
            dist = i*pixel_size
            dist_rel = np.mod(dist+self.line_offset, self.line_dist)
            mesh_array[i] = int(dist_rel>self.line_thickness)
        pass

    def check_collision(self, ray_r, ray_z):
        """Check if a ray collides with the mesh"""
        collision_index = np.argmin(np.abs(ray_z - self.pos))

        collision_r_pos = ray_r[collision_index]
        collision_r_pos_rel = np.mod(collision_r_pos+self.line_offset, self.line_dist)
        return collision_r_pos_rel<self.line_thickness


if __name__ == "__main__":
    R_1 = 0.8
    R_2 = 3.25
    R_1_magnet=4.5
    R_2_magnet=6
    d = 0.8
    d_magnet=2
    B_r_magnet_theoretical=1.17
    leak_factor=1
    B_r_magnet=B_r_magnet_theoretical*leak_factor
    T = 30*10**3

    plot_B_field_interactive(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet)

    # permanent_magnet_lens = Lens(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T)
    # permanent_magnet_lens.setup_parameters(object_pos=11.5, object_height=1.5, lens_pos=27.98)
    # mesh1 = Mesh(pos=10, line_dist=254e-3, line_thickness=50e-3)
    # permanent_magnet_lens.add_mesh(mesh1)
    # permanent_magnet_lens.display_properties()

    # plot_operating_point(BH_curve_magnet(), 0.4)

    # opening_angle=100e-3
    # initial_values = np.linspace(-1, 1, 3)
    # initial_angles = np.linspace(-opening_angle, opening_angle, 5)
    # combinations = np.array(np.meshgrid(initial_values, initial_angles)).T.reshape(-1, 2)
    # permanent_magnet_lens.monte_carlo(object_height=0e-3, opening_angle=opening_angle, pixel_size=55e-3, camera_pos=132.42, pixel_count=256, voxel_length=0.05)

    # permanent_magnet_lens.variable_T(28, 32, 100)

    make_lens_interactive()