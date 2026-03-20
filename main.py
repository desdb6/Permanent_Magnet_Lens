import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider
from scipy.interpolate import CubicSpline

mu_r=1.05

def B_field_ring(z, R_1, R_2):
    return (z/1000)/2*(1/np.sqrt((z/1000)**2+(R_1/1000)**2)-1/np.sqrt((z/1000)**2+(R_2/1000)**2))

def B_field_z(z, R_1, R_2, d):
    return B_field_ring(z+d/2, R_1, R_2)-B_field_ring(z-d/2, R_1, R_2)

def plot_B_field(ax, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds=30, n=1000):
    z=np.linspace(-bounds, bounds, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    B_z=B_r_yoke*B_field_z(z, R_1, R_2, d)

    ax.clear()
    ax.plot(z, B_z, linestyle='-', color='b', label='$B_z$ veld')
    ax.axhline(0, color='red', linewidth=1.5, linestyle='--', label='$B_z$=0')
    ax.set_xlabel('z (mm)')
    ax.set_ylabel('$B_z$ (Tesla)')
    ax.grid()
    ax.legend()

    return

def plot_B_field_interactive(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, bounds=30, n=1000):
    # Create plot
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.55)  # leave space for more sliders

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
    s_R1_magnet = Slider(ax_R1_magnet, "$R_{1, magnet}$ (mm)", 1, 20, valinit=R_1_magnet)
    s_R2_magnet = Slider(ax_R2_magnet, "$R_{2, magnet}$ (mm)", 1, 20, valinit=R_2_magnet)
    s_d = Slider(ax_d, "$d$ (mm)", 0.1, 6, valinit=d)
    s_d_magnet = Slider(ax_d_magnet, "$d_{ magnet}$ (mm)", 0.1, 10, valinit=d_magnet)
    s_B_r_magnet = Slider(ax_B_r_magnet, "$B_{remanence}$", 0, 2, valinit=B_r_magnet)

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

def trajectory_solver(r0, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta=296548.4789, bounds=30, n=1000, object_plane = None):
    if object_plane == None:
        object_plane = -bounds

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    def B(z): return B_r_yoke*B_field_z(z, R_1, R_2, d)

    epsilon=9.78475592*10**-7
    T=T*(1+epsilon*T) # Relativistic

    def paraxialequations(z, r):
        x1, x2 = r
        dx1dz = x2
        dx2dz = - (eta*B(z))**2/(4*T)*x1/10**6 # Conversion to mm
        return [dx1dz, dx2dz]

    z_span=(object_plane, bounds)
    z_eval=np.linspace(*z_span, n)

    sol=solve_ivp(paraxialequations, z_span, np.array(r0), t_eval=z_eval, method='DOP853')

    return np.array(sol.y[0])

def plot_trajectories(initial_values, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta=296548.4789, bounds=30, n=1000, object_plane = None):
    if object_plane == None:
        object_plane = -bounds

    fig, ax=plt.subplots()
    z_span=(object_plane, bounds)
    z_eval=np.linspace(*z_span, n)

    G=1/1000*trajectory_solver([1000, 0], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n, object_plane)
    H=1/1000*trajectory_solver([0, 1000], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n, object_plane)


    ### PLot trajectories ##
    for i in range(0,np.shape(initial_values)[0]):
        traj=initial_values[i, 0]*G+initial_values[i, 1]*H
        ax.plot(z_eval, traj, color='blue', linewidth=1)

    z_span_B_field=(-bounds, bounds)
    z_eval=np.linspace(*z_span_B_field, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    B_field=B_r_yoke*B_field_z(z_eval, R_1, R_2, d)
    ax.plot(z_eval, B_field/np.max(B_field)*3, color='red')
    ax.grid()
    plt.show()

def calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds=30, n=500000, eta=296548.4789):
    dx = 2 * bounds / (n - 1)
    z_span=(-bounds, bounds)
    z_eval=np.linspace(*z_span, n)

    G=1/10*trajectory_solver([10, 0], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n)
    H=1/10*trajectory_solver([0, 10], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n)

    Gi=(G[-1]-G[-2])/dx
    f=-1/Gi

    y_intercept=G[-1]-Gi*bounds
    Z_Fi=(np.argmin(np.abs(Gi * z_eval + y_intercept))-n/2)/n*bounds*2
    Z_Pi=Z_Fi-f

    return z_eval, G, Gi*z_eval+y_intercept, Z_Fi, Z_Pi, f

def display_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T, bounds=30, n=100000, eta=296548.4789):
    z_eval, G, asymptotic_image_ray, Z_Fi, Z_Pi, f=calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds, n)

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet_theoretical

    print(f"Maximum flux density = {B_r_yoke:.6f} T")
    print(f"Reluctance correction = {reluctance_correction:.6f}")
    print(f"Z_Fi = {Z_Fi:.6f} mm")
    print(f"Z_Pi = {Z_Pi:.6f} mm")
    print(f"f = {f:.6f} mm")

    fig, ax=plt.subplots()

    ax.plot(z_eval, G, color='black', linewidth=2, label='Electron path')
    ax.plot(z_eval, asymptotic_image_ray, color='red', linewidth=1, label='Asymtotic image ray')
    ax.plot(z_eval, np.ones(n), color='blue', linewidth=1, label='Asymtotic object ray')
    ax.plot(Z_Fi*np.ones(2), [-5, 5], linestyle='--', label='Backfocal plane')
    ax.plot(Z_Pi*np.ones(2), [-5, 5], linestyle='--', label='Image principal plane')

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    B_field=B_r_yoke*B_field_z(z_eval, R_1, R_2, d)
    ax.plot(z_eval, B_field/np.max(B_field)*3, linewidth=0.5, linestyle='--', color='red', label='B-field')

    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("r (mm)")
    ax.set_title("Lens properties")
    ax.grid()
    ax.legend()

    plt.show()

def variable_R_1(R_1_min, R_1_max, R_1_n, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds=30, n=100000, eta=296548.4789):
    def return_properties(R_1):
        _, _, _, Z_Fi, Z_Pi, f = calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds, n, eta)
        return Z_Fi, Z_Pi, f

    R_1_eval=np.linspace(R_1_min, R_1_max, R_1_n)
    results = np.array([return_properties(R_1) for R_1 in R_1_eval])
    Z_Fi_values, Z_Pi_values, f_values = results.T

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1_eval**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    fig, ax=plt.subplots(1, 3)

    ax[0].plot(R_1_eval, f_values, color='red', label='Focal length')
    ax[0].plot(R_1_eval, Z_Pi_values, color='blue', label='Principal plane')
    ax[0].plot(R_1_eval, Z_Fi_values, color='green', label='Backfocal plane')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel("R_1 (mm)")
    ax[0].set_ylabel("(mm)")

    ax[1].plot(R_1_eval, reluctance_correction, color='purple', label='Reluctance correction')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim(0, 1)

    ax[2].plot(R_1_eval, B_r_yoke, color='purple', label='Max flux density in polepiece')
    ax[2].legend()
    ax[2].grid()
    plt.show()

def variable_R_2(R_2_min, R_2_max, R_2_n, R_1, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds=30, n=100000, eta=296548.4789):
    def return_properties(R_2):
        _, _, _, Z_Fi, Z_Pi, f = calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds, n, eta)
        return Z_Fi, Z_Pi, f

    R_2_eval=np.linspace(R_2_min, R_2_max, R_2_n)
    results = np.array([return_properties(R_2) for R_2 in R_2_eval])
    Z_Fi_values, Z_Pi_values, f_values = results.T

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2_eval**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    fig, ax=plt.subplots(1, 3)

    ax[0].plot(R_2_eval, f_values, color='red', label='Focal length')
    ax[0].plot(R_2_eval, Z_Pi_values, color='blue', label='Principal plane')
    ax[0].plot(R_2_eval, Z_Fi_values, color='green', label='Backfocal plane')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel("R_2 (mm)")
    ax[0].set_ylabel("(mm)")

    ax[1].plot(R_2_eval, reluctance_correction, color='purple', label='Reluctance correction')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim(0, 1)

    ax[2].plot(R_2_eval, B_r_yoke, color='purple', label='Max flux density in polepiece')
    ax[2].legend()
    ax[2].grid()
    plt.show()

def variable_d(d_min, d_max, d_n, R_1, R_2, R_1_magnet, R_2_magnet, d_magnet, B_r_magnet, T, bounds=30, n=100000, eta=296548.4789):
    def return_properties(d):
        _, _, _, Z_Fi, Z_Pi, f = calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds, n, eta)
        return Z_Fi, Z_Pi, f

    d_eval=np.linspace(d_min, d_max, d_n)
    results = np.array([return_properties(d) for d in d_eval])
    Z_Fi_values, Z_Pi_values, f_values = results.T

    A_magnet=(R_2_magnet**2-R_1_magnet**2)
    A_gap=(R_2**2-R_1**2)
    reluctance_correction=(1+mu_r*(A_magnet*d_eval)/(A_gap*d_magnet))**-1
    B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

    fig, ax=plt.subplots(1, 2)

    ax[0].plot(d_eval, f_values, color='red', label='Focal length')
    ax[0].plot(d_eval, Z_Pi_values, color='blue', label='Principal plane')
    ax[0].plot(d_eval, Z_Fi_values, color='green', label='Backfocal plane')
    ax[0].legend()
    ax[0].grid()
    ax[0].set_xlabel("d (mm)")
    ax[0].set_ylabel("(mm)")

    ax[1].plot(d_eval, reluctance_correction, color='purple', label='Reluctance correction')
    ax[1].legend()
    ax[1].grid()
    ax[1].set_ylim(0, 1)
    plt.show()

def variable_B_r(B_r_magnet_min, B_r_magnet_max, B_r_magnet_n, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, T, bounds=30, n=100000, eta=296548.4789):
    def return_properties(B_r):
        _, _, _, Z_Fi, Z_Pi, f = calculate_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r, T, bounds, n, eta)
        return Z_Fi, Z_Pi, f

    B_eval=np.linspace(B_r_magnet_min, B_r_magnet_max, B_r_magnet_n)
    results = np.array([return_properties(B) for B in B_eval])
    Z_Fi_values, Z_Pi_values, f_values = results.T

    fig, ax=plt.subplots()

    ax.plot(B_eval, f_values, color='red', label='Focal length')
    ax.plot(B_eval, Z_Pi_values, color='blue', label='Principal plane')
    ax.plot(B_eval, Z_Fi_values, color='green', label='Backfocal plane')

    ax.legend()
    ax.grid()
    ax.set_xlabel("B_r (T)")
    ax.set_ylabel("(mm)")
    plt.show()

def plot_HB_Curve():
    data = np.loadtxt('HBCurve.txt')
    H = data[:, 0]
    B = data[:, 1]

    spline = CubicSpline(H, B)

    H_spline = np.linspace(H.min(), H.max(), 500)
    B_spline = spline(H_spline)

    fig, ax = plt.subplots()
    ax.plot(H, B, 'ro', label=None)
    ax.plot(H_spline, B_spline, color = "b", label='H-B curve')

    ax.set_xlabel("H (A/m)")
    ax.set_ylabel("B (T)")
    ax.grid()
    ax.legend()

    plt.show()

class Lens:
    def __init__(self, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T):

        self.R_1 = R_1
        self.R_2 = R_2
        self.R_1_magnet = R_1_magnet
        self.R_2_magnet = R_2_magnet
        self.d = d
        self.d_magnet = d_magnet
        self.B_r_magnet = B_r_magnet
        self.B_r_magnet_theoretical = B_r_magnet_theoretical
        self.T = T

    def calculate_properties(self, bounds=30, n=100000, eta=296548.4789):
        _, _, _, self.Z_Fi, self.Z_Pi, self.f = calculate_properties(self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet, self.T, bounds, n, eta)
        self.Z_Fo = -self.Z_Fi
        self.Z_Po = -self.Z_Pi

    def plot_B_field(self, ax, bounds=30, n=1000):
        plot_B_field(ax, self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet, bounds, n)

    def plot_trajectories(self, initial_values, eta=296548.4789, bounds=30, n=1000, object_plane=None):
        plot_trajectories(initial_values, self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet, self.T, eta, bounds, n, object_plane)

    def display_properties(self, bounds=30, n=100000, eta=296548.4789):
        display_properties(self.R_1, self.R_2, self.R_1_magnet, self.R_2_magnet, self.d, self.d_magnet, self.B_r_magnet, self.B_r_magnet_theoretical, self.T, bounds, n, eta)

    def calculate_image_properties(self, object_plane, object_height = 1):
        self.object_plane = object_plane
        self.object_height = object_height
        self.image_plane = self.Z_Pi + (self.object_plane - self.Z_Po) * self.f / (self.object_plane - self.Z_Po+self.f)
        self.M = -(self.image_plane - self.Z_Pi) / (self.object_plane - self.Z_Po)
        self.image_height = self.M * self.object_height
        print(f"Object plane: {self.object_plane} mm, Image plane: {self.image_plane} mm, Magnification: {self.M}, Image height: {self.image_height} mm")

    def calculate_image_properties_mag(self, mag):
        self.object_plane = self.Z_Po-(mag+1)/mag*self.f
        self.calculate_image_properties(self.object_plane)

    def plot_setup(self, eta=296548.4789, bounds=30, n=1000):
        r0_values = np.linspace(-self.object_height, self.object_height, 4)
        slope0_values = np.linspace(0, 0, 1)
        initial_values = np.array([[u, v] for u in r0_values for v in slope0_values])

        fig, ax=plt.subplots()
        z_span=(self.object_plane, bounds)
        z_eval=np.linspace(*z_span, n)

        G=1/1000*trajectory_solver([1000, 0], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n, self.object_plane)
        H=1/1000*trajectory_solver([0, 1000], R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, eta, bounds, n, self.object_plane)

        ### PLot trajectories ##
        for i in range(0,np.shape(initial_values)[0]):
            traj=initial_values[i, 0]*G+initial_values[i, 1]*H
            ax.plot(z_eval, traj, color='blue', linewidth=1)

        z_span_B_field=(-bounds, bounds)
        z_eval=np.linspace(*z_span_B_field, n)

        A_magnet=(R_2_magnet**2-R_1_magnet**2)
        A_gap=(R_2**2-R_1**2)
        reluctance_correction=(1+mu_r*(A_magnet*d)/(A_gap*d_magnet))**-1
        B_r_yoke=A_magnet/A_gap*reluctance_correction*B_r_magnet

        B_field=B_r_yoke*B_field_z(z_eval, R_1, R_2, d)
        ax.plot(z_eval, B_field/np.max(B_field)*3, color='red')

        ax.plot(self.Z_Po*np.ones(2), [-5, 5], linestyle='--', label='Object principal plane')
        ax.plot(self.Z_Fi*np.ones(2), [-5, 5], linestyle='--', label='Backfocal plane')
        ax.plot(-self.Z_Fi*np.ones(2), [-5, 5], linestyle='--', label='Frontfocal plane')
        ax.plot(self.image_plane*np.ones(2), [-5, 5], linestyle='--', label='Image plane')
        ax.set_xlim(-bounds, bounds)
        ax.set_ylim(auto=True)
        ax.set_xlabel("z (mm)")
        ax.set_ylabel("r (mm)")
        ax.set_title("Lens setup")
        ax.grid()
        ax.legend()
        plt.show()

if __name__ == "__main__":
    R_1 = 0.8
    R_2 = 3.25
    R_1_magnet=4.5
    R_2_magnet=6
    d = 0.8
    d_magnet=2
    B_r_magnet_theoretical=1.37
    leak_factor=1
    B_r_magnet=B_r_magnet_theoretical*leak_factor
    T = 30*10**3

    plot_B_field_interactive(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet)

    # display_properties(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T)

    # variable_R_1(0.5, 1.5, 100, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T)
    # variable_R_2(2, 4, 100, R_1, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T)
    # variable_d(0.3, 1.2, 100, R_1, R_2, R_1_magnet, R_2_magnet, d_magnet, B_r_magnet, T)
    # variable_B_r(0.5, 1.5, 100, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, T)

    # r0_values = np.linspace(-5, 5, 10)
    # slope0_values = np.linspace(-.4, .4, 5)
    # initial_values = np.array([[u, v] for u in r0_values for v in slope0_values])
    # plot_trajectories(initial_values, R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, T, bounds=15)

    # plot_HB_Curve()

    permanent_magnet_lens = Lens(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T)
    permanent_magnet_lens.calculate_properties()
    # permanent_magnet_lens.calculate_image_properties(-12.78)
    permanent_magnet_lens.calculate_image_properties_mag(mag=2)
    # permanent_magnet_lens.calculate_image_properties_mag(mag=3)
    # permanent_magnet_lens.calculate_image_properties_mag(mag=5)
    # permanent_magnet_lens.calculate_image_properties_mag(mag=10)
    permanent_magnet_lens.plot_setup(bounds = 30)