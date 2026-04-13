import numpy as np
import matplotlib.pyplot as plt
from sim import B_field_z, Lens

plt.rcParams['text.usetex'] = True
DPI=100

def B_field_ring(z, R_1, R_2):
    return (z)/2*(1/np.sqrt((z)**2+(R_1)**2)-1/np.sqrt((z)**2+(R_2)**2))

def B_field_lens(z, R_1, R_2, d):
    return B_field_ring(z+d/2, R_1, R_2)-B_field_ring(z-d/2, R_1, R_2)

def plot_B_field_ring_report():
    z_eval = np.linspace(-10, 10, 1000)
    B_eval = B_field_ring(z_eval, 1, 2)
    _, ax = plt.subplots()
    ax.plot([-10, 10], [0, 0], color='red', linestyle='dashed')
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel(r'$B_{ring}(z)$ (eenheid $\mu_0\sigma_M$)', fontsize=14)
    ax.set_title('$B_{ring}(z)$', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/B_field_1_annulus.png', dpi=DPI)
    print(f'Saved B field ring plot')

def plot_B_field_lens_report():
    z_eval = np.linspace(-15, 15, 1500)
    B_eval = B_field_lens(z_eval, 1, 2, 2)
    B_max = np.max(B_eval)
    _, ax = plt.subplots()
    ax.plot([-15, 15], [0, 0], color='red', linestyle='dashed')
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel(r'$B(z)$ (eenheid $\mu_0\sigma_M$)', fontsize=14)
    ax.set_ylim(-(B_max+0.05), B_max+0.05)
    ax.set_title('$B(z)$', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/B_field_2_annula.png', dpi=DPI)
    print(f'Saved B field lens plot')

def plot_B_field_SIMION_comparison():
    ### Numeric Field###
    numericpotential1=np.loadtxt("simion_potential/numericpotential1.txt")[:, 4]
    numericpotential2=np.loadtxt("simion_potential/numericpotential2.txt")[:, 4]
    numericpotential=numericpotential1-numericpotential2

    numericfield=np.zeros(np.shape(numericpotential))
    numericfield[0]=-(numericpotential[1]-numericpotential[0])
    numericfield[600]=-(numericpotential[600]-numericpotential[599])

    for i in range(1, 600):
        numericfield[i]=-(numericpotential[i+1]-numericpotential[i-1])/2

    bounds=3*10**-2
    n=601

    z_span=(-bounds, bounds)
    z_numeric=np.linspace(*z_span, n)

    ### Analytical field
    R_1 = 2
    R_2 = 3
    d = 4
    bounds=3*10**-2
    n=1000

    z_span=(-bounds, bounds)
    z_analytic=np.linspace(*z_span, n)
    analyticalfield=B_field_z(z_analytic, R_1/1000, R_2/1000, d/1000)

    z_numeric = z_numeric[::3]
    numericfield = numericfield[::3]

    fig, ax=plt.subplots()
    ax.plot(z_analytic*100, analyticalfield/np.max(analyticalfield), color='r', label='Analytisch')
    ax.plot(z_numeric*100, numericfield/np.max(numericfield), linestyle='None', marker='x', markersize=4, color='b',  label='Numeriek')
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$B(z)$ (arb. eenheid)', fontsize=14)
    ax.set_title('Vergelijking tussen analytische uitdrukking en numeriek berekend veld', fontsize=14)
    ax.legend()
    ax.plot([-3, 3], [0, 0], color='red', linestyle='dashed')
    plt.tight_layout()
    plt.savefig('report/Images/B_field_SIMION.png', dpi=DPI)
    print(f'Saved B field SIMION comparison plot')

def variable_plots():
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
    
    permanent_magnet_lens = Lens(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T)
    permanent_magnet_lens.setup_parameters(object_pos=0, object_height=1, lens_pos=100)

    permanent_magnet_lens.variable_R_1(0.6, 1.2, 50, 'report/Images/variable_R_1.png', dpi=DPI)
    print("Saved variable R1 plot")
    permanent_magnet_lens.variable_R_2(2.5, 4, 50, 'report/Images/variable_R_2.png', dpi=DPI)
    print("Saved variable R2 plot")
    permanent_magnet_lens.variable_d(0.6, 4, 50, 'report/Images/variable_d.png', dpi=DPI)
    print("Saved variable d plot")
    permanent_magnet_lens.variable_B_r(0.7, 1.42, 50, 'report/Images/variable_B_r.png', dpi=DPI)
    print("Saved variable Br plot")

def plot_glasers_field():
    z_eval = np.linspace(-5, 5, 1000)
    B_eval = 1/(1+z_eval**2)

    _, ax = plt.subplots()
    ax.plot([-5, 5], [0, 0], color='red', linestyle='dashed')
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$B(z)$', fontsize=14)
    ax.set_title('Glaser\'s bell-shaped field', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/Glasers_B_field.png', dpi=DPI)
    print(f'Saved Glasers B field lens plot')

def plot_constant_field():
    z_eval = np.linspace(-5, 5, 1000)
    B_eval = np.zeros(1000)
    B_eval[400:600]=1

    _, ax = plt.subplots()
    ax.plot([-5, 5], [0, 0], color='red', linestyle='dashed')
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$B(z)$', fontsize=14)
    ax.set_title('Constant veld', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/Constant_B_field.png', dpi=DPI)
    print(f'Saved Constant B field lens plot')

def plot_actual_field():
    z_eval = np.linspace(-5, 5, 1000)
    B_eval = B_field_lens(z_eval, 1, 2, 1)

    _, ax = plt.subplots()
    ax.plot([-5, 5], [0, 0], color='red', linestyle='dashed')
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel('$B(z)$', fontsize=14)
    ax.set_title('Lens veld', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/Actual_B_field.png', dpi=DPI)
    print(f'Saved Actual B field lens plot')

def plot_setup_1_report():
    R_1 = 1.5
    R_2 = 5
    R_1_magnet=6
    R_2_magnet=7
    d = 5
    d_magnet=4
    B_r_magnet_theoretical=1.37
    leak_factor=1
    B_r_magnet=B_r_magnet_theoretical*leak_factor
    T = 30*10**3

    lens = Lens(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T, setup_length=50)
    lens.setup_parameters(0, 1, 25)
    lens.display_properties(output_path='report/Images/setup_1.png', dpi=DPI)
    print(lens.B_r_yoke)
    print("Saved setup plot")

def plot_setup_2_report():
    R_1 = 1.5
    R_2 = 5
    R_1_magnet=6
    R_2_magnet=7
    d = 5
    d_magnet=4
    B_r_magnet_theoretical=1.37
    leak_factor=1
    B_r_magnet=B_r_magnet_theoretical*leak_factor
    T = 30*10**3

    lens = Lens(R_1, R_2, R_1_magnet, R_2_magnet, d, d_magnet, B_r_magnet, B_r_magnet_theoretical, T, setup_length=130)
    lens.setup_parameters(10, 1, 23.3)
    initial_values = np.array([[1, 0],
                      [1, -0.070160398],
                      [1, -0.41]])
    ax = lens.plot_setup(initial_values, report=True)
    ax.set_xlabel("$z$ (mm)", fontsize=14)
    ax.set_ylabel("$r$ (mm)", fontsize=14)
    ax.set_title("Stralendiagram", fontsize=16)
    ax.annotate('', xy=(10, 1), xytext=(10, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
    ax.text(8, 0.5, 'Object', ha='right', va='center', fontsize=12)
    ax.annotate('', xy=(91.4169, -5), xytext=(91.4169, 0),
            arrowprops=dict(arrowstyle='->', color='black', lw=3.5))
    ax.text(90, -2.5, 'Beeld', ha='right', va='center', fontsize=12)
    plt.savefig('report/Images/setup_2.png', dpi=DPI)
    print("Saved setup plot")

def main():
    plot_B_field_ring_report()
    plot_B_field_lens_report()
    plot_B_field_SIMION_comparison()
    plot_glasers_field()
    plot_constant_field()
    plot_actual_field()
    plot_setup_1_report()
    plot_setup_2_report()
    # variable_plots()


if __name__ == "__main__":
    main()