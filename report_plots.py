import numpy as np
import matplotlib.pyplot as plt
from sim import B_field_z

plt.rcParams['text.usetex'] = True

def B_field_ring(z, R_1, R_2):
    return (z)/2*(1/np.sqrt((z)**2+(R_1)**2)-1/np.sqrt((z)**2+(R_2)**2))

def B_field_lens(z, R_1, R_2, d):
    return B_field_ring(z+d/2, R_1, R_2)-B_field_ring(z-d/2, R_1, R_2)

def plot_B_field_ring_report():
    z_eval = np.linspace(-10, 10, 1000)
    B_eval = B_field_ring(z_eval, 1, 2)
    _, ax = plt.subplots()
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel(r'$B_{ring}(z)$ (eenheid $\mu_0\sigma_M$)', fontsize=14)
    ax.set_title('$B_{ring}(z)$', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/B_field_1_annulus.png')
    print(f'Saved B field ring plot')

def plot_B_field_lens_report():
    z_eval = np.linspace(-15, 15, 1500)
    B_eval = B_field_lens(z_eval, 1, 2, 2)
    B_max = np.max(B_eval)
    _, ax = plt.subplots()
    ax.plot(z_eval, B_eval)
    ax.grid('on')
    ax.set_xlabel('$z$', fontsize=14)
    ax.set_ylabel(r'$B(z)$ (eenheid $\mu_0\sigma_M$)', fontsize=14)
    ax.set_ylim(-(B_max+0.05), B_max+0.05)
    ax.set_title('$B(z)$', fontsize=16)
    plt.tight_layout()
    plt.savefig('report/Images/B_field_2_annula.png')
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
    plt.tight_layout()
    plt.savefig('report/Images/B_field_SIMION.png')
    print(f'Saved B field SIMION comparison plot')

def main():
    plot_B_field_ring_report()
    plot_B_field_lens_report()
    plot_B_field_SIMION_comparison()


if __name__ == "__main__":
    main()