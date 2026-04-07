import numpy as np
import matplotlib.pyplot as plt
from sim import B_field_z




### Numeric Field###
numericpotential1=np.loadtxt("numericpotential1.txt")[:, 4]
numericpotential2=np.loadtxt("numericpotential2.txt")[:, 4]
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

fig, ax=plt.subplots()
ax.plot(z_analytic*100, analyticalfield/np.max(analyticalfield), color='r', label='Analytisch')
ax.plot(z_numeric*100, numericfield/np.max(numericfield), color='b', label='Numeriek')
ax.grid()
ax.legend()
plt.show()


