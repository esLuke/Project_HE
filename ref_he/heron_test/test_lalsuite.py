from heron.models.lalsuite import HyperbolicTD
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times New Roman']})

waveform = HyperbolicTD()
f, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
eccentricities = [1.5, 1.3, 1.1, 1.05]
labels = ['Eccentricity = 1.5', 'Eccentricity = 1.3', 'Eccentricity = 1.1', 'Eccentricity = 1.05']

for ax, ecc, label in zip(axes, eccentricities, labels):
    data = waveform.time_domain(parameters={"m1": 10 * u.solMass, "m2": 10 * u.solMass, "hyperbolic_eccentricity": ecc})
    ax.plot(data['plus'], label=label)
    ax.legend()
    ax.set_title(label)
    ax.set_xlabel('l')
    ax.set_ylabel(r'$h_{+}|_{Q}(l)$')

plt.tight_layout(pad = 3.5)
f.suptitle('Hyperbolic Waveform', fontsize=16)
f.savefig("waveform.png")
plt.show()

'''

data1 = waveform.time_domain(parameters={"m1": 10 * u.solMass, "m2": 10*u.solMass, "hyperbolic_eccentricity": 1.3})
data2 = waveform.time_domain(parameters={"m1": 10 * u.solMass, "m2": 10*u.solMass, "hyperbolic_eccentricity": 1.2})
data3 = waveform.time_domain(parameters={"m1": 10 * u.solMass, "m2": 10*u.solMass, "hyperbolic_eccentricity": 1.1})
data4 = waveform.time_domain(parameters={"m1": 10 * u.solMass, "m2": 10*u.solMass, "hyperbolic_eccentricity": 1.05})

f, ax = plt.subplots(1,1)

ax.plot(data['plus'])

f.savefig("waveform.png")
'''
