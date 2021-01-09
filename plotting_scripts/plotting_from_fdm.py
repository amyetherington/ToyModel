import matplotlib.pyplot as plt
import numpy as np
import lens1d as l1d
from astropy import cosmology


cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 50, 0.001)

hernquist = l1d.Hernquist(
    mass=10 ** 11.21, effective_radius=5.68, redshift_lens=0.351, redshift_source=1.071
)
total = l1d.CombinedProfile.from_hernquist_and_dark_matter_fraction_within_effective_radius(hernquist=hernquist,
                                                                                            dark_matter_fraction=0.4)

kappa_total = total.convergence_from_radii(radii=radii)


fig1 = plt.figure(1)
plt.loglog(radii, kappa_total, label="total")
plt.legend()

plt.show()
