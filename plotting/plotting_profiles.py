import matplotlib.pyplot as plt
import numpy as np
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profile
from astropy import cosmology

cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 1000, 0.001)

DM_Hilb = profile.NFW_Hilbert(m200=1.2e12, z_l=0.3, z_s=0.9)
Hernquist = profile.Hernquist(mass=3.4e11, effective_radius=3.2, z_l=0.3, z_s=0.9)
Hernquist_2 = cp.CombinedProfile(profiles=[Hernquist])
total = cp.CombinedProfile(profiles=[Hernquist, DM_Hilb])
kappa_total = total.convergence_from_radii(radii=radii)
alpha = Hernquist.deflection_angles_from_radii(radii=radii)

d_alpha = np.gradient(alpha, radii[:])
dd_alpha = np.gradient(d_alpha, radii[:])

fig1 = plt.figure(1)
plt.loglog(radii, alpha, label="alpha")
plt.loglog(radii, d_alpha, label="d_alpha")
plt.loglog(radii, dd_alpha, label="d_alpha")
plt.show()

mask = total.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)

print(DM_Hilb.concentration)
print(total.two_dimensional_mass_enclosed_within_effective_radius)
print(total.effective_radius)
print(total.einstein_radius_in_kpc_from_radii(radii=radii))
print(total.einstein_mass_in_solar_masses_from_radii(radii=radii))
print(total.slope_via_lensing(radii=radii))
print(total.slope_via_dynamics(radii=radii))

rho_NFW = DM_Hilb.surface_mass_density_from_radii(radii=radii)
kappa_NFW = DM_Hilb.convergence_from_radii(radii=radii)
kappa_via_sigma_NFW = rho_NFW / DM_Hilb.critical_surface_density_of_lens

rho_H = Hernquist.surface_mass_density_from_radii(radii=radii)
kappa_H = Hernquist.convergence_from_radii(radii=radii)
kappa_via_sigma_H = rho_H / Hernquist.critical_surface_density_of_lens


fig1 = plt.figure(1)
plt.loglog(radii, kappa_H, label="Hernquist")
plt.loglog(radii, kappa_NFW, label="NFW")
plt.loglog(radii, kappa_total, label="total")
plt.legend()

plt.show()
