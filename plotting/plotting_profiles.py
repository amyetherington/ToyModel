import matplotlib.pyplot as plt
import numpy as np
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profile
from astropy import cosmology


cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 1000, 0.001)

DM_Hilb = profile.NFW_Hilbert(m200=1.27e13, z_l=0.351, z_s=1.071)
Hernquist = profile.Hernquist(mass=10**11.21, effective_radius=5.68, z_l=0.351, z_s=1.071)
Hernquist_2 = cp.CombinedProfile(profiles=[Hernquist])
total = cp.CombinedProfile(profiles=[Hernquist, DM_Hilb])
kappa_total = total.convergence_from_radii(radii=radii)
alpha = Hernquist.deflection_angles_from_radii(radii=radii)

d_alpha = np.gradient(alpha, radii[:])
dd_alpha = np.gradient(d_alpha, radii[:])

no_mask = total.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)
mask_einstein_radius = total.mask_radial_range_from_radii(lower_bound=0.9, upper_bound=1.0, radii=radii)

print("concentration:", DM_Hilb.concentration)
print("3D mass enclosed in Reff:", total.three_dimensional_mass_enclosed_within_effective_radius)
print("2D mass enclosed in Reff:", total.two_dimensional_mass_enclosed_within_effective_radius)
print("Reff:", total.effective_radius)
print("Rein:", total.einstein_radius_in_kpc_from_radii(radii=radii))
print("Rein via regression to kappa:", total.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(mask=no_mask, radii=radii)[0])
print("Mein:", total.einstein_mass_in_solar_masses_from_radii(radii=radii))
print("slope via regression to alpha:", total.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(mask=no_mask, radii=radii))
print("slope via regression to kappa:",total.best_fit_power_law_slope_with_error_from_mask_and_radii(mask=no_mask, radii=radii)[0])
print("slope via regression to alpha around rein:", total.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(mask=mask_einstein_radius, radii=radii))
print("slope via regression to kappa around rein:", total.best_fit_power_law_slope_with_error_from_mask_and_radii(mask=mask_einstein_radius, radii=radii)[0])
print("slope via lensing:", total.slope_via_lensing(radii=radii))
print("slope via lensing and dynamics:", total.slope_and_normalisation_via_dynamics(radii=radii)[1])

print("rsquared:",total.power_law_r_squared_value(mask=no_mask, radii=radii))

rho_NFW = DM_Hilb.surface_mass_density_from_radii(radii=radii)
kappa_NFW = DM_Hilb.convergence_from_radii(radii=radii)
kappa_via_sigma_NFW = rho_NFW / DM_Hilb.critical_surface_density_of_lens

rho_H = Hernquist.surface_mass_density_from_radii(radii=radii)
kappa_H = Hernquist.convergence_from_radii(radii=radii)
kappa_via_sigma_H = rho_H / Hernquist.critical_surface_density_of_lens
total_rho = total.surface_mass_density_from_radii(radii=radii)


fig1 = plt.figure(1)
plt.loglog(radii, kappa_H, label="Hernquist")
plt.loglog(radii, kappa_NFW, label="NFW")
plt.loglog(radii, kappa_total, label="total")
plt.legend()

fig2 = plt.figure(2)
plt.loglog(radii, rho_H, label="Hernquist")
plt.loglog(radii, rho_NFW, label="NFW")
plt.loglog(radii, total_rho, label="total")
plt.legend()

plt.show()
