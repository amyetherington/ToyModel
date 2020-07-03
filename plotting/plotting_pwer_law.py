import matplotlib.pyplot as plt
import numpy as np
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profile
from astropy import cosmology


cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 1000, 0.001)

effective_radii = np.arange(0.1, 6.7, 0.1)

power_law = profile.SphericalPowerLaw(einstein_radius=6.7, slope=1.8, z_l=0.3, z_s=0.8, effective_radius=1.3)
power_law_total = cp.CombinedProfile(profiles=[power_law])
kappa_power_law_total = power_law_total.convergence_from_radii(radii=radii)
surface_mass_density = power_law_total.surface_mass_density_from_radii(radii=radii)
density = power_law_total.density_from_radii(radii=radii)

kappa_via_surface_mass = surface_mass_density / power_law_total.critical_surface_density_of_lens

no_mask = power_law_total.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)
mask_einstein_radius = power_law_total.mask_radial_range_from_radii(lower_bound=0.9, upper_bound=1.0, radii=radii)

print("3D mass enclosed in Reff:", power_law_total.three_dimensional_mass_enclosed_within_effective_radius)
print("2D mass enclosed in Reff:", power_law_total.two_dimensional_mass_enclosed_within_effective_radius)
print("Rein:", power_law_total.einstein_radius_in_kpc_from_radii(radii=radii))
print("Rein via regression to kappa:", power_law_total.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(mask=no_mask, radii=radii)[0])
print("Mein:", power_law_total.einstein_mass_in_solar_masses_from_radii(radii=radii))
print("slope via regression to alpha:", power_law_total.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(mask=no_mask, radii=radii))
print("slope via regression to kappa:",power_law_total.best_fit_power_law_slope_with_error_from_mask_and_radii(mask=no_mask, radii=radii)[0])
print("slope via regression to alpha around rein:", power_law_total.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(mask=mask_einstein_radius, radii=radii))
print("slope via regression to kappa around rein:", power_law_total.best_fit_power_law_slope_with_error_from_mask_and_radii(mask=mask_einstein_radius, radii=radii)[0])
print("slope via lensing:", power_law_total.slope_via_lensing(radii=radii))
print("slope via lensing and dynamics:", power_law_total.slope_and_normalisation_via_dynamics(radii=radii)[1])
print("slope via lensing and dynamics 2d:", power_law_total.slope_and_normalisation_via_2d_mass_and_einstain_mass(radii=radii)[1])

slope_via_lensing = []
slope_via_dynamics_2d = []
slope_via_dynamics = []

for i in range(len(effective_radii)):
    power_law = profile.SphericalPowerLaw(einstein_radius=6.7, slope=1.8, z_l=0.3, z_s=0.8, effective_radius=effective_radii[i])
    power_law_total = cp.CombinedProfile(profiles=[power_law])
    dynamics = power_law_total.slope_and_normalisation_via_dynamics(radii=radii)[1]
    dynamics_2 = power_law_total.slope_and_normalisation_via_2d_mass_and_einstain_mass(radii=radii)[1]
    lensing = power_law_total.slope_via_lensing(radii=radii)
    slope_via_lensing.append(lensing)
    slope_via_dynamics_2d.append(dynamics_2)
    slope_via_dynamics.append(dynamics)


fig1 = plt.figure(1)
plt.loglog(radii, kappa_power_law_total, label="kappa")
plt.loglog(radii, surface_mass_density, label="surface mass density")
plt.loglog(radii, kappa_via_surface_mass, label="kappa via surface mass density")
plt.loglog(radii, density, label="density")
plt.legend()

fig2 = plt.figure(2)
plt.plot(effective_radii, slope_via_lensing, label="lensing")
plt.plot(effective_radii, slope_via_dynamics, label="dynamics")
plt.plot(effective_radii, slope_via_dynamics_2d, label="2d")
plt.xlabel("effective radius (kpc)")
plt.ylabel("slope measured")
plt.legend()

plt.show()
