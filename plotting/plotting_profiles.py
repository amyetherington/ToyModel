import matplotlib.pyplot as plt
import numpy as np
import lens1d as l1d
from astropy import cosmology


cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 1000, 0.001)

DM_Hilb = l1d.NFWHilbert(
    mass_at_200=1.27e13, redshift_lens=0.351, redshift_source=1.071
)
Hernquist = l1d.Hernquist(
    mass=10 ** 11.21, effective_radius=5.68, redshift_lens=0.351, redshift_source=1.071
)
Hernquist_2 = l1d.CombinedProfile(profiles=[Hernquist])
total = l1d.CombinedProfile(profiles=[Hernquist, DM_Hilb])
kappa_total = total.convergence_from_radii(radii=radii)
alpha = Hernquist.deflections_from_radii(radii=radii)

d_alpha = np.gradient(alpha, radii[:])
dd_alpha = np.gradient(d_alpha, radii[:])

mask_einstein_radius = total.mask_radial_range_from_radii(
    lower_bound=0.9, upper_bound=1.0, radii=radii
)

fit = l1d.PowerLawFit(profile=total, mask=None, radii=radii)
fit_ein = l1d.PowerLawFit(profile=total, mask=mask_einstein_radius, radii=radii)
print("concentration:", DM_Hilb.concentration)
print(
    "3D mass enclosed in Reff:",
    total.three_dimensional_mass_enclosed_within_effective_radius,
)
print(
    "2D mass enclosed in Reff:",
    total.two_dimensional_mass_enclosed_within_effective_radius,
)
print("Reff:", total.effective_radius)
print("Rein:", total.einstein_radius_in_kpc_from_radii(radii=radii))
print(
    "Rein via regression to kappa:",
    fit.einstein_radius_with_error(
    )[0],
)
print("Mein:", total.einstein_mass_in_solar_masses_from_radii(radii=radii))
print(
    "slope via regression to alpha:",
    fit.slope_via_deflections(
    ),
)
print(
    "slope via regression to kappa:",
    fit.slope_with_error(
    )[0],
)
print(
    "slope via regression to alpha around rein:",
    fit.slope_via_deflections(
    ),
)
print(
    "slope via regression to kappa around rein:",
    fit_ein.slope_with_error(
    )[0],
)
print("slope via lensing:", fit.slope_via_lensing())
print(
    "slope via lensing and dynamics:",
    fit.slope_and_normalisation_via_dynamics()[1],
)

print("rsquared:", fit.r_squared_value())

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
