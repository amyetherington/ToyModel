import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from lens1d import combined_profiles as cp
from lens1d import one_d_profiles as profile
from astropy import cosmology


cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.2, 20, 0.01)

DM_Hilb = profile.NFWHilbert(
    mass_at_200=1.1e13, redshift_lens=0.351, redshift_source=1.071
)
Hernquist = profile.Hernquist(
    mass=10 ** 11.21, effective_radius=5.68, redshift_lens=0.351, redshift_source=1.071
)
Hernquist_2 = cp.CombinedProfile(profiles=[Hernquist])
total = cp.CombinedProfile(profiles=[Hernquist, DM_Hilb])
kappa_total = total.convergence_from_radii(radii=radii)

no_mask = total.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)
mask_einstein_radius = total.mask_radial_range_from_radii(
    lower_bound=0.8, upper_bound=1.2, radii=radii
)


kappa_baryons = Hernquist.convergence_from_radii(radii=radii)
kappa_DM = DM_Hilb.convergence_from_radii(radii=radii)


kappa_best_fit = total.inferred_convergence_from_mask_and_radii(
    radii=radii, mask=no_mask
)
kappa_best_fit_ein = total.inferred_convergence_from_mask_and_radii(
    radii=radii, mask=mask_einstein_radius
)
kappa_dynamics = total.power_law_convergence_via_dynamics(radii=radii)
kappa_lensing = total.power_law_convergence_via_lensing(radii=radii)

einstein_radius = total.einstein_radius_in_kpc_from_radii(radii=radii)
einstein_radius_dyn = total.einstein_radius_via_dynamics(radii=radii)

effective_radius = total.effective_radius

d_A = cosmo.angular_diameter_distance(z=0.351).to("kpc").value

einstein_radius_kpc = (einstein_radius_dyn / d_A) / np.pi / 180 / 3600

coeffs = total.inferred_convergence_coefficients_from_mask_and_radii(
    mask=no_mask, radii=radii
)

print(total.dark_matter_mass_fraction_within_effective_radius)
print(einstein_radius)

fig1 = plt.figure(1)
plt.loglog(radii, kappa_baryons, "--", label="baryons", alpha=0.5, color="lightcoral")
plt.loglog(radii, kappa_DM, "--", label="dark matter", alpha=0.5, color="lightskyblue")
plt.axvline(x=einstein_radius, color="grey", alpha=0.5)
plt.axvline(x=effective_radius, color="darkslategrey", alpha=0.5)
plt.loglog(radii, kappa_best_fit, "-.", label="best fit kappa", color="navy", alpha=0.8)
plt.loglog(
    radii, kappa_lensing, "-.", label="kappa via lensing", color="blue", alpha=0.8
)
plt.loglog(radii, kappa_dynamics, "-.", label="kappa via dyn", color="cyan", alpha=0.8)
plt.loglog(radii, kappa_total, label="total", color="plum")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Convergence", fontsize=14)
plt.savefig(
    fig_path + "kappa_v_r_slacs0330", bbox_inches="tight", dpi=300, transparent=True
)
# plt.close()


plt.show()
