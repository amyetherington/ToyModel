import one_d_profiles as profile
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math


fig_path = "/Users/dgmt59/Documents/Plots/slope_v_radial_range/"

M_o = 1.989e30

radii = np.arange(0.2, 10, 0.0015)

baryons = profile.SphericalPowerLaw(slope=2.2, einstein_radius=1.2)
DM = profile.NFW(kappa_s=0.22, scale_radius=4.6)
true_profile = profile.CombinedProfile(mass_profiles=[baryons, DM])
mask = true_profile.mask_radial_range_from_radii(
    radii=radii, lower_bound=0.8, upper_bound=1.2
)
mask_inner_image = true_profile.mask_two_radial_ranges_from_radii(
    radii=radii,
    lower_bound_1=0.2,
    upper_bound_1=0.3,
    lower_bound_2=0.8,
    upper_bound_2=1.2,
)

kappa_baryons = baryons.convergence_from_radii(radii=radii)
kappa_DM = DM.convergence_from_radii(radii=radii)
kappa_true = true_profile.convergence_from_radii(radii=radii)

alpha_baryons = baryons.deflection_angles_from_radii(radii=radii)
alpha_DM = DM.deflection_angles_from_radii(radii=radii)
alpha_true = true_profile.deflection_angles_from_radii(radii=radii)


print(DM.r_200_in_kpc_from_redshifts(z_s=0.8, z_l=0.3))
print(DM.m_200_in_solar_masses_fromn_redshifts(z_s=0.8, z_l=0.3))
print(DM.concentration_from_redshifts(z_s=0.8, z_l=0.3))

kappa_best_fit = true_profile.best_fit_power_law_convergence_from_mask_and_radii(
    radii=radii, mask=mask
)
kappa_best_fit_inner_image = true_profile.best_fit_power_law_convergence_from_mask_and_radii(
    radii=radii, mask=mask_inner_image
)
alpha_best_fit = true_profile.best_fit_power_law_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask
)
alpha_best_fit_inner_image = true_profile.best_fit_power_law_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask_inner_image
)

kappa_best_fit_from_alpha = true_profile.best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask
)
kappa_best_fit_from_alpha_inner_image = true_profile.best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask_inner_image
)

kappa_best_fit_slope = true_profile.best_fit_power_law_slope_with_error_from_mask_and_radii(
    radii=radii, mask=mask
)
kappa_best_fit_slope_inner_image = true_profile.best_fit_power_law_slope_with_error_from_mask_and_radii(
    radii=radii, mask=mask_inner_image
)
kappa_best_fit_slope_from_alpha = true_profile.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask
)
kappa_best_fit_slope_from_alpha_inner_image = true_profile.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
    radii=radii, mask=mask_inner_image
)
kappa_best_fit_einstein_radius_alpha = true_profile.best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
    radii=radii, mask=mask
)
einstein_radius = true_profile.einstein_radius_in_arcseconds_from_radii(radii=radii)

print(kappa_best_fit_slope)
print(kappa_best_fit_slope_from_alpha)
print(kappa_best_fit_slope_inner_image)
print(kappa_best_fit_slope_from_alpha_inner_image)


fig1 = plt.figure(1)
plt.loglog(radii, kappa_baryons, "--", label="baryons", alpha=0.5)
plt.loglog(radii, kappa_DM, "--", label="dark matter", alpha=0.5)
plt.loglog(radii, kappa_true, "--", label="total")
plt.loglog(radii, kappa_best_fit_from_alpha, label="best fit alpha")
plt.loglog(
    radii, kappa_best_fit_from_alpha_inner_image, label="best fit alpha with image"
)
plt.loglog(radii, kappa_best_fit, label="best fit kappa")
plt.loglog(radii, kappa_best_fit_inner_image, label="best fit kappa with image")
plt.legend()
plt.xlabel("Radius (arcseconds)", fontsize=14)
plt.ylabel("Convergence", fontsize=14)
plt.savefig(fig_path + "kappa_v_r", bbox_inches="tight", dpi=300, transparent=True)
# plt.close()

fig2 = plt.figure(2)
plt.loglog(radii, alpha_baryons, "--", label="baryons", alpha=0.5)
plt.loglog(radii, alpha_DM, "--", label="dark matter", alpha=0.5)
plt.loglog(radii, alpha_true, "--", label="total")
plt.loglog(radii, alpha_best_fit, label="best fit")
plt.loglog(radii, alpha_best_fit_inner_image, label="best fit inner image")
plt.xlabel("Radius (arcseconds)", fontsize=14)
plt.ylabel("Deflection Angles", fontsize=14)
plt.savefig(fig_path + "alpha_v_r", bbox_inches="tight", dpi=300, transparent=True)
# plt.close()


plt.show()
