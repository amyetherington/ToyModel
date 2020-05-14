import matplotlib.pyplot as plt
import numpy as np
import one_d_profiles as profile

plots_path = "/Users/dgmt59/Documents/Plots/slope_v_radial_range/"

radii = np.arange(0.2, 10, 0.0015)

baryons = profile.SphericalPowerLaw(slope=2.2, einstein_radius=1.2)
DM = profile.NFW(kappa_s=0.22, scale_radius=4.6)
true_profile = profile.CombinedProfile(mass_profiles=[baryons, DM])

upper_bound = np.arange(1.1, 1.9, 0.01)
lower_bound = np.flip(np.arange(0.1, 1, 0.01))

kappa_best_fit_slope_from_alpha = []
kappa_best_fit_slope = []
kappa_best_fit_einstein_mass_from_alpha = []
kappa_best_fit_einstein_mass = []
radial_range = []

fig_path = (
    plots_path
    + "slope_"
    + str(baryons.slope)
    + "__ein_"
    + str(baryons.einstein_radius)
    + "__scale_r_"
    + str(DM.scale_radius)
    + "__kappa_s_"
    + str(DM.kappa_s)
)

for i in range(len(upper_bound)):

    kappa_best_fit_slope.append(
        true_profile.best_fit_power_law_slope_with_error_between_radii_from_radii(
            radii=radii, lower_bound=lower_bound[i], upper_bound=upper_bound[i]
        )[0]
    )
    kappa_best_fit_slope_from_alpha.append(
        true_profile.best_fit_power_law_slope_via_deflection_angles_between_radii_from_radii(
            radii=radii, lower_bound=lower_bound[i], upper_bound=upper_bound[i]
        )
    )

    kappa_best_fit_einstein_mass.append(
        true_profile.best_fit_einstein_mass_in_solar_masses_between_radii_from_radii_and_redshifts(
            radii=radii,
            lower_bound=lower_bound[i],
            upper_bound=upper_bound[i],
            z_l=0.3,
            z_s=1.0,
        )[
            0
        ]
    )
    kappa_best_fit_einstein_mass_from_alpha.append(
        true_profile.best_fit_einstein_mass_in_solar_masses_via_deflection_angles_between_radii_from_radii_and_redshifts(
            radii=radii,
            lower_bound=lower_bound[i],
            upper_bound=upper_bound[i],
            z_l=0.3,
            z_s=1.0,
        )
    )

    radial_range.append(upper_bound[i] - lower_bound[i])

fig1 = plt.figure(1)
plt.plot(radial_range, kappa_best_fit_slope, label="best fit kappa")
plt.plot(radial_range, kappa_best_fit_slope_from_alpha, label="best fit alpha")
plt.xlabel("Radial Range (Fraction of Einstein Radius)", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(plots_path + "_slopes", bbox_inches="tight", dpi=300, transparent=True)


fig2 = plt.figure(2)
plt.plot(radial_range, kappa_best_fit_einstein_mass, label="best fit kappa")
plt.plot(radial_range, kappa_best_fit_einstein_mass_from_alpha, label="best fit alpha")
plt.xlabel("Radial Range (Fraction of Einstein Radius)", fontsize=14)
plt.ylabel("Einstein Mass (Solar Masses)", fontsize=14)
plt.legend()
plt.savefig(plots_path + "_masses", bbox_inches="tight", dpi=300, transparent=True)
plt.show()
