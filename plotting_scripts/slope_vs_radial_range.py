import matplotlib.pyplot as plt
import numpy as np
import lens1d as l1d
import os
import pandas as pd

fig_path = "/Users/dgmt59/Documents/Plots/one_d_slacs/"

slacs_path = "{}/../../autolens_slacs_pre_v_1/dataset/slacs_data_table.xlsx".format(
    os.path.dirname(os.path.realpath(__file__))
)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name

lens_name = "slacs0728+3835"

radii = np.arange(0.01, 50, 0.001)

DM_Hilb = l1d.NFWHilbert(
        mass_at_200=slacs["M200"][lens_name],
        redshift_lens=slacs["z_lens"][lens_name],
        redshift_source=slacs["z_source"][lens_name],
    )

Hernquist = l1d.Hernquist(
        mass=10 ** slacs["log[M*/M]_chab"][lens_name],
        effective_radius=slacs["R_eff"][lens_name],
        redshift_lens=slacs["z_lens"][lens_name],
        redshift_source=slacs["z_source"][lens_name],
    )

true_profile = l1d.CombinedProfile(profiles=[Hernquist, DM_Hilb])

kappa_baryons = Hernquist.convergence_from_radii(radii=radii)
kappa_DM = DM_Hilb.convergence_from_radii(radii=radii)
einstein_radius = true_profile.einstein_radius_in_kpc_from_radii(radii=radii)
effective_radius = true_profile.effective_radius
kappa_total = true_profile.convergence_from_radii(radii=radii)

fit = l1d.PowerLawFit(profile=true_profile, mask=None, radii=radii)

lens_slope = fit.slope_via_lensing()
dyn_slope = fit.slope_and_normalisation_via_dynamics()[1]
kappa_lensing = fit.convergence_via_lensing()
kappa_dynamics = fit.convergence_via_dynamics()
kappa_best_fit = fit.convergence()

radial_range = np.arange(0.2, 20, 0.1)

slope_ein = []
slope_eff = []

for i in range(len(radial_range)):
    mask_ein = true_profile.mask_einstein_radius_from_radii(radii=radii, width=radial_range[i])
    mask_eff = true_profile.mask_effective_radius_from_radii(radii=radii, width=radial_range[i])

    fit_ein = l1d.PowerLawFit(profile=true_profile, mask=mask_ein, radii=radii)
    fit_eff = l1d.PowerLawFit(profile=true_profile, mask=mask_eff, radii=radii)

    slope_ein.append(fit_ein.slope_with_error()[0])
    slope_eff.append(fit_eff.slope_with_error()[0])




fig1 = plt.figure(1)
plt.loglog(
    radii, kappa_baryons, "--", label="baryons", alpha=0.5, color="lightcoral"
)
plt.loglog(
    radii, kappa_DM, "--", label="dark matter", alpha=0.5, color="lightskyblue"
)
plt.axvline(x=einstein_radius, color="grey", alpha=0.5)
plt.axvline(x=effective_radius, color="darkslategrey", alpha=0.5)
plt.loglog(
    radii, kappa_best_fit, "-.", label="best fit kappa", color="navy", alpha=0.8
)
plt.loglog(
    radii, kappa_lensing, "-.", label="kappa via lensing", color="blue", alpha=0.8
)
plt.loglog(
    radii, kappa_dynamics, "-.", label="kappa via dyn", color="cyan", alpha=0.8
)
plt.loglog(radii, kappa_total, label="total", color="plum")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Convergence", fontsize=14)

#plt.savefig(fig_path + lens_name + "_radial_range_test.png", bbox_inches="tight", dpi=300, transparent=True)

fig2 = plt.figure(2)
plt.axhline(lens_slope, color="grey", ls="--")
plt.axhline(dyn_slope, color="darkslategrey", ls="-.")
plt.plot(radial_range, slope_ein, label="mask around einstein radius")
plt.plot(radial_range, slope_eff, label="mask around effective radius")
plt.xlabel("Radial Range (kpc)", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(fig_path + lens_name + "slope_v_radial_range.png", bbox_inches="tight", dpi=300, transparent=True)
plt.show()


