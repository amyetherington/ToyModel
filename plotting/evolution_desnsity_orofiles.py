import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import lens1d as l1d

fig_path = "/Users/dgmt59/Documents/Plots/one_d_slacs/"

baryons = l1d.Hernquist(
    mass=10 ** 11.53,
    effective_radius=8.4,
    redshift_lens=0.232,
    redshift_source=1.3
)

baryons_mass = l1d.Hernquist(
    mass=10 ** 11.53,
    effective_radius=8.4,
    redshift_lens=0.232,
    redshift_source=1.3
)

DM = l1d.NFWHilbert(
    mass_at_200=1.5e12,
    redshift_lens=0.232,
    redshift_source=1.3
)

isothermal = l1d.SphericalPowerLaw(
    einstein_radius=1.5,
    slope=2.0,
    redshift_lens=0.132,
    redshift_source=1.3
)

DM_cored = l1d.generalisedNFW(
    mass_at_200=1.5e12,
    beta=1.6,
    redshift_lens=0.232,
    redshift_source=1.3
)

print(DM.scale_radius)

radii = np.arange(0.2, 200, 0.001)

true_profile = l1d.CombinedProfile(profiles=[baryons, DM])
true_profile_cored = l1d.CombinedProfile(profiles=[baryons_mass, DM_cored])

einstein_radius = true_profile.einstein_radius_in_kpc_from_radii(radii=radii)


rho_isothermal = isothermal.density_from_radii(radii=radii)*0.3
rho_baryons = baryons.density_from_radii(radii=radii)
rho_DM = DM.density_from_radii(radii=radii)
rho_total = true_profile.density_from_radii(radii=radii)
rho_DM_cored = DM_cored.density_from_radii(radii=radii)
total_DM_cored = true_profile_cored.density_from_radii(radii=radii)

kappa_isothermal = isothermal.convergence_from_radii(radii=radii)*0.7
kappa_baryons = baryons.convergence_from_radii(radii=radii)
kappa_DM = DM.convergence_from_radii(radii=radii)
kappa_total = true_profile.convergence_from_radii(radii=radii)
kappa_DM_cored = DM_cored.convergence_from_radii(radii=radii)
kappa_total_cored = true_profile_cored.convergence_from_radii(radii=radii)

sigma_isothermal = isothermal.surface_mass_density_from_radii(radii=radii)
sigma_baryons = baryons.surface_mass_density_from_radii(radii=radii)
sigma_DM = DM.surface_mass_density_from_radii(radii=radii)
sigma_total = true_profile.surface_mass_density_from_radii(radii=radii)



fig3 = plt.figure(3)
plt.loglog(
    radii, rho_baryons, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, rho_DM, "--", label="dark matter", color="lightskyblue"
)
plt.loglog(
    radii, rho_DM_cored, "--", label="dark matter cored", color="skyblue"
)
plt.loglog(
    radii, total_DM_cored, label="total matter cored",  color="teal"
)
plt.loglog(radii, rho_total, label="total", color="purple")
plt.loglog(radii, rho_isothermal, "-.", label="isothermal", color="grey")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
plt.savefig(fig_path + "density.png", bbox_inches="tight", dpi=300, transparent=True)

fig4 = plt.figure(4)
plt.loglog(
    radii,  kappa_baryons, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, kappa_DM, "--", label="dark matter", color="lightskyblue"
)
plt.loglog(
    radii, kappa_DM_cored, "--", label="dark matter cored", color="skyblue"
)
plt.loglog(
    radii, kappa_total_cored, label="total matter cored",  color="teal"
)
plt.loglog(radii, rho_total, label="total", color="purple")
plt.loglog(radii, rho_isothermal, "-.", label="isothermal", color="grey")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
plt.savefig(fig_path + "kappa.png", bbox_inches="tight", dpi=300, transparent=True)

fig5 = plt.figure(5)
plt.loglog(
    radii, rho_baryons, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, rho_DM, "--", label="dark matter", color="lightskyblue"
)
plt.loglog(radii, rho_total, label="total", color="purple")
plt.loglog(radii, rho_isothermal, "-.", label="isothermal", color="grey")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
plt.savefig(fig_path + "density_not_cored.png", bbox_inches="tight", dpi=300, transparent=True)

plt.show()