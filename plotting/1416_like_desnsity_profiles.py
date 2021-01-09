import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import lens1d as l1d

fig_path = "/Users/dgmt59/Documents/Plots/one_d_slacs/"

baryons = l1d.Hernquist(
    mass=10 ** 11.4,
    effective_radius=5.4,
    redshift_lens=0.232,
    redshift_source=1.3
)

baryons_mass = l1d.Hernquist(
    mass=10 ** 12.9,
    effective_radius=1.1,
    redshift_lens=0.3,
    redshift_source=0.8
)

DM = l1d.NFWHilbert(
    mass_at_200=2.31e13,
    redshift_lens=0.3,
    redshift_source=0.8
)

isothermal = l1d.SphericalPowerLaw(
    einstein_radius=1.4,
    slope=2.0,
    redshift_lens=0.3,
    redshift_source=0.8
)
no_bh = l1d.SphericalPowerLaw(
    einstein_radius=1.4,
    slope=3.0,
    redshift_lens=0.3,
    redshift_source=0.8
)

DM_cored = l1d.generalisedNFW(
    mass_at_200=1.21e13,
    beta=1.9,
    redshift_lens=0.3,
    redshift_source=0.8
)


print(DM.scale_radius)

radii = np.arange(0.4, 100, 0.001)

true_profile = l1d.CombinedProfile(profiles=[baryons, DM])
true_profile_cored = l1d.CombinedProfile(profiles=[baryons_mass, DM_cored])

einstein_radius = true_profile.einstein_radius_in_kpc_from_radii(radii=radii)


rho_isothermal = isothermal.density_from_radii(radii=radii)
rho_no_bh = no_bh.density_from_radii(radii=radii)*2
rho_baryons = baryons.density_from_radii(radii=radii)
rho_baryons_2 = baryons_mass.density_from_radii(radii=radii)*0.05
rho_DM = DM.density_from_radii(radii=radii)
rho_total = true_profile.density_from_radii(radii=radii)
rho_DM_cored = DM_cored.density_from_radii(radii=radii)*0.05
total_DM_cored = true_profile_cored.density_from_radii(radii=radii)*0.05




fig6 = plt.figure(6)
plt.loglog(
    radii, rho_baryons, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, rho_DM, "--", label="dark matter", color="lightskyblue"
)
plt.loglog(radii, rho_total, label="total - stronger winds", color="dodgerblue")
plt.loglog(radii, rho_isothermal, "-.", label=r"isothermal ($\gamma$ = 2)", color="grey")
plt.legend()
plt.ylim(ymin=1e3, ymax=1e11)
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
#plt.savefig(fig_path + "density_stronger_winds.png", bbox_inches="tight", dpi=300)
plt.show()



fig3 = plt.figure(3)
plt.loglog(
    radii, rho_baryons_2, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, rho_DM_cored, "--", label="dark matter", color="lightskyblue"
)
plt.loglog(
    radii, total_DM_cored, label="total - no BH",  color="darkgreen"
)
plt.loglog(radii, rho_isothermal, "-.", label="isothermal ($\gamma$ = 2)", color="grey")
#plt.loglog(radii, rho_no_bh, "-.", label="$\gamma$ = 3", color="grey")
plt.legend()
plt.ylim(ymin=1e3, ymax=1e11)
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
#plt.savefig(fig_path + "density_gamma_3.png", bbox_inches="tight", dpi=300)
plt.show()


