import matplotlib.pyplot as plt
import numpy as np
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profile
from astropy import cosmology



cosmo = cosmology.Planck15

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.01, 500, 0.01)

DM_Hilb = profile.NFW_Hilbert(m200=1.27e13, z_l=0.351, z_s=1.071)
Hernquist = profile.Hernquist(mass=10**11.21, effective_radius=5.68, z_l=0.351, z_s=1.071)
Hernquist_2 = cp.CombinedProfile(profiles=[Hernquist])
total = cp.CombinedProfile(profiles=[Hernquist, DM_Hilb])
kappa_total = total.convergence_from_radii(radii=radii)

no_mask = total.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)
mask_einstein_radius = total.mask_radial_range_from_radii(lower_bound=0.8, upper_bound=1.2, radii=radii)

kappa_baryons = Hernquist.convergence_from_radii(radii=radii)
kappa_DM = DM_Hilb.convergence_from_radii(radii=radii)


kappa_best_fit = total.best_fit_power_law_convergence_from_mask_and_radii(
    radii=radii, mask=no_mask
)
kappa_best_fit_ein = total.best_fit_power_law_convergence_from_mask_and_radii(
    radii=radii, mask=mask_einstein_radius
)

einstein_radius = total.einstein_radius_in_kpc_from_radii(radii=radii)
effective_radius = total.effective_radius

fig1 = plt.figure(1)
plt.loglog(radii, kappa_baryons, "--", label="baryons", alpha=0.5)
plt.loglog(radii, kappa_DM, "--", label="dark matter", alpha=0.5)
plt.loglog(radii, kappa_total, "--", label="total")
plt.loglog(radii, kappa_best_fit_ein, label="best fit ein")
plt.loglog(radii, kappa_best_fit, label="best fit kappa")
plt.axvline(x=einstein_radius, color='red')
plt.axvline(x=effective_radius, color='orange')
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel("Convergence", fontsize=14)
plt.savefig(fig_path + "kappa_v_r_slacs0330", bbox_inches="tight", dpi=300, transparent=True)
# plt.close()


plt.show()
