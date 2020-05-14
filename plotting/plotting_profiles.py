import matplotlib.pyplot as plt
import numpy as np
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profile

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.2, 1000, 0.002)

DM_Hilb = profile.NFW_Bartelmann(m200=2.5e12, concentration=3.4, z_l=0.6, z_s=0.8)
Hernquist = profile.Hernquist(mass=3.4e10, effective_radius=8.4, z_l=0.6, z_s=0.8)
Hernquist_2 = cp.CombinedProfile(profiles=[Hernquist])
total = cp.CombinedProfile(profiles=[Hernquist, DM_Hilb])
power_law = profile.SphericalPowerLaw(einstein_radius=1.2, slope=1.8, z_l=0.3, z_s=0.8)
kappa_total = total.convergence_from_radii(radii=radii)

print(DM_Hilb.r200)
print(DM_Hilb.r_s)
print(DM_Hilb.kappa_s)
print(DM_Hilb.critical_surface_density_of_lens)
print(Hernquist.einstein_radius_in_kpc_from_radii(radii=radii))
print(DM_Hilb.einstein_radius_in_kpc_from_radii(radii=radii))
print(power_law.einstein_radius_in_kpc_from_radii(radii=radii))

rho = power_law.surface_mass_density_from_radii(radii=radii)
kappa_pl = power_law.convergence_from_radii(radii=radii)
kappa_via_sigma = rho / power_law.critical_surface_density_of_lens

rho_NFW = DM_Hilb.surface_mass_density_from_radii(radii=radii)
kappa_NFW = DM_Hilb.convergence_from_radii(radii=radii)
kappa_via_sigma_NFW = rho_NFW / DM_Hilb.critical_surface_density_of_lens

rho_H = Hernquist.surface_mass_density_from_radii(radii=radii)
kappa_H = Hernquist.convergence_from_radii(radii=radii)
kappa_via_sigma_H = rho_H / Hernquist.critical_surface_density_of_lens

fig1 = plt.figure(1)
plt.loglog(radii, kappa_pl, label="kappa")
plt.loglog(radii, kappa_via_sigma, label="kappa sig")
plt.legend()
plt.show()

fig2 = plt.figure(2)
plt.loglog(radii, kappa_H, label="Hernquist")
plt.loglog(radii, kappa_NFW, label="NFW")
plt.loglog(radii, kappa_total, label="total")
plt.legend()

fig3 = plt.figure(3)
plt.loglog(radii, kappa_NFW, label="kappa")
plt.loglog(radii, kappa_via_sigma_NFW, label="kappa sig")
plt.legend()

fig4 = plt.figure(4)
plt.loglog(radii, rho_NFW, label="rho")
plt.loglog(radii, kappa_NFW, label="kappa")
plt.loglog(radii, kappa_via_sigma_NFW, label="kappa sig")
plt.legend()

plt.show()
