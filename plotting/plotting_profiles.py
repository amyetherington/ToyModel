import one_d_profiles as profile
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math

fig_path = "/Users/dgmt59/Documents/Plots/1D/"

radii = np.arange(0.2, 50, 0.002)

DM_Bart = profile.NFW_Bartelmann(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
DM_Keeton = profile.NFW_Keeton(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
DM_Hilbert = profile.NFW_Hilbert(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
Hernquist = profile.Hernquist(mass=3.4e10, r_eff=8.4, z_l=0.3, z_s=0.8)
total = profile.CombinedProfile(mass_profiles=[Hernquist, DM_Hilbert], z_l=0.3, z_s=0.8)

density_NFW = DM_Hilbert.density_from_radii(radii=radii)
density_Hernquist =Hernquist.density_from_radii(radii=radii)
density_total = total.density_from_radii(radii=radii)
total_density = density_NFW + density_Hernquist

kappa_Bartelmann = DM_Bart.convergence_from_radii(radii=radii)
kappa_Keeton = DM_Keeton.convergence_from_radii(radii=radii)
kappa_Hilbert = DM_Hilbert.convergence_from_radii(radii=radii)
kappa_Hernquist = Hernquist.convergence_from_radii(radii=radii)
total_kappa = total.convergence_from_radii(radii=radii)

surface_density_DM = DM_Hilbert.surface_mass_density_from_radii(radii=radii)


alpha_Bartelmann = DM_Bart.deflection_angles_from_radii(radii=radii)
alpha_Keeton = DM_Keeton.deflection_angles_from_radii(radii=radii)
alpha_Hilbert = DM_Hilbert.deflection_angles_from_radii(radii=radii)

k_Bartelmann = 0.5 * ((alpha_Bartelmann / radii) + np.gradient(alpha_Bartelmann, radii[:]))
k_Keeton = 0.5 * ((alpha_Keeton / radii) + np.gradient(alpha_Keeton, radii[:]))

einstein_radius = 

fig1 = plt.figure(1)
plt.loglog(radii, density_Hernquist, label='Hernquist')
plt.loglog(radii, density_NFW, label='NFW')
plt.loglog(radii, density_total, label='total')
plt.legend()


fig2 = plt.figure(2)
plt.loglog(radii, kappa_Hernquist, label='Hernquist')
plt.loglog(radii, kappa_Hilbert, label='NFW')
plt.loglog(radii, total_kappa, label='total')
plt.legend()
plt.show()

fig2 = plt.figure(2)
plt.loglog(radii, k_Bartelmann, label='Bartelmann')
plt.loglog(radii, k_Keeton, label='Keeton')
plt.legend()

fig3 = plt.figure(3)
plt.loglog(radii, alpha_Bartelmann, label='Bartelmann')
plt.loglog(radii, alpha_Keeton, label='Keeton')
plt.legend()

plt.show()

fig1 = plt.figure(1)
plt.loglog(radii, kappa_Keeton, label='Keeton')
plt.loglog(radii, kappa_Bartelmann, label='Bartelmann')
plt.loglog(radii, kappa_Hilbert, label='Hilbert')
plt.legend()
#plt.savefig(fig_path + 'kappa_comparison', bbox_inches='tight', dpi=300)


fig2 = plt.figure(2)
plt.loglog(radii, alpha_Keeton, label='Keeton')
plt.loglog(radii, alpha_Bartelmann, label='Bartelmann')
plt.loglog(radii, alpha_Hilbert, label='Hilbert')
plt.legend()
#plt.savefig(fig_path + 'alpha_comparison', bbox_inches='tight', dpi=300)

