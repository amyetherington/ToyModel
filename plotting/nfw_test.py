from astropy import cosmology
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as const
from autogalaxy.util import cosmology_util



cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

R = np.arange(0.2, 300, 0.002)

def r200_from_M200(M200,z,cosmo=cosmo):
    return ((M200*u.solMass/(1.333 * np.pi * 200 * cosmo.critical_density(z).to("Msun/kpc3")))**(1./3))

def NFW_Sigma_Bartelmann(R,M200,c,z,cosmo=cosmo):
    # http://articles.adsabs.harvard.edu//full/1996A%26A...313..697B/0000697.000.html
    r200 = r200_from_M200(M200,z,cosmo=cosmo)
    print(r200)
    r_s = (r200 / c).value
    rho_0 = M200 / (4*np.pi*r_s**3*(np.log(1.+c)-c/(1.+c)))
    x = (R/r_s)
    f = 1 - 2*np.arctan(np.sqrt((x-1)/(x+1)))/np.sqrt(x**2 - 1)
    f[x<1] = 1 - 2*np.arctanh(np.sqrt((1-x[x<1])/(x[x<1]+1)))/np.sqrt(1-x[x<1]**2)
    return 2*rho_0*r_s*f / (x**2 -1)

D_s = cosmo.angular_diameter_distance(1).to(u.kpc)
D_l = cosmo.angular_diameter_distance(0.6).to(u.kpc)
D_ls = cosmo.angular_diameter_distance_z1z2(0.6, 1).to(u.kpc)

sigma_crit_autolens = cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
    redshift_0=0.6, redshift_1=1, cosmology=cosmo, unit_length="kpc")

sigma_crit = np.divide(const.c.to("kpc/s") ** 2, 4 * np.pi * const.G.to("kpc3 / (Msun s2)")) * np.divide(D_s, D_l * D_ls)

NFW = NFW_Sigma_Bartelmann(M200=2.5e12, c=3.4, z=0.6, R=R)
NFW_kappa = NFW / sigma_crit

print(cosmo.critical_density(0.6).to("Msun/kpc**3"))
print(sigma_crit)
print(sigma_crit_autolens)

fig1 = plt.figure(1)
plt.loglog(R, NFW, label="sigma")
plt.loglog(R, NFW_kappa, label="kappa")
plt.show()




