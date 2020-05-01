import numpy as np
from astropy import cosmology
from astropy import units as u
from scipy.optimize import fsolve
from scipy import integrate

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

class LensProfile:

    def __init__(self, z_l, z_s):
        self.z_s = z_s
        self.z_l = z_l

    def convergence_from_radii(self, radii):
        raise NotImplementedError()

    def surface_mass_density_from_radii(self, radii):
        raise NotImplementedError()

    def deflection_angles_from_radii(self, radii):
        raise NotImplementedError()

    def density_from_radii(self, radii):
        raise NotImplementedError()

    def shear_from_radii(self, radii):

        alpha = self.deflection_angles_from_radii(radii=radii)

        gamma = 0.5 * ((alpha / radii) - np.gradient(alpha, radii[:]))

        return gamma

    def tangential_eiginvalue_from_radii(self, radii):

        kappa = self.convergence_from_radii(radii=radii)
        gamma = self.shear_from_radii(radii=radii)

        return 1 - kappa - gamma

    def einstein_radius_in_arcseconds_from_radii(self, radii):

        lambda_t = self.tangential_eiginvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(lambda_t))

        return radii[index]

    # TODO : If a functioon has no inputs, you probably want it to be a property. This means you dont need the () when
    # TODO : you call it.

    @property
    def critical_surface_density_of_lens(self):

        D_s = cosmo.angular_diameter_distance(self.z_s).to(u.m)
        D_l = cosmo.angular_diameter_distance(self.z_l).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(self.z_l, self.z_s).to(u.m)

        sigma_crit = (
                np.divide(2.998e8 ** 2, 4 * np.pi * 6.674e-11) * np.divide(D_s, D_l * D_ls)
        ).value

        return sigma_crit

    def einstein_mass_in_solar_masses_from_radii(self, radii):

        einstein_radius_radians = (
                self.einstein_radius_in_arcseconds_from_radii(radii=radii) * u.arcsec
        ).to(u.rad)

        D_l = cosmo.angular_diameter_distance(self.z_l).to(u.m)

        einstein_radius = (einstein_radius_radians * D_l).value

        sigma_crit = self.critical_surface_density_of_lens()

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30

    def two_dimensional_mass_enclosed_within_radii(self, radii):

        integrand = (
            lambda r: 2 * np.pi * r * self.surface_mass_density_from_radii(radii=r)
        )

        mass = integrate.quad(integrand, 0, radii)[0]

        return mass

    def three_dimensional_mass_enclosed_within_radii(self, radii):

        integrand = lambda r: 4 * np.pi * r ** 2 * self.density_from_radii(radii=r)

        mass = integrate.quad(integrand, 0, radii)[0]

        return mass

    # TODO : Moved this to here as all functions below use it.

    def f_func(self, x):
        f = np.where(
            x < 1,
            (np.divide(1, np.sqrt(1 - x ** 2)) * np.arctanh(np.sqrt(1 - x ** 2))),
            x,
            )
        f = np.where(
            x > 1,
            (np.divide(1, np.sqrt(x ** 2 - 1)) * np.arctan(np.sqrt(x ** 2 - 1))),
            f,
            )
        return f