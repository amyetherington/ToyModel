import numpy as np
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from scipy import integrate

cosmo = cosmology.Planck15


class AbstractProfile:
    def __init__(self, redshift_lens, redshift_source):
        self.redshift_source = redshift_source
        self.redshift_lens = redshift_lens

    def convergence_from_radii(self, radii):
        raise NotImplementedError()

    def surface_mass_density_from_radii(self, radii):
        raise NotImplementedError()

    def deflections_from_radii(self, radii):
        raise NotImplementedError()

    def density_from_radii(self, radii):
        raise NotImplementedError()

    def shear_from_radii(self, radii):

        deflections = self.deflections_from_radii(radii=radii)

        return 0.5 * ((deflections / radii) - np.gradient(deflections, radii[:]))

    def tangential_eigenvalue_from_radii(self, radii):

        kappa = self.convergence_from_radii(radii=radii)
        gamma = self.shear_from_radii(radii=radii)

        return 1 - kappa - gamma

    def einstein_radius_in_kpc_from_radii(self, radii):

        lambda_t = self.tangential_eigenvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(lambda_t))

        return radii[index]

    @property
    def critical_surface_density_of_lens(self):

        d_s = cosmo.angular_diameter_distance(self.redshift_source).to(u.kpc)
        d_l = cosmo.angular_diameter_distance(self.redshift_lens).to(u.kpc)
        d_ls = cosmo.angular_diameter_distance_z1z2(
            self.redshift_lens, self.redshift_source
        ).to(u.kpc)

        return (
            np.divide(
                const.c.to("kpc/s") ** 2, 4 * np.pi * const.G.to("kpc3 / (Msun s2)")
            )
            * np.divide(d_s, d_l * d_ls)
        ).value

    def einstein_mass_in_solar_masses_from_radii(self, radii):

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = self.critical_surface_density_of_lens

        return np.pi * einstein_radius ** 2 * sigma_crit

    def two_dimensional_mass_enclosed_within_radii(self, radii):

        integrand = (
            lambda r: 2 * np.pi * r * self.surface_mass_density_from_radii(radii=r)
        )

        return integrate.quad(integrand, 0, radii)[0]

    def three_dimensional_mass_enclosed_within_radii(self, radii):

        integrand = lambda r: 4 * np.pi * r ** 2 * self.density_from_radii(radii=r)

        return integrate.quad(integrand, 0, radii)[0]

    def second_derivative_of_deflections_from_radii(self, radii):
        alpha = self.deflections_from_radii(radii=radii)

        d_alpha = np.gradient(alpha, radii[:])

        return np.gradient(d_alpha, radii[:])

    def second_derivative_of_deflections_at_einstein_radius_from_radii(self, radii):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        index = np.argmin(np.abs(np.array(radii) - einstein_radius))

        dd_alpha = self.second_derivative_of_deflections_from_radii(radii=radii)

        return dd_alpha[index]

    def convergence_at_einstein_radius_from_radii(self, radii):

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        return self.convergence_from_radii(radii=einstein_radius)

    def f_func(self, x):
        f = np.where(
            x > 1,
            1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1),
            x,
        )
        f = np.where(x < 1, 1 - (1 / np.sqrt(1 - x ** 2)) * np.arccosh(1 / x), f)
        return np.where(x == 1, 0, f)

