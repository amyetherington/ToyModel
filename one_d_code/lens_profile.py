import numpy as np
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from scipy import integrate
from scipy import optimize

cosmo = cosmology.Planck15


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

        D_s = cosmo.angular_diameter_distance(self.z_s).to(u.kpc)
        D_l = cosmo.angular_diameter_distance(self.z_l).to(u.kpc)
        D_ls = cosmo.angular_diameter_distance_z1z2(self.z_l, self.z_s).to(u.kpc)

        sigma_crit = (
            np.divide(
                const.c.to("kpc/s") ** 2, 4 * np.pi * const.G.to("kpc3 / (Msun s2)")
            )
            * np.divide(D_s, D_l * D_ls)
        ).value

        return sigma_crit

    def einstein_mass_in_solar_masses_from_radii(self, radii):

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = self.critical_surface_density_of_lens

        return np.pi * einstein_radius ** 2 * sigma_crit

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

    def second_derivative_of_deflection_angles_from_radii(self, radii):
        alpha = self.deflection_angles_from_radii(radii=radii)

        d_alpha = np.gradient(alpha, radii[:])

        return np.gradient(d_alpha, radii[:])

    def second_derivative_of_deflection_angles_at_einstein_radius_from_radii(
        self, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        index = np.argmin(
            np.abs(np.array(radii) - einstein_radius)
        )

        dd_alpha = self.second_derivative_of_deflection_angles_from_radii(
            radii=radii
        )

        return dd_alpha[index]

    def convergence_at_einstein_radius_from_radii(self, radii):

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        kappa = self.convergence_from_radii(radii=einstein_radius)

        return kappa

    def xi_two(self, radii):

        kappa_ein = self.convergence_at_einstein_radius_from_radii(radii=radii)

        dd_alpha_ein = self.second_derivative_of_deflection_angles_at_einstein_radius_from_radii(radii=radii)

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        return np.divide(einstein_radius * dd_alpha_ein, 1 - kappa_ein)

    def slope_via_lensing(self, radii):

        xi_two = self.xi_two(radii=radii)

        return np.divide(xi_two, 2) + 2

    def f_func(self, x):
        f = np.where(
            x > 1,
            1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1),
            x,
        )
        f = np.where(x < 1, 1 - (1 / np.sqrt(1 - x ** 2)) * np.arccosh(1 / x), f)
        return np.where(x==1, 0, f)

