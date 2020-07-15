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

    def xi_two(self, radii):

        kappa_ein = self.convergence_at_einstein_radius_from_radii(radii=radii)

        dd_alpha_ein = self.second_derivative_of_deflections_at_einstein_radius_from_radii(
            radii=radii
        )

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
        return np.where(x == 1, 0, f)

    # TODO : If lots of functions have redudant names (e.g. 'inferred') its a good idea to try and remove the
    # TODO : redundancy. So I'd rrename these.

    # TODO : For long functin names we've started to drop the from_thing_and_thing for just from, which Ive done below.

    ## also can I make it so mask is either used or not so I don't need from mask and radii and from radii functions?
    def inferred_convergence_coefficients_from_mask_and_radii(
            self, mask, radii
    ):
        kappa = self.convergence_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(kappa), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def inferred_convergence_coefficients_from_radii(self, radii):
        kappa = self.convergence_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(kappa), deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def power_law_r_squared_value(self, mask, radii):
        kappa = self.convergence_from_radii(radii=radii)
        polydata = self.inferred_convergence_from_mask_and_radii(
            mask=mask, radii=radii
        )

        sstot = sum((np.log(kappa) - np.mean(np.log(kappa))) ** 2)
        ssres = sum((np.log(kappa) - np.log(polydata)) ** 2)

        return 1 - (ssres / sstot)

    def inferred_slope_with_error_from_mask_and_radii(self, mask, radii):
        coeff, error = self.inferred_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[
                       :, 0
                       ]

        slope = np.abs(coeff - 1)

        return np.array([slope, error])

    def inferred_convergence_from_mask_and_radii(self, mask, radii):
        coeffs = self.inferred_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def inferred_slope_with_error_from_radii(self, radii):
        coeff, error = self.inferred_convergence_coefficients_from_radii(
            radii=radii
        )[:, 0]

        slope = np.abs(coeff - 1)

        return np.array([slope, error])

    def inferred_convergence_from_radii(self, radii):
        coeffs = self.inferred_convergence_coefficients_from_radii(
            radii=radii
        )[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def inferred_einstein_radius_with_error_from_mask_and_radii(
            self, mask, radii
    ):
        normalization, error = self.inferred_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[
                               :, 1
                               ]

        slope = self.inferred_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius = np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

        return np.array([einstein_radius, error])

    def inferred_einstein_radius_with_error_from_radii(self, radii):
        normalization, error = self.inferred_convergence_coefficients_from_radii(
            radii=radii
        )[
                               :, 1
                               ]

        slope = self.inferred_slope_with_error_from_radii(radii=radii)[0]

        einstein_radius = np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

        return np.array([einstein_radius, error])

    def inferred_einstein_mass_in_solar_masses_from_mask_and_radii(
            self, radii, mask, redshift_lens
    ):
        einstein_radius_rad = (
                self.inferred_einstein_radius_with_error_from_mask_and_radii(
                    radii=radii, mask=mask
                )
                * u.arcsec
        ).to(u.rad)
        D_l = cosmo.angular_diameter_distance(self.redshift_lens).to(u.m)

        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = self.critical_surface_density_of_lens

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30

    ###  FROM HERE ON IS OLD CODE CALCULATING SLOPES AS A BEST FIT TO THE DEFLECTION ANGLES

    def inferred_deflections_coefficients_from_mask_and_radii(
            self, mask, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)
        alpha = self.deflections_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(alpha), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def inferred_deflections_from_mask_and_radii(self, mask, radii):
        coeffs = self.inferred_deflections_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def inferred_convergence_via_deflections_from_mask_and_radii(
            self, mask, radii
    ):
        alpha = self.inferred_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        return 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    def inferred_convergence_coefficients_via_deflections_from_mask_and_radii(
            self, mask, radii
    ):
        best_fit_kappa = self.inferred_convergence_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        return np.polyfit(np.log(radii), np.log(best_fit_kappa), deg=1)

    def inferred_slope_via_deflections_from_mask_and_radii(self, mask, radii):
        coeff = self.inferred_convergence_coefficients_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        return np.abs(coeff[0] - 1)

    def inferred_einstein_radius_via_deflections_with_error_from_mask_and_radii(
            self, mask, radii
    ):
        normalization = self.inferred_convergence_coefficients_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )[
            1
        ]

        slope = self.inferred_slope_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        return np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

    def inferred_einstein_mass_in_solar_masses_via_deflections_from_mask_and_radii_and_redshifts(
            self, radii, mask, redshift_source, redshift_lens
    ):
        einstein_radius_rad = (
                self.inferred_einstein_radius_via_deflections_with_error_from_mask_and_radii(
                    radii=radii, mask=mask
                )
                * u.arcsec
        ).to(u.rad)

        D_s = cosmo.angular_diameter_distance(redshift_source).to(u.m)
        D_l = cosmo.angular_diameter_distance(redshift_lens).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(redshift_lens, redshift_source).to(
            u.m
        )
        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = (
                np.divide(2.998e8 ** 2, 4 * np.pi * 6.674e-11) * np.divide(D_s, D_l * D_ls)
        ).value

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30
