import numpy as np
from astropy import constants as const
from astropy import cosmology
from astropy import units as u
from scipy import integrate
from lens1d.profiles import oned
from scipy import optimize
from scipy.special import gamma

cosmo = cosmology.Planck15

def mass_dynamical(rho0, g, Reff, r0=1 * u.kpc):
    return ((4 * np.pi * rho0 / (3 - g)) * Reff ** 3 * (Reff / r0) ** -g).to("Msun")


def mass_einstein(rho0, g, Rein, r0=1 * u.kpc):
    return (
        (2 * np.pi ** 1.5 * gamma(0.5 * (g - 1)) / ((3 - g) * gamma(0.5 * g)))
        * rho0
        * Rein ** 3
        * (Rein / r0) ** -g
    ).to("Msun")


def vector_residuals(params, mass_dynamical_true, mass_einstein_true, Reff, Rein):

    log_rho0, g = params

    rho0 = 10 ** log_rho0 * u.Msun / u.kpc ** 3

    mass_dynamical_prediction = mass_dynamical(rho0, g, Reff)
    mass_einstein_prediction = mass_einstein(rho0, g, Rein)
    return (
        np.log10((mass_dynamical_prediction / mass_dynamical_true).to("")),
        np.log10((mass_einstein_prediction / mass_einstein_true).to("")),
    )


class PowerLawFit:
    def __init__(self, profile, radii, mask=None):
        self.profile = profile
        self.radii = radii

        if mask is None:
            self.mask = np.ones(len(self.radii))
        else:
            self.mask = mask

    def slope_and_normalisation_via_dynamics(self):

        mass_ein = self.profile.einstein_mass_in_solar_masses_from_radii(radii=self.radii)

        r_ein = self.profile.einstein_radius_in_kpc_from_radii(radii=self.radii)

        r_dyn = self.profile.effective_radius*1.33

        mass_dyn = self.profile.three_dimensional_mass_enclosed_within_radii(radii=r_dyn)

        init_guess = np.array([7, 1.9])

        root_finding_data = optimize.root(
            vector_residuals,
            init_guess,
            args=(mass_dyn * u.Msun, mass_ein * u.Msun, r_dyn * u.kpc, r_ein * u.kpc),
            method="hybr",
            options={"xtol": 1.5e-8},
        )

        print(root_finding_data.success)
        print(root_finding_data.message)

        return np.array([10 ** root_finding_data.x[0], root_finding_data.x[1]])

    def convergence_via_dynamics(self):
        einstein_radius = self.profile.einstein_radius_in_kpc_from_radii(radii=self.radii)
        slope = self.slope_and_normalisation_via_dynamics()[1]

        power_law = oned.SphericalPowerLaw(
            einstein_radius=einstein_radius,
            slope=slope,
            redshift_lens=self.profile.redshift_lens,
            redshift_source=self.profile.redshift_source,
        )

        return power_law.convergence_from_radii(radii=self.radii)

    def xi_two(self):

        kappa_ein = self.profile.convergence_at_einstein_radius_from_radii(radii=self.radii)

        dd_alpha_ein = self.profile.second_derivative_of_deflections_at_einstein_radius_from_radii(
            radii=self.radii
        )

        einstein_radius = self.profile.einstein_radius_in_kpc_from_radii(radii=self.radii)

        return np.divide(einstein_radius * dd_alpha_ein, 1 - kappa_ein)

    def slope_via_lensing(self):

        xi_two = self.xi_two()

        return np.divide(xi_two, 2) + 2

    def convergence_via_lensing(self):
        slope = self.slope_via_lensing()

        einstein_radius = self.profile.einstein_radius_in_kpc_from_radii(radii=self.radii)

        power_law = oned.SphericalPowerLaw(
            einstein_radius=einstein_radius,
            slope=slope,
            redshift_lens=self.profile.redshift_lens,
            redshift_source=self.profile.redshift_source,
        )

        return power_law.convergence_from_radii(radii=self.radii)
    
    def convergence_coefficients(
            self):
        kappa = self.profile.convergence_from_radii(radii=self.radii)

        coeffs, cov = np.polyfit(np.log(self.radii), np.log(kappa), w=self.mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def r_squared_value(self):
        kappa = self.profile.convergence_from_radii(radii=self.radii)
        polydata = self.convergence()

        sstot = sum((np.log(kappa) - np.mean(np.log(kappa))) ** 2)
        ssres = sum((np.log(kappa) - np.log(polydata)) ** 2)

        return 1 - (ssres / sstot)

    def slope_with_error(self):
        coeff, error = self.convergence_coefficients()[:, 0]

        slope = np.abs(coeff - 1)

        return np.array([slope, error])

    def convergence(self):
        coeffs = self.convergence_coefficients()[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(self.radii)

    def einstein_radius_with_error(self):
        normalization, error = self.convergence_coefficients()[:, 1]

        slope = self.slope_with_error()[0]

        einstein_radius = np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

        return np.array([einstein_radius, error])

    def einstein_mass_in_solar_masses(self):
        einstein_radius_rad = (
                self.einstein_radius_with_error()
                * u.arcsec
        ).to(u.rad)
        D_l = cosmo.angular_diameter_distance(self.profile.redshift_lens).to(u.m)

        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = self.profile.critical_surface_density_of_lens

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30

    ###  FROM HERE ON IS OLD CODE CALCULATING SLOPES AS A BEST FIT TO THE DEFLECTION ANGLES

    def deflections_coefficients(
            self
    ):
        alpha = self.profile.deflections_from_radii(radii=self.radii)

        coeffs, cov = np.polyfit(np.log(self.radii), np.log(alpha), w=self.mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def deflections(self):
        coeffs = self.deflections_coefficients()[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(self.radii)

    def convergence_via_deflections(
            self
    ):
        alpha = self.deflections(
        )

        return 0.5 * ((alpha / self.radii) + np.gradient(alpha, self.radii[:]))

    def convergence_coefficients_via_deflections(
            self
    ):
        best_fit_kappa = self.convergence_via_deflections()

        return np.polyfit(np.log(self.radii), np.log(best_fit_kappa), deg=1)

    def slope_via_deflections(self):
        coeff = self.convergence_coefficients_via_deflections()

        return np.abs(coeff[0] - 1)

    def einstein_radius_via_deflections_with_error(
            self
    ):
        normalization = self.convergence_coefficients_via_deflections()[
            1
        ]

        slope = self.slope_via_deflections()

        return np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

    def einstein_mass_in_solar_masses_via_deflections(
            self
    ):
        einstein_radius_rad = (
                self.einstein_radius_via_deflections_with_error()
                * u.arcsec
        ).to(u.rad)

        D_s = cosmo.angular_diameter_distance(self.profile.redshift_source).to(u.m)
        D_l = cosmo.angular_diameter_distance(self.profile.redshift_lens).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(self.profile.redshift_lens, self.profile.redshift_source).to(
            u.m
        )
        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = (
                np.divide(2.998e8 ** 2, 4 * np.pi * 6.674e-11) * np.divide(D_s, D_l * D_ls)
        ).value

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30



