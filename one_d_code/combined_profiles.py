import numpy as np
from astropy import cosmology
from astropy import units as u
from one_d_code import lens_profile as lp
from scipy import optimize
from scipy.special import gamma


cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

def Mdyn(rho0,g,Reff,r0=1*u.kpc):
    return ((4*np.pi*rho0 / (3-g)) * Reff**3 * (Reff/r0)**-g).to('Msun')

def Mein(rho0,g,Rein,r0=1*u.kpc):
    return ((2*np.pi**1.5 * gamma(0.5*(g-1)) / ((3-g)*gamma(0.5*g))) * rho0 * Rein**3 * (Rein/r0)**-g).to('Msun')


def vector_residuals(params, Mdyn_true, Mein_true, Reff, Rein):

    log_rho0, g = params

    rho0 = 10 ** log_rho0 * u.Msun / u.kpc ** 3

    Mdyn_pred = Mdyn(rho0, g, Reff)
    Mein_pred = Mein(rho0, g, Rein)
    return np.log10((Mdyn_pred / Mdyn_true).to('')), np.log10((Mein_pred / Mein_true).to(''))


class CombinedProfile(lp.LensProfile):

    def __init__(self, profiles=None):
        self.profiles = (
            profiles or []
        )  # If None, the profiles default to an empty list.

        # TODO : Check input redshifts and raise an error
        super().__init__(z_l=self.profiles[0].z_l, z_s=self.profiles[0].z_s)

    def density_from_radii(self, radii):
        return sum(
            [profile.density_from_radii(radii=radii) for profile in self.profiles]
        )

    def surface_mass_density_from_radii(self, radii):
        return sum(
            [
                profile.surface_mass_density_from_radii(radii=radii)
                for profile in self.profiles
            ]
        )

    def convergence_from_radii(self, radii):
        return sum(
            [profile.convergence_from_radii(radii=radii) for profile in self.profiles]
        )

    def deflection_angles_from_radii(self, radii):
        return sum(
            [
                profile.deflection_angles_from_radii(radii=radii)
                for profile in self.profiles
            ]
        )

    @property
    def effective_radius(self):
        effective_radii = [
            profile.effective_radius if hasattr(profile, "effective_radius") else None
            for profile in self.profiles
        ]

        effective_radii = list(filter(None, effective_radii))

        if len(effective_radii) == 0:
            raise ValueError("There are no effective radii in this Combined Profile")
        elif len(effective_radii) > 1:
            raise ValueError(
                "There are multiple effective radii in this Combined Profile, it is ambiguous which to use."
            )

        return effective_radii[0]

    @property
    def three_dimensional_mass_enclosed_within_effective_radius(self):
        return sum(
            [
                profile.three_dimensional_mass_enclosed_within_radii(
                    radii=self.effective_radius
                )
                for profile in self.profiles
            ]
        )

    @property
    def two_dimensional_mass_enclosed_within_effective_radius(self):
        return sum(
            [
                profile.two_dimensional_mass_enclosed_within_radii(
                    radii=self.effective_radius
                )
                for profile in self.profiles
            ]
        )

    def second_derivative_of_deflection_angles_at_einstein_radius_from_radii(
        self, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        index = np.argmin(
            np.abs(np.array(radii) - (einstein_radius))
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

    def slope_via_dynamics(self, radii):

        mass_ein = self.einstein_mass_in_solar_masses_from_radii(radii=radii)

        r_ein = self.einstein_radius_in_kpc_from_radii(radii=radii)

        r_dyn = self.effective_radius

        mass_dyn = self.three_dimensional_mass_enclosed_within_radii(radii=r_dyn)

        init_guess = np.array([7, 2])

        root_finding_data = optimize.root(vector_residuals, init_guess, args=(mass_dyn*u.Msun, mass_ein*u.Msun, r_dyn*u.kpc, r_ein*u.kpc),
                                          method='hybr', options={'xtol': 0.0001})

        return np.array([10**root_finding_data.x[0], root_finding_data.x[1]])

    def mask_radial_range_from_radii(self, lower_bound, upper_bound, radii):

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        index1 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + lower_bound * einstein_radius))
        )
        index2 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + upper_bound * einstein_radius))
        )
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1

        return weights

    def mask_two_radial_ranges_from_radii(
        self, lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        index1 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + lower_bound_1 * einstein_radius))
        )
        index2 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + upper_bound_1 * einstein_radius))
        )
        index3 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + lower_bound_2 * einstein_radius))
        )
        index4 = np.argmin(
            np.abs(np.array(radii) - (radii[0] + upper_bound_2 * einstein_radius))
        )
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1
        weights[index3:index4] = 1

        return weights

    def best_fit_power_law_convergence_coefficients_from_mask_and_radii(
        self, mask, radii
    ):

        kappa = self.convergence_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(kappa), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def best_fit_power_law_slope_with_error_from_mask_and_radii(self, mask, radii):
        coeff, error = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[
            :, 0
        ]

        slope = np.abs(coeff - 1)

        return np.array([slope, error])

    def best_fit_power_law_convergence_from_mask_and_radii(self, mask, radii):
        coeffs = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(
        self, mask, radii
    ):

        normalization, error = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[
            :, 1
        ]

        slope = self.best_fit_power_law_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius = np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

        return np.array([einstein_radius, error])

    def best_fit_einstein_mass_in_solar_masses_from_mask_and_radii_and_redshifts(
        self, radii, mask, z_l
    ):
        einstein_radius_rad = (
            self.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(
                radii=radii, mask=mask
            )
            * u.arcsec
        ).to(u.rad)
        D_l = cosmo.angular_diameter_distance(z_l).to(u.m)

        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = self.critical_surface_density_of_lens

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30

    def best_fit_power_law_deflection_angles_coefficients_from_mask_and_radii(
        self, mask, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)
        alpha = self.deflection_angles_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(alpha), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def best_fit_power_law_deflection_angles_from_mask_and_radii(self, mask, radii):
        coeffs = self.best_fit_power_law_deflection_angles_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[
            0
        ]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
        self, mask, radii
    ):
        alpha = self.best_fit_power_law_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

        return kappa

    def best_fit_power_law_convergence_coefficients_via_deflection_angles_from_mask_and_radii(
        self, mask, radii
    ):

        best_fit_kappa = self.best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        coeffs = np.polyfit(np.log(radii), np.log(best_fit_kappa), deg=1)

        ## need to figure out how to get error on convergence slope given error on slope of best fit deflection angles

        return coeffs

    def best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
        self, mask, radii
    ):
        coeff = self.best_fit_power_law_convergence_coefficients_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        return np.abs(coeff[0] - 1)

    def best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
        self, mask, radii
    ):

        normalization = self.best_fit_power_law_convergence_coefficients_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )[
            1
        ]

        slope = self.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        einstein_radius = np.exp(
            np.divide(normalization - np.log(np.divide(3 - slope, 2)), slope - 1)
        )

        return einstein_radius

    def best_fit_einstein_mass_in_solar_masses_via_deflection_angles_from_mask_and_radii_and_redshifts(
        self, radii, mask, z_s, z_l
    ):
        einstein_radius_rad = (
            self.best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
                radii=radii, mask=mask
            )
            * u.arcsec
        ).to(u.rad)

        D_s = cosmo.angular_diameter_distance(z_s).to(u.m)
        D_l = cosmo.angular_diameter_distance(z_l).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m)
        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = (
            np.divide(2.998e8 ** 2, 4 * np.pi * 6.674e-11) * np.divide(D_s, D_l * D_ls)
        ).value

        return (4 * np.pi * einstein_radius ** 2 * sigma_crit) / 1.989e30
