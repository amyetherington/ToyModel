import numpy as np
from astropy import cosmology
from astropy import units as u
from lens1d.profiles import abstract
from lens1d.profiles import oned
from scipy import optimize
from scipy.special import gamma

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)


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


class CombinedProfile(abstract.AbstractProfile):
    def __init__(self, profiles=None):

        self.profiles = profiles or []

        # TODO : Check input redshifts and raise an error
        super().__init__(
            redshift_lens=self.profiles[0].redshift_lens,
            redshift_source=self.profiles[0].redshift_source,
        )

    @classmethod
    def from_dark_matter_fraction_within_einstein_radius(cls, hernquist, nfw, einstein_mass):

        # TODO : The input hernquist and nfw will not give us the desired dark matter fraction at the einstein radius
        # TODO : and will not give an einstein mass corresponing to the input einstein mass.

        # TODO : Below, we need a function which compures the correct values for hernquist.mass and nfw.mass_at_200
        # TODO : that does. This may have some analytic form, but could also be done with a root

        # MAGIC FUNCTION

        # hernquist.mass = new_hernquist_mass
        # nfw.mass_at_200 = new_nfw_mass_at_200

        return CombinedProfile(hernquist, nfw)

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

    def deflections_from_radii(self, radii):
        return sum(
            [profile.deflections_from_radii(radii=radii) for profile in self.profiles]
        )

    @property
    def stellar(self):

        profiles = [
            profile if isinstance(profile, oned.StellarProfile) else None
            for profile in self.profiles
        ]

        profiles = list(filter(None, profiles))

        check_is_single_profile(profiles=profiles)

        return profiles[0]

    @property
    def dark(self):

        profiles = [
            profile if isinstance(profile, oned.DarkProfile) else None
            for profile in self.profiles
        ]

        profiles = list(filter(None, profiles))

        check_is_single_profile(profiles=profiles)

        return profiles[0]

    @property
    def effective_radius(self):
        return self.stellar.effective_radius

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

    @property
    def dark_matter_mass_fraction_within_effective_radius(self):
        dm = self.dark

        dm_mass = dm.two_dimensional_mass_enclosed_within_radii(
            radii=self.effective_radius
        )
        total_mass = self.three_dimensional_mass_enclosed_within_effective_radius

        return dm_mass / total_mass

    def dark_matter_mass_fraction_within_einstein_radius_from_radii(self, radii):
        dm = self.dark
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        dm_mass = dm.two_dimensional_mass_enclosed_within_radii(radii=einstein_radius)
        total_mass = self.three_dimensional_mass_enclosed_within_radii(
            radii=einstein_radius
        )

        return dm_mass / total_mass

    def slope_and_normalisation_via_dynamics(self, radii):

        mass_ein = self.einstein_mass_in_solar_masses_from_radii(radii=radii)

        r_ein = self.einstein_radius_in_kpc_from_radii(radii=radii)

        r_dyn = self.effective_radius

        mass_dyn = self.three_dimensional_mass_enclosed_within_radii(radii=r_dyn)

        init_guess = np.array([7, 1.9])

        root_finding_data = optimize.root(
            vector_residuals,
            init_guess,
            args=(mass_dyn * u.Msun, mass_ein * u.Msun, r_dyn * u.kpc, r_ein * u.kpc),
            method="hybr",
            options={"xtol": 0.0001},
        )

        return np.array([10 ** root_finding_data.x[0], root_finding_data.x[1]])

    def einstein_radius_via_dynamics(self, radii):
        rho_0, slope = self.slope_and_normalisation_via_dynamics(radii=radii)

        A = np.divide(
            gamma(slope / (2 * self.critical_surface_density_of_lens)),
            gamma((slope - 1) / 2) * np.sqrt(np.pi),
        )

        return ((2 * rho_0) / (A * (3 - slope))) ** (1 - slope)

    # TODO : Delete? Surely we can just do this via the individual PowerLaw class.

    def power_law_convergence_via_dynamics(self, radii):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)
        slope = self.slope_and_normalisation_via_dynamics(radii=radii)[1]

        power_law = oned.SphericalPowerLaw(
            einstein_radius=einstein_radius,
            slope=slope,
            redshift_lens=self.redshift_lens,
            redshift_source=self.redshift_source,
        )

        return power_law.convergence_from_radii(radii=radii)

    def power_law_convergence_via_lensing(self, radii):
        slope = self.slope_via_lensing(radii=radii)

        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        power_law = oned.SphericalPowerLaw(
            einstein_radius=einstein_radius,
            slope=slope,
            redshift_lens=self.redshift_lens,
            redshift_source=self.redshift_source,
        )

        return power_law.convergence_from_radii(radii=radii)

    # should masks be a separate object and not part of a combined profile???
    def mask_radial_range_from_radii(self, lower_bound, upper_bound, radii):
        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1

        return weights

    def mask_two_radial_ranges_from_radii(
        self, lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2, radii
    ):
        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound_1)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound_1)))
        index3 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound_2)))
        index4 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound_2)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1
        weights[index3:index4] = 1

        return weights


def check_is_single_profile(profiles):

    if len(profiles) == 0:
        raise ValueError("No profile found in the CombinedProfile of this type.")
    elif len(profiles) > 1:
        raise ValueError(
            "Multiple profiles found in the CombinedProfile of this type, it is ambiguous which to use."
        )