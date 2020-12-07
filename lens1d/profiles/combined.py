import numpy as np
from astropy import cosmology
from astropy import units as u
from lens1d.profiles import abstract
from lens1d.profiles import oned
from scipy import optimize
from scipy.special import gamma

cosmo = cosmology.Planck15

def residuals(m200, r_eff, dm_mass_within_r_eff_true, redshift_lens, redshift_source):

    nfw = oned.NFWHilbert(mass_at_200=m200, redshift_lens=redshift_lens, redshift_source=redshift_source)

    dm_mass_within_r_eff_prediction = nfw.three_dimensional_mass_enclosed_within_radii(radii=r_eff)

    return np.log10((dm_mass_within_r_eff_prediction / dm_mass_within_r_eff_true))

class CombinedProfile(abstract.AbstractProfile):
    def __init__(self, profiles=None):

        self.profiles = profiles or []

        # TODO : Check input redshifts and raise an error
        super().__init__(
            redshift_lens=self.profiles[0].redshift_lens,
            redshift_source=self.profiles[0].redshift_source,
        )

    @classmethod
    def from_hernquist_and_dark_matter_fraction_within_effective_radius(cls, hernquist, dark_matter_fraction):

        stellar_mass_within_r_eff = hernquist.three_dimensional_mass_enclosed_within_radii(radii=hernquist.effective_radius)

        dm_mass_within_r_eff_from_f_dm = np.divide(dark_matter_fraction*stellar_mass_within_r_eff, 1 - dark_matter_fraction)

        init_guess = np.array([1e13])

        root_finding_data = optimize.root(
            residuals,
            init_guess,
            args=(hernquist.effective_radius, dm_mass_within_r_eff_from_f_dm, hernquist.redshift_lens, hernquist.redshift_source),
            method="hybr",
            options={"xtol": 0.0001},
        )

        nfw = oned.NFWHilbert(mass_at_200=root_finding_data.x,
                              redshift_source=hernquist.redshift_source,
                              redshift_lens=hernquist.redshift_lens)

        return CombinedProfile([hernquist, nfw])

    @classmethod
    def from_effective_and_einstein_radii_and_dark_matter_fraction_within_effective_radius(cls, effective_radius, einstein_radius, dark_matter_fraction):

        stellar_mass_within_r_eff = hernquist.three_dimensional_mass_enclosed_within_radii(
            radii=hernquist.effective_radius)

        dm_mass_within_r_eff_from_f_dm = np.divide(dark_matter_fraction * stellar_mass_within_r_eff,
                                                   1 - dark_matter_fraction)

        init_guess = np.array([1e13])

        root_finding_data = optimize.root(
            residuals,
            init_guess,
            args=(hernquist.effective_radius, dm_mass_within_r_eff_from_f_dm, hernquist.redshift_lens,
                  hernquist.redshift_source),
            method="hybr",
            options={"xtol": 0.0001},
        )

        nfw = oned.NFWHilbert(mass_at_200=root_finding_data.x,
                              redshift_source=hernquist.redshift_source,
                              redshift_lens=hernquist.redshift_lens)

        return CombinedProfile([hernquist, nfw])

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
        total_mass = self.two_dimensional_mass_enclosed_within_effective_radius

        return dm_mass / total_mass

    def dark_matter_mass_fraction_within_einstein_radius_from_radii(self, radii):
        dm = self.dark
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)

        dm_mass = dm.two_dimensional_mass_enclosed_within_radii(radii=einstein_radius)
        total_mass = self.two_dimensional_mass_enclosed_within_radii(
            radii=einstein_radius
        )

        return dm_mass / total_mass

    def mask_einstein_radius_from_radii(self, width, radii):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)
        lower_bound = einstein_radius-width/2
        upper_bound = einstein_radius+width/2
        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1

        return weights

    def mask_effective_radius_from_radii(self, width, radii):
        effective_radius = self.effective_radius
        lower_bound = effective_radius - width / 2
        upper_bound = effective_radius + width / 2
        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1

        return weights

    def mask_einstein_and_effective_radius_from_radii(
        self, width_around_einstein_radius, width_around_effective_radius, radii
    ):
        einstein_radius = self.einstein_radius_in_kpc_from_radii(radii=radii)
        effective_radius = self.effective_radius
        lower_bound_1 = effective_radius - width_around_effective_radius / 2
        upper_bound_1 = effective_radius + width_around_effective_radius / 2
        lower_bound_2 = einstein_radius - width_around_einstein_radius / 2
        upper_bound_2 = einstein_radius + width_around_einstein_radius / 2

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