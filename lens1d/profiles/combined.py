import numpy as np
from astropy import cosmology
from astropy import units as u
from lens1d.profiles import abstract
from lens1d.profiles import oned
from scipy import optimize
from scipy.special import gamma

cosmo = cosmology.Planck15

class CombinedProfile(abstract.AbstractProfile):
    def __init__(self, profiles=None):

        self.profiles = profiles or []

        # TODO : Check input redshifts and raise an error
        super().__init__(
            redshift_lens=self.profiles[0].redshift_lens,
            redshift_source=self.profiles[0].redshift_source,
        )

    @classmethod
    def from_dark_matter_fraction_within_effective_radius(cls, hernquist, nfw, effective_radius):

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