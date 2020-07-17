import numpy as np
from astropy import cosmology
from astropy import units as u

from colossus.halo.concentration import concentration as col_concentration
from colossus.cosmology import cosmology as col_cosmology

from lens1d.profiles import abstract

cosmo = cosmology.Planck15


class StellarProfile:

    pass


class DarkProfile:

    pass


class SphericalPowerLaw(abstract.AbstractProfile):
    def __init__(self, einstein_radius, slope, redshift_source, redshift_lens):
        super().__init__(redshift_lens=redshift_lens, redshift_source=redshift_source)
        self.einstein_radius = einstein_radius
        self.slope = slope
        self.m_ein = (
            np.pi * einstein_radius ** 2 * self.critical_surface_density_of_lens
        )
        self.rho_s = np.divide(self.m_ein, np.pi * einstein_radius ** 2)

    def density_from_radii(self, radii):
        return self.critical_surface_density_of_lens * np.divide(1, radii) ** self.slope

    def surface_mass_density_from_radii(self, radii):
        return (
            self.critical_surface_density_of_lens
            * np.divide((3 - self.slope), 2)
            * np.divide(self.einstein_radius, radii) ** (self.slope - 1)
        )

    def convergence_from_radii(self, radii):
        return np.divide((3 - self.slope), 2) * np.divide(
            self.einstein_radius, radii
        ) ** (self.slope - 1)

    def deflections_from_radii(self, radii):
        return self.einstein_radius * np.divide(self.einstein_radius, radii) ** (
            self.slope - 2
        )


class Hernquist(abstract.AbstractProfile, StellarProfile):
    def __init__(self, mass, effective_radius, redshift_source, redshift_lens):

        super().__init__(redshift_lens=redshift_lens, redshift_source=redshift_source)

        self.mass = mass
        self.effective_radius = effective_radius
        self.half_mass_radius = self.effective_radius * 1.33
        self.scale_radius = self.effective_radius / 1.8153
        self.rho_s = np.divide(self.mass, 2 * np.pi * self.scale_radius ** 3)
        self.kappa_s = (
            self.rho_s * self.scale_radius / self.critical_surface_density_of_lens
        )

    def density_from_radii(self, radii):

        x = np.array(radii / self.scale_radius)

        return np.divide(self.rho_s, x * (1 + x) ** 3)

    def surface_mass_density_from_radii(self, radii):

        x = np.array(radii / self.scale_radius)
        f = self.f_func(x)

        return np.where(
            f == 0,
            0,
            np.divide(self.rho_s * self.scale_radius, (x ** 2 - 1) ** 2)
            * (-3 + (1 - f) * (2 + x ** 2)),
        )

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.scale_radius)
        f = self.f_func(x)

        return np.real(np.where(
            f == 0,
            0,
            np.real(np.divide(self.kappa_s, (x ** 2 - 1) ** 2) * (-3 + (1 - f) * (2 + x ** 2)),
        )))

    def deflections_from_radii(self, radii):
        x = np.array(radii / self.scale_radius)
        f = self.f_func(x)

        return np.real(np.where(
            f == 0,
            0,
            2 * self.kappa_s * self.scale_radius * np.divide(x * f, x ** 2 - 1),
        ))


class NFWHilbert(abstract.AbstractProfile, DarkProfile):
    def __init__(self, mass_at_200, redshift_source, redshift_lens):

        super().__init__(redshift_lens=redshift_lens, redshift_source=redshift_source)

        self.mass_at_200 = mass_at_200
        col_cosmo = col_cosmology.setCosmology("planck15")
        self.concentration = col_concentration(
            self.mass_at_200 * col_cosmo.h, "200c", self.redshift_lens, model="ludlow16"
        )
        self.radius_at_200 = (
            self.mass_at_200
            * u.solMass
            / (
                1.333
                * np.pi
                * 200
                * cosmo.critical_density(self.redshift_lens).to("Msun/kpc**3")
            )
        ) ** (1.0 / 3)
        self.scale_radius = self.radius_at_200.value / self.concentration
        self.rho_s = self.mass_at_200 / (
            4
            * np.pi
            * self.scale_radius ** 3
            * (
                np.log(1.0 + self.concentration)
                - self.concentration / (1.0 + self.concentration)
            )
        )
        self.kappa_s = np.divide(
            self.rho_s * self.scale_radius, self.critical_surface_density_of_lens
        )

    def density_from_radii(self, radii):
        x = np.divide(radii, self.scale_radius)
        return np.divide(self.rho_s, x * (1 + x) ** 2)

    def surface_mass_density_from_radii(self, radii):
        x = np.divide(radii, self.scale_radius)
        f = self.f_func(x)
        sigma = 2 * self.rho_s * self.scale_radius * (np.array(f) / (x ** 2 - 1))

        return np.where(f == 0, 0, np.real(sigma))

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.scale_radius)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.array(f) / (x ** 2 - 1)

        return np.where(f == 0, 0, np.real(kappa))

    def deflections_from_radii(self, radii):
        x = np.divide(radii, self.scale_radius)
        f = self.f_func(x)
        return np.real(
            4
            * self.kappa_s
            * self.scale_radius
            * np.divide((np.log(x / 2) + np.array(1 - f)), x)
        )
