import numpy as np
from astropy import cosmology
from astropy import units as u
from scipy.optimize import fsolve
from scipy import integrate

from one_d_code import lens_profile as lp

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)


class SphericalPowerLaw(lp.LensProfile):
    def __init__(self, einstein_radius, slope, z_s, z_l):
        super().__init__(z_l=z_l, z_s=z_s)
        self.einstein_radius = einstein_radius
        self.slope = slope

    def density_from_radii(self, radii):
        rho = np.divide(self.einstein_radius, radii**self.slope)

        return rho

    def surface_mass_density_from_radii(self, radii):
        rho = np.divide((3-self.slope), 2)*np.divide(1, radii**(self.slope-1))

        return rho

    def convergence_from_radii(self, radii):
        kappa = np.divide((3-self.slope), 2) * np.divide(self.einstein_radius, radii)**(self.slope-1)

        return kappa

    def deflection_angles_from_radii(self, radii):
        alpha = self.einstein_radius * np.divide(self.einstein_radius, radii)**(self.slope-2)

        return alpha


class Hernquist(lp.LensProfile):

    def __init__(self, mass, effective_radius, z_s, z_l):

        # TODO : With only one inheritance we can just use super to call inheritance. If you ever feel like you
        # TODO : *need* to do multiple inheritance ask me as its not a rabbit hall worth going down.

        super().__init__(z_l=z_l, z_s=z_s)

        self.mass = mass
        self.effective_radius = effective_radius
        self.r_s = self.effective_radius / 1.8153
        self.rho_s = np.divide(self.mass, 2 * np.pi * self.r_s ** 3)
        self.kappa_s = self.rho_s * self.r_s / self.critical_surface_density_of_lens

    def density_from_radii(self, radii):

        x = np.array(radii / self.r_s)

        return np.divide(self.rho_s, x * (1 + x) ** 3)

    def surface_mass_density_from_radii(self, radii):

        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return np.divide(self.rho_s * self.r_s, (x ** 2 - 1) ** 2) * (
            -3 + f * (2 + x ** 2)
        )

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return np.divide(self.kappa_s, (x ** 2 - 1) ** 2) * (-3 + f * (2 + x ** 2))

    def deflection_angles_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return 2 * self.kappa_s * self.r_s * np.divide(x * (1 - f), x ** 2 - 1)


class NFW_Keeton(lp.LensProfile):

    def __init__(self, m200, concentration, z_s, z_l):

        super().__init__(z_s=z_s, z_l=z_l)

        self.m200 = m200
        self.concentration = concentration
        self.r200 = (
            self.m200
            * u.solMass
            / (1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))
        ) ** (1.0 / 3)
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (
            4
            * np.pi
            * self.r_s ** 3
            * (
                np.log(1.0 + self.concentration)
                - self.concentration / (1.0 + self.concentration)
            )
        )
        self.kappa_s = np.divide(self.rho_s * self.r_s, self.critical_surface_density_of_lens)

    def density_from_radii(self, radii):

        x = np.array(radii / self.r_s)

        return np.divide(self.rho_s, x * (1 + x) ** 2)

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.divide(1 - np.array(f), x ** 2 - 1)

        return kappa

    def deflection_angles_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        alpha = (
            4 * self.kappa_s * self.r_s * np.divide((np.log(x / 2) + np.array(f)), x)
        )

        return alpha


class NFW_Bartelmann(lp.LensProfile):

    def __init__(self, m200, concentration, z_s, z_l):

        super().__init__(z_l=z_l, z_s=z_s)

        self.m200 = m200
        self.concentration = concentration
        self.r200 = (
            self.m200
            * u.solMass
            / (1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))
        ) ** (1.0 / 3)
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (
            4
            * np.pi
            * self.r_s ** 3
            * (
                np.log(1.0 + self.concentration)
                - self.concentration / (1.0 + self.concentration)
            )
        )

        self.kappa_s = np.divide(self.rho_s * self.r_s, self.critical_surface_density_of_lens)


    def f_func(self, x):
        f = np.where(
            x > 1,
            1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1),
            x,
        )
        f = np.where(
            x < 1,
            1 - 2 * np.arctanh(np.sqrt((1 - x) / (x + 1))) / np.sqrt(1 - x ** 2),
            f,
        )
        return f

    def density_from_radii(self, radii):

        x = np.divide(radii, self.r_s)

        return np.divide(self.rho_s, x * (1 + x) ** 2)

    def surface_mass_density_from_radii(self, radii):

        x = np.divide(radii, self.r_s)
        f = self.f_func(x)

        return 2 * self.rho_s * self.r_s * f / (x ** 2 - 1)

    def convergence_from_radii(self, radii):

        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = np.divide(2 * self.kappa_s, x ** 2 - 1) * np.array(f)

        return kappa

    def deflection_angles_from_radii(self, radii):

        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        alpha = (
            4 * self.kappa_s * self.r_s * np.divide(np.log(x / 2) + np.array(1 - f), x)
        )

        return alpha


class NFW_Hilbert(lp.LensProfile):

    def __init__(self, m200, concentration, z_s, z_l):

        super().__init__(z_l=z_l, z_s=z_s)

        self.m200 = m200
        self.concentration = concentration
        self.r200 = (
            self.m200
            * u.solMass
            / (1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))
        ) ** (1.0 / 3)
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (
            4
            * np.pi
            * self.r_s ** 3
            * (
                np.log(1.0 + self.concentration)
                - self.concentration / (1.0 + self.concentration)
            )
        )
        self.kappa_s = np.divide(self.rho_s * self.r_s, self.critical_surface_density_of_lens)

    def f_func(self, x):
        f = np.where(
            x > 1,
            1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1),
            x,
        )
        f = np.where(x < 1, 1 - (1 / np.sqrt(1 - x ** 2)) * np.arccosh(1 / x), f)
        return f

    def density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        return np.divide(self.rho_s, x * (1 + x) ** 2)

    def surface_mass_density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        return 2 * self.rho_s * self.r_s * f / (x ** 2 - 1)

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.array(f) / (x ** 2 - 1)
        return kappa

    def deflection_angles_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        alpha = (
            4
            * self.kappa_s
            * self.r_s
            * np.divide((np.log(x / 2) + np.array(1 - f)), x)
        )

        return alpha



