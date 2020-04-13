import numpy as np
from astropy import cosmology
from astropy import units as u
from scipy.optimize import fsolve

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

class LensProfile:
    def __init__(self, z_l, z_s):
        self.z_s = z_s
        self.z_l = z_l

    def convergence_from_radii(self, radii):
        raise NotImplementedError()

    def deflection_angles_from_radii(self, radii):
        raise NotImplementedError()

    def shear_from_radii(self, radii):
        alpha = self.deflection_angles_from_radii(radii=radii)

        gamma = 0.5 * ((alpha / radii) - np.gradient(alpha, radii[:]))

        return gamma

    def tangential_eiginvalue_from_radii(self, radii):
        kappa = self.convergence_from_radii(radii=radii)
        gamma = self.shear_from_radii(radii=radii)

        return 1 - kappa - gamma

    def einstein_radius_in_arcseconds_from_radii(self, radii):
        lambda_t = self.tangential_eiginvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(lambda_t))

        return radii[index]

    def critical_surface_density_of_lens(self):
        D_s = cosmo.angular_diameter_distance(self.z_s).to(u.m)
        D_l = cosmo.angular_diameter_distance(self.z_l).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(self.z_l, self.z_s).to(u.m)

        sigma_crit = (np.divide(2.998e8 ** 2, 4 * np.pi * 6.674e-11) * np.divide(D_s, D_l * D_ls)).value

        return sigma_crit

    def einstein_mass_in_solar_masses_from_radii(self, radii):
        einstein_radius_rad = (self.einstein_radius_in_arcseconds_from_radii(radii=radii)*u.arcsec).to(u.rad)
        D_l = cosmo.angular_diameter_distance(self.z_l).to(u.m)
        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = self.critical_surface_density_of_lens()

        return (4 * np.pi * einstein_radius**2 * sigma_crit)/1.989e30



class SphericalPowerLaw(LensProfile):
    def __init__(self, einstein_radius, slope, z_s, z_l):
        LensProfile.__init__(self, z_l, z_s)
        self.einstein_radius = einstein_radius
        self.slope = slope

    def convergence_from_radii(self, radii):
        kappa = np.divide((3-self.slope), 2) * np.divide(self.einstein_radius, radii)**(self.slope-1)

        return kappa

    def deflection_angles_from_radii(self, radii):
        alpha = self.einstein_radius * np.divide(self.einstein_radius, radii)**(self.slope-2)

        return alpha

class Hernquist(LensProfile):
    def __init__(self, mass, r_eff, z_s, z_l):
        LensProfile.__init__(self, z_l, z_s)
        self.mass = mass
        self.r_eff = r_eff
        self.r_s = self.r_eff/1.8153
        self.rho_s = np.divide(self.mass, 2*np.pi*self.r_s**3)
        self.kappa_s = self.rho_s * self.r_s / self.critical_surface_density_of_lens()

    def density_from_radii(self, radii):
        x = np.array(radii / self.r_s)

        return np.divide(self.rho_s, x * (1+x)**3)

    def f_func(self, x):
        f = np.where(x < 1,
                     (np.divide(1, np.sqrt(1 - x ** 2)) * np.arctanh(np.sqrt(1 - x ** 2))), x)
        f = np.where(x > 1,
                     (np.divide(1, np.sqrt(x ** 2 - 1)) * np.arctan(np.sqrt(x ** 2 - 1))), f)
        return f

    def surface_mass_density_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return np.divide(self.rho_s * self.r_s, (x ** 2 - 1) ** 2) * (-3 + f*(2 + x**2))

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return np.divide(self.kappa_s, (x**2-1)**2)*(-3+f*(2+x**2))

    def deflection_angles_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)

        return 2 * self.kappa_s * self.r_s * np.divide(x * (1-f), x**2-1)







class NFW_Keeton(LensProfile):
    def __init__(self, m200, concentration, z_s, z_l):
        LensProfile.__init__(self, z_l, z_s)
        self.m200 = m200
        self.concentration = concentration
        self.r200 = ((self.m200 * u.solMass / (
                    1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))) ** (1. / 3))
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (4 * np.pi * self.r_s ** 3 * (
                    np.log(1. + self.concentration) - self.concentration / (1. + self.concentration)))
        self.kappa_s = self.rho_s * self.r_s / self.critical_surface_density_of_lens()

    def density_from_radii(self, radii):
        x = np.array(radii / self.r_s)

        return np.divide(self.rho_s, x * (1+x)**2)

    def f_func(self, x):
        f = np.where(x < 1,
                     (np.divide(1, np.sqrt(1 - x ** 2)) * np.arctanh(np.sqrt(1 - x ** 2))), x)
        f = np.where(x > 1,
                     (np.divide(1, np.sqrt(x ** 2 - 1)) * np.arctan(np.sqrt(x ** 2 - 1))), f)
        return f

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.divide(1 - np.array(f), x ** 2 - 1)

        return kappa

    def deflection_angles_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        alpha = 4 * self.kappa_s * self.r_s * np.divide((np.log(x/2) + np.array(f)), x)

        return alpha


class NFW_Bartelmann(LensProfile):
    def __init__(self, m200, concentration, z_s, z_l):
        LensProfile.__init__(self, z_l, z_s)
        self.m200 = m200
        self.concentration = concentration
        self.r200 = ((self.m200*u.solMass / (1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))) ** (1. / 3))
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (4*np.pi*self.r_s**3*(np.log(1.+self.concentration)-self.concentration/(1.+self.concentration)))
        self.kappa_s = self.rho_s * self.r_s / self.critical_surface_density_of_lens()

    def f_func(self, x):
        f = np.where(x > 1,
                     1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1), x)
        f = np.where(x < 1,
                     1 - 2 * np.arctanh(np.sqrt((1 - x) / (x + 1))) / np.sqrt(1 - x ** 2), f)
        return f

    def density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        return np.divide(self.rho_s, x * (1+x)**2)

    def surface_mass_density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        return 2*self.rho_s*self.r_s*f / (x**2 - 1)

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.array(f) / (x ** 2 - 1)
        return kappa

    def deflection_angles_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        alpha = 4 * self.kappa_s * self.r_s * np.divide(np.log(x/2) + np.array(1-f), x)

        return alpha

class NFW_Hilbert(LensProfile):
    def __init__(self, m200, concentration, z_s, z_l):
        LensProfile.__init__(self, z_l, z_s)
        self.m200 = m200
        self.concentration = concentration
        self.r200 = ((self.m200*u.solMass / (1.333 * np.pi * 200 * cosmo.critical_density(self.z_l).to("Msun/kpc**3"))) ** (1. / 3))
        self.r_s = self.r200.value / self.concentration
        self.rho_s = self.m200 / (4*np.pi*self.r_s**3*(np.log(1.+self.concentration)-self.concentration/(1.+self.concentration)))
        self.kappa_s = self.rho_s * self.r_s / self.critical_surface_density_of_lens()

    def f_func(self, x):
        f = np.where(x > 1,
                     1 - 2 * np.arctan(np.sqrt((x - 1) / (x + 1))) / np.sqrt(x ** 2 - 1), x)
        f = np.where(x < 1,
                     1 - (1/np.sqrt(1-x**2))*np.arccosh(1/x), f)
        return f

    def density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        return np.divide(self.rho_s, x * (1+x)**2)

    def surface_mass_density_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        return 2*self.rho_s*self.r_s*f / (x**2 - 1)

    def convergence_from_radii(self, radii):
        x = np.array(radii / self.r_s)
        f = self.f_func(x)
        kappa = 2 * self.kappa_s * np.array(f) / (x ** 2 - 1)
        return kappa

    def deflection_angles_from_radii(self, radii):
        x = np.divide(radii, self.r_s)
        f = self.f_func(x)
        alpha = 4 * self.kappa_s * self.r_s * np.divide((np.log(x/2) + np.array(1-f)), x)

        return alpha


class CombinedProfile(LensProfile):
    def __init__(self, mass_profiles, z_l, z_s):
        LensProfile.__init__(self, z_l, z_s)
        self.mass_profiles = mass_profiles

    def density_from_radii(self, radii):
        return sum([profile.density_from_radii(radii=radii) for profile in self.mass_profiles])

    def convergence_from_radii(self, radii):
        return sum([profile.convergence_from_radii(radii=radii) for profile in self.mass_profiles])

    def deflection_angles_from_radii(self, radii):
        return sum([profile.deflection_angles_from_radii(radii=radii) for profile in self.mass_profiles])

    def mask_radial_range_from_radii(self, lower_bound, upper_bound, radii):
        einstein_radius = self.einstein_radius_in_arcseconds_from_radii(radii=radii)

        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound * einstein_radius)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound * einstein_radius)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1

        return weights

    def mask_two_radial_ranges_from_radii(self, lower_bound_1, upper_bound_1, lower_bound_2, upper_bound_2, radii):
        einstein_radius = self.einstein_radius_in_arcseconds_from_radii(radii=radii)

        index1 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound_1 * einstein_radius)))
        index2 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound_1 * einstein_radius)))
        index3 = np.argmin(np.abs(np.array(radii) - (radii[0] + lower_bound_2 * einstein_radius)))
        index4 = np.argmin(np.abs(np.array(radii) - (radii[0] + upper_bound_2 * einstein_radius)))
        weights = np.zeros(len(radii))
        weights[index1:index2] = 1
        weights[index3:index4] = 1

        return weights

    def best_fit_power_law_convergence_coefficients_from_mask_and_radii(self, mask, radii):

        kappa = self.convergence_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(kappa), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def best_fit_power_law_slope_with_error_from_mask_and_radii(self, mask, radii):
        coeff, error = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[:, 0]

        slope = np.abs(coeff-1)

        return np.array([slope, error])

    def best_fit_power_law_convergence_from_mask_and_radii(self, mask, radii):
        coeffs = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask)[0]

        poly = np.poly1d(coeffs)

        best_fit = lambda radius: np.exp(poly(np.log(radius)))

        return best_fit(radii)

    def best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(self, mask, radii):

        normalization, error = self.best_fit_power_law_convergence_coefficients_from_mask_and_radii(
            radii=radii, mask=mask
        )[:, 1]

        slope = self.best_fit_power_law_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius = np.exp(np.divide(normalization-np.log(np.divide(3-slope, 2)), slope - 1))

        return np.array([einstein_radius, error])

    def best_fit_einstein_mass_in_solar_masses_from_mask_and_radii_and_redshifts(
            self, radii, mask, z_s, z_l
    ):
        einstein_radius_rad = (self.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(
            radii=radii, mask=mask)*u.arcsec).to(u.rad)
        D_l = cosmo.angular_diameter_distance(z_l).to(u.m)

        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = self.critical_surface_density_of_lens_from_redshifts(z_s=z_s, z_l=z_l)

        return (4 * np.pi * einstein_radius**2 * sigma_crit)/1.989e30

    def best_fit_power_law_deflection_angles_coefficients_from_mask_and_radii(
            self, mask, radii
    ):
        einstein_radius = self.einstein_radius_in_arcseconds_from_radii(radii=radii)
        alpha = self.deflection_angles_from_radii(radii=radii)

        coeffs, cov = np.polyfit(np.log(radii), np.log(alpha), w=mask, deg=1, cov=True)

        error = np.sqrt(np.diag(cov))

        return np.array([coeffs, error])

    def best_fit_power_law_deflection_angles_from_mask_and_radii(
            self, mask, radii
    ):
        coeffs = self.best_fit_power_law_deflection_angles_coefficients_from_mask_and_radii(
            radii=radii, mask=mask)[0]

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
        )[1]

        slope = self.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        einstein_radius = np.exp(np.divide(normalization-np.log(np.divide(3-slope, 2)), slope - 1))

        return einstein_radius

    def best_fit_einstein_mass_in_solar_masses_via_deflection_angles_from_mask_and_radii_and_redshifts(
            self, radii, mask, z_s, z_l
    ):
        einstein_radius_rad = (self.best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
            radii=radii, mask=mask)*u.arcsec).to(u.rad)

        D_s = cosmo.angular_diameter_distance(z_s).to(u.m)
        D_l = cosmo.angular_diameter_distance(z_l).to(u.m)
        D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m)
        einstein_radius = (einstein_radius_rad * D_l).value

        sigma_crit = (np.divide(2.998e8**2, 4*np.pi*6.674e-11) * np.divide(D_s, D_l * D_ls)).value

        return (4 * np.pi * einstein_radius**2 * sigma_crit)/1.989e30


















