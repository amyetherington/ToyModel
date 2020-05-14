import os
import numpy as np
import pytest

from astropy import cosmology
from autoconf import conf
from autogalaxy.util import cosmology_util
from one_d_code import one_d_profiles as profiles
from scipy import integrate

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/config/".format(directory))
    conf.instance = conf.Config("{}/config/".format(directory))


class TestShear:
    def test__shear_isothermal_sphere_equals_analytic(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = power_law.einstein_radius / (2 * radii)

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4

    def test__shear_isothermal_sphere_equals_convergence(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        kappa_isothermal = power_law.convergence_from_radii(radii=radii)

        mean_error = np.average(shear_isothermal - kappa_isothermal)

        assert mean_error < 1e-4

    def test__shear_spherical_power_law_equals_analytic(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = np.divide(power_law.slope - 1, 2) * (
            np.divide(power_law.einstein_radius, radii) ** (power_law.slope - 1)
        )

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4


class TestEinsteinRadius:
    def test__mean_power_law_convergence_within_einstein_radius_equal_to_one(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2.3, z_s=0.8, z_l=0.3
        )

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, power_law.einstein_radius)[0] / (
            np.pi * power_law.einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_Hernquist_convergence_within_einstein_radius_equal_to_one(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )
        radii = np.arange(0.2, 100, 0.001)

        integrand = lambda r: 2 * np.pi * r * Hernquist.convergence_from_radii(radii=r)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)
        print(einstein_radius)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_power_law_convergence_within_calculated_einstein_radius_equal_to_one(
        self
    ):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.6, slope=2.3, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        einstein_radius = power_law.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_NFW_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW_Bartelmann(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
        radii = np.arange(0.2, 30, 0.001)

        einstein_radius = NFW.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * NFW.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_combined_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW(kappa_s=0.2, scale_radius=2.2)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2.3)
        combined = profiles.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.001)

        einstein_radius = combined.einstein_radius_in_arcseconds_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * combined.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)


class TestCriticalSurfaceDensity:
    def test_equal_to_autolens_function_in_desired_units(self):

        # TODO : You can print the path the configs look for as follows (this is super useful so note it down somewhere)

        print(conf.instance.config_path)

        NFW = profiles.NFW_Bartelmann(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
        sigma_crit = NFW.critical_surface_density_of_lens

        autolens_sigma_crit = cosmology_util.critical_surface_density_between_redshifts_from_redshifts_and_cosmology(
            redshift_1=0.8, redshift_0=0.3, unit_length="kpc", unit_mass="solMass", cosmology=None
        )

        assert sigma_crit == pytest.approx(autolens_sigma_crit, 1e-4)
