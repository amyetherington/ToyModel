import os
import numpy as np
import pytest

from astropy import cosmology
from autoconf import conf

workspace_path = '{}'.format(os.path.dirname(os.path.realpath(__file__)))
conf.instance = conf.Config(config_path=f"{workspace_path}/config")

from autogalaxy.util import cosmology_util
from autogalaxy.profiles import mass_profiles as mp
from one_d_code import one_d_profiles as profiles
from one_d_code import combined_profiles as cp
from scipy import integrate

cosmo = cosmology.Planck15

directory = os.path.dirname(os.path.realpath(__file__))


@pytest.fixture(scope="session", autouse=True)
def do_something():
    print("{}/config/".format(directory))
    conf.instance = conf.Config("{}/config/".format(directory))


class TestShearAndEigenvalue:
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

    def test__power_law_tangential_eigenvalue_equal_to_zero_at_einstein_radius(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(np.array(radii) - (power_law.einstein_radius)))

        assert eigenvalue[index] < 1e-4

    def test__power_law_tangential_eigenvalue_equal_to_from_alpha(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.8, z_s=0.8, z_l=0.3
        )

        radii = np.arange(0.2, 3, 0.001)

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        alpha = power_law.deflection_angles_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha/radii

        mean_error = np.mean(eigenvalue_alpha-eigenvalue)

        assert mean_error < 1e-4

    def test__hernquist_tangential_eigenvalue_equal_to_from_alpha(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=1.8153, z_l=0.3, z_s=0.8
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = Hernquist.tangential_eigenvalue_from_radii(radii=radii)

        alpha = Hernquist.deflection_angles_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha/radii

        mean_error = np.mean(eigenvalue_alpha-eigenvalue)

        assert mean_error < 1e-4

    def test__nfw_tangential_eigenvalue_equal_to_from_alpha(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e12, z_l=0.3, z_s=0.8)

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = NFW.tangential_eigenvalue_from_radii(radii=radii)

        alpha = NFW.deflection_angles_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha/radii

        mean_error = np.mean(eigenvalue_alpha-eigenvalue)

        assert mean_error < 1e-4


class TestMassAndSurfaceMassDensity:
    def test__critical_surface_mass_density_correct_value(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, z_s=0.8, z_l=0.3
        )

        sigma_crit = power_law.critical_surface_density_of_lens

        assert sigma_crit == pytest.approx(3076534993.9914, 1e-4)

    def test__power_law_average_density_inside_einstein_radius_equal_to_sigma_crit(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, z_s=0.8, z_l=0.3
        )

        sigma_crit = power_law.critical_surface_density_of_lens

        integrand = (
            lambda r: 2* np.pi * r * power_law.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, power_law.einstein_radius)[0]/ (
            np.pi * power_law.einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__hernquist_average_density_inside_einstein_radius_equal_to_sigma_crit(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )

        radii = np.arange(0.01, 10, 0.001)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = Hernquist.critical_surface_density_of_lens

        integrand = (
            lambda r: 2 * np.pi * r * Hernquist.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, einstein_radius)[0]/ (
            np.pi * einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__nfw_average_density_inside_einstein_radius_equal_to_sigma_crit(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e13, z_l=0.6, z_s=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = NFW.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = NFW.critical_surface_density_of_lens

        integrand = (
            lambda r: 2 * np.pi * r * NFW.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, einstein_radius)[0]/ (
            np.pi * einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__power_law__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        einstein_mass = power_law.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = power_law.two_dimensional_mass_enclosed_within_radii(power_law.einstein_radius)

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-4)

    def test__hernquist__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )
        radii = np.arange(0.2, 100, 0.001)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        einstein_mass = Hernquist.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = Hernquist.two_dimensional_mass_enclosed_within_radii(einstein_radius)

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-3)

    def test__nfw__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e13, z_l=0.6, z_s=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = NFW.einstein_radius_in_kpc_from_radii(radii=radii)

        einstein_mass = NFW.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = NFW.two_dimensional_mass_enclosed_within_radii(einstein_radius)

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-3)

    def test__hernquist__mass_within_effective_radius_equal_to_half_total_two_d_mass(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )

        mass_2d = Hernquist.two_dimensional_mass_enclosed_within_radii(radii=Hernquist.effective_radius)

        assert mass_2d == pytest.approx(Hernquist.mass*0.5, 1e-3)

    def test__hernquist__mass_within_half_mass_radius_equal_to_half_total_three_d_mass(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )

        mass_3d = Hernquist.three_dimensional_mass_enclosed_within_radii(radii=Hernquist.half_mass_radius)

        assert mass_3d == pytest.approx(Hernquist.mass*0.5, 1e-3)


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

    def test__mean_Hernquist_convergence_within_einstein_radius_equal_to_one(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )
        radii = np.arange(0.2, 100, 0.001)

        integrand = lambda r: 2 * np.pi * r * Hernquist.convergence_from_radii(radii=r)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_NFW_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e13, z_l=0.6, z_s=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = NFW.einstein_radius_in_kpc_from_radii(radii=radii)


        integrand = lambda r: 2 * np.pi * r * NFW.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_combined_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e13, z_l=0.5, z_s=1.0)
        Hernquist = profiles.Hernquist(mass=2e11, effective_radius=5, z_l=0.5, z_s=1.0)
        combined = cp.CombinedProfile([NFW, Hernquist])

        radii = np.arange(0.1, 8, 0.001)

        einstein_radius = combined.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * combined.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)


class TestF:
    def test__f_func_correct_values(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )

        f = Hernquist.f_func(x=np.array([0.5, 1, 1.5]))

        assert f == pytest.approx(np.array([-0.520691993, 0, 0.2477253115]))


class TestSlopeFromLensing:
    def test__xi_two_equal_to_zero_for_sis(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        xi_two = power_law.xi_two(radii=radii)

        assert xi_two == pytest.approx(0, 1e-3)

    def test__power_law_convergence_at_einstein_radius_less_than_one(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = power_law.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1

    def test__hernquist_convergence_at_einstein_radius_less_than_one(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = Hernquist.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1

    def test__nfw_convergence_at_einstein_radius_less_than_one(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e12, z_l=0.3, z_s=0.8)
        radii = np.arange(0.2, 30, 0.002)

        kappa_ein = NFW.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1

    def test_slope_equal_to_power_law_input(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        slope_lensing = power_law.slope_via_lensing(radii=radii)

        assert slope_lensing == pytest.approx(power_law.slope, 1e-4)


class TestCompareAutoLens:
    def test_sigma_crit_equal_to_autolens_function_in_desired_units(self):

        NFW = profiles.NFW_Hilbert(m200=2.5e10, z_l=0.3, z_s=0.8)
        sigma_crit = NFW.critical_surface_density_of_lens

        autolens_sigma_crit = cosmology_util.critical_surface_density_between_redshifts_from(
            redshift_1=0.8,
            redshift_0=0.3,
            unit_length="kpc",
            unit_mass="solMass",
            cosmology=cosmo,
        )

        assert sigma_crit == pytest.approx(autolens_sigma_crit, 1e-4)

    def test_sigma_crit_power_law_equal_to_autolens_function_in_desired_units(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, z_s=0.8, z_l=0.3
        )
        sigma_crit = power_law.critical_surface_density_of_lens

        autolens_sigma_crit = cosmology_util.critical_surface_density_between_redshifts_from(
            redshift_1=0.8,
            redshift_0=0.3,
            unit_length="kpc",
            unit_mass="solMass",
            cosmology=cosmo,
        )

        assert sigma_crit == pytest.approx(autolens_sigma_crit, 1e-4)

    def test_r200_equal_to_autolens_function_in_desired_units(self):

        autolens = mp.dark_mass_profiles.kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=2.5e14, redshift_source=1.0, redshift_object=0.5
        )
        NFW = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.5, z_s=1.0)
        print(NFW.rho_s)

        r200 = NFW.r200.value

        kappa_s = NFW.kappa_s

        scale_radius = NFW.r_s

        scale_radius_kpc = autolens[1]/cosmology_util.arcsec_per_kpc_from(redshift=0.5, cosmology=cosmo)

        assert r200 == pytest.approx(autolens[2], 1e-3)

        assert scale_radius == pytest.approx(scale_radius_kpc, 1e-3)

        assert kappa_s == pytest.approx(autolens[0], 1e-3)

    def test_NFW_einstein_radius_equal_to_autolens_in_desired_units(self):

        ## ONLY PASSSES FOR HIGH RESOLUTION AUTOLENS GRID

        NFW = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        radii = np.arange(0.01, 5, 0.001)

        einstein_radius = NFW.einstein_radius_in_kpc_from_radii(radii=radii)

        NFW_autolens = mp.dark_mass_profiles.SphericalNFWMCRLudlow(
            mass_at_200=2.5e14, redshift_object=0.3, redshift_source=1.0
        )

        einstein_radius_autolens = NFW_autolens.einstein_radius_in_units(
            unit_length="kpc", redshift_object=0.3, cosmology=cosmo
        )

        assert einstein_radius == pytest.approx(einstein_radius_autolens, 1e-2)
