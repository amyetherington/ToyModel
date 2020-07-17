import os
import numpy as np
import pytest
from astropy import cosmology
from scipy import integrate


import lens1d as l1d

cosmo = cosmology.Planck15

directory = os.path.dirname(os.path.realpath(__file__))

# TODO : A lot of the tests below seem specific to the profile (power law / nfw / hernquist). It may make sense to
# TODO : move them to test_oned.py? I get the functions are in abstract.py, but its fine to test these functions elsewhere!

# TODO : If we're the functions are generically testing a method, there isn't really a need to test it for every
# TODO : individal profile. This is just making more unit tests which take more time to manage, so feel free to ddelete
# TODO : a lot of the tests below (unless they are testing specifc parts of the profile, in which case move them to test_oned.py

# TODO : Either way, its super weird for the 'abstract' profile to have specific references to profiles in the test names
# TODO : stucture. You could also condense the functions for each profile... I'll put ane example belwo.


class TestShearAndEigenvalue:
    def test__shear_isothermal_sphere_equals_analytic(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = power_law.einstein_radius / (2 * radii)

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4

    def test__shear_isothermal_sphere_equals_convergence(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        kappa_isothermal = power_law.convergence_from_radii(radii=radii)

        mean_error = np.average(shear_isothermal - kappa_isothermal)

        assert mean_error < 1e-4

    def test__shear_spherical_power_law_equals_analytic(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = np.divide(power_law.slope - 1, 2) * (
            np.divide(power_law.einstein_radius, radii) ** (power_law.slope - 1)
        )

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4

    def test__power_law_tangential_eigenvalue_equal_to_zero_at_einstein_radius(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(np.array(radii) - (power_law.einstein_radius)))

        assert eigenvalue[index] < 1e-4

    # TODO : These 3 tets are testing the same functionality using different profiles. I'd either reduce them to one
    # TODO : profile, or if you don't want to losse the full test put them in one unit tests. I've put an example at the bottom.

    def test__power_law_tangential_eigenvalue_equal_to_from_alpha(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.8, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.001)

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        alpha = power_law.deflections_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha / radii

        mean_error = np.mean(eigenvalue_alpha - eigenvalue)

        assert mean_error < 1e-4

    def test__hernquist_tangential_eigenvalue_equal_to_from_alpha(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = Hernquist.tangential_eigenvalue_from_radii(radii=radii)

        alpha = Hernquist.deflections_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha / radii

        mean_error = np.mean(eigenvalue_alpha - eigenvalue)

        assert mean_error < 1e-4

    def test__nfw_tangential_eigenvalue_equal_to_from_alpha(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=0.8)

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = nfw.tangential_eigenvalue_from_radii(radii=radii)

        alpha = nfw.deflections_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha / radii

        mean_error = np.mean(eigenvalue_alpha - eigenvalue)

        assert mean_error < 1e-4

    # TODO : By writing the test in this way, we are testing the 'abstract' module but not explicitly thinking about
    # TODO : the specific profiles until the inner workings of the test. This will reduce the number of unit tests
    # TODO : overall a lot.

    def test__example__tangential_eigenvalue_equal_to_from_alpha(self):

        radii = np.arange(0.2, 3, 0.001)

        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.8, redshift_source=0.8, redshift_lens=0.3
        )

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        alpha = power_law.deflections_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha / radii

        mean_error = np.mean(eigenvalue_alpha - eigenvalue)

        assert mean_error < 1e-4

        nfw = l1d.NFWHilbert(mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=0.8)

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = nfw.tangential_eigenvalue_from_radii(radii=radii)

        alpha = nfw.deflections_from_radii(radii=radii)

        eigenvalue_alpha = 1 - alpha / radii

        mean_error = np.mean(eigenvalue_alpha - eigenvalue)

        assert mean_error < 1e-4


class TestMassAndSurfaceMassDensity:
    def test__critical_surface_mass_density_correct_value(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )

        sigma_crit = power_law.critical_surface_density_of_lens

        assert sigma_crit == pytest.approx(3076534993.9914, 1e-4)

    def test__power_law_average_density_inside_einstein_radius_equal_to_sigma_crit(
        self
    ):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )

        sigma_crit = power_law.critical_surface_density_of_lens

        integrand = (
            lambda r: 2 * np.pi * r * power_law.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, power_law.einstein_radius)[0] / (
            np.pi * power_law.einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__hernquist_average_density_inside_einstein_radius_equal_to_sigma_crit(
        self
    ):
        Hernquist = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        radii = np.arange(0.01, 10, 0.001)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = Hernquist.critical_surface_density_of_lens

        integrand = (
            lambda r: 2 * np.pi * r * Hernquist.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__nfw_average_density_inside_einstein_radius_equal_to_sigma_crit(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.6, redshift_source=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        sigma_crit = nfw.critical_surface_density_of_lens

        integrand = (
            lambda r: 2 * np.pi * r * nfw.surface_mass_density_from_radii(radii=r)
        )

        av_density = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_density == pytest.approx(sigma_crit, 1e-3)

    def test__power_law__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        einstein_mass = power_law.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = power_law.two_dimensional_mass_enclosed_within_radii(
            power_law.einstein_radius
        )

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-4)

    def test__hernquist__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )
        radii = np.arange(0.2, 100, 0.001)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        einstein_mass = Hernquist.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = Hernquist.two_dimensional_mass_enclosed_within_radii(
            einstein_radius
        )

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-3)

    def test__nfw__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.6, redshift_source=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        einstein_mass = nfw.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = nfw.two_dimensional_mass_enclosed_within_radii(
            einstein_radius
        )

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-3)

    def test__hernquist__mass_within_effective_radius_equal_to_half_total_two_d_mass(
        self
    ):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )

        mass_2d = Hernquist.two_dimensional_mass_enclosed_within_radii(
            radii=Hernquist.effective_radius
        )

        assert mass_2d == pytest.approx(Hernquist.mass * 0.5, 1e-3)

    def test__hernquist__mass_within_half_mass_radius_equal_to_half_total_three_d_mass(
        self
    ):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )

        mass_3d = Hernquist.three_dimensional_mass_enclosed_within_radii(
            radii=Hernquist.half_mass_radius
        )

        assert mass_3d == pytest.approx(Hernquist.mass * 0.5, 1e-3)


class TestEinsteinRadius:
    def test__mean_power_law_convergence_within_einstein_radius_equal_to_one(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2.3, redshift_source=0.8, redshift_lens=0.3
        )

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, power_law.einstein_radius)[0] / (
            np.pi * power_law.einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_power_law_convergence_within_calculated_einstein_radius_equal_to_one(
        self
    ):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.6, slope=2.3, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        einstein_radius = power_law.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_Hernquist_convergence_within_einstein_radius_equal_to_one(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )
        radii = np.arange(0.2, 100, 0.001)

        integrand = lambda r: 2 * np.pi * r * Hernquist.convergence_from_radii(radii=r)

        einstein_radius = Hernquist.einstein_radius_in_kpc_from_radii(radii=radii)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_NFW_convergence_within_einstein_radius_equal_to_one(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.6, redshift_source=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * nfw.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_combined_convergence_within_einstein_radius_equal_to_one(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.5, redshift_source=1.0)
        Hernquist = l1d.Hernquist(
            mass=2e11, effective_radius=5, redshift_lens=0.5, redshift_source=1.0
        )
        combined = l1d.CombinedProfile([nfw, Hernquist])

        radii = np.arange(0.1, 8, 0.001)

        einstein_radius = combined.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * combined.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)


class TestFFunction:
    def test__f_func_correct_values(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )

        f = Hernquist.f_func(x=np.array([0.5, 1, 1.5]))

        assert f == pytest.approx(np.array([-0.520691993, 0, 0.2477253115]))


class TestSlopeFromLensing:
    def test__convergence_at_einstein_radius_less_than_one(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = power_law.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1

        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = Hernquist.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1

        nfw = l1d.NFWHilbert(mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=0.8)
        radii = np.arange(0.2, 30, 0.002)

        kappa_ein = nfw.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1
