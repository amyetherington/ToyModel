import os
import numpy as np
import pytest
from astropy import cosmology
from scipy import integrate


import lens1d as l1d

cosmo = cosmology.Planck15

directory = os.path.dirname(os.path.realpath(__file__))


class TestShearAndEigenvalue:
    def test__tangential_eigenvalue_equal_to_from_alpha(self):

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

        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = Hernquist.tangential_eigenvalue_from_radii(radii=radii)

        alpha = Hernquist.deflections_from_radii(radii=radii)

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

    def test__average_density_inside_einstein_radius_equal_to_sigma_crit(
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

    def test__mass_inside_einstein_radius_equal_to_einstein_mass(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        einstein_mass = power_law.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = power_law.two_dimensional_mass_enclosed_within_radii(
            power_law.einstein_radius
        )

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-4)

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

        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.6, redshift_source=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        einstein_mass = nfw.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mass_within_r_ein = nfw.two_dimensional_mass_enclosed_within_radii(
            einstein_radius
        )

        assert einstein_mass == pytest.approx(mass_within_r_ein, 1e-3)


class TestFFunction:
    def test__f_func_correct_values(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )

        f = Hernquist.f_func(x=np.array([0.5, 1.5]))

        assert f == pytest.approx(np.array([-0.520691993, 0.2477253115]))
