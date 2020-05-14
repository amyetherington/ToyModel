import numpy as np
import pytest
from astropy import cosmology
from one_d_code import one_d_profiles as profiles

cosmo = cosmology.FlatLambdaCDM(H0=70, Om0=0.3)


def convergence_via_deflection_angles_from_profile_and_radii(profile, radii):

    alpha = profile.deflection_angles_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa


class TestSphericalPowerLaw:
    def test__convergence_from_deflections_and_analytic(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)
        ## include some interpolation in convergence from deflection angles
        # so don't need to have such a finely sampled radii grid, this test will fail if > 0.002

        kappa_analytic = power_law.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=power_law, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_isothermal_equal_to_analytic_formula(self):
        isothermal = profiles.SphericalPowerLaw(
            einstein_radius=1, slope=2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_isothermal = isothermal.convergence_from_radii(radii=radii)

        kappa_isothermal_analytic = np.divide(1 * isothermal.einstein_radius, 2 * radii)

        assert kappa_isothermal == pytest.approx(kappa_isothermal_analytic, 1e-4)

    def test__convergence_equals_surface_mass_density_divided_by_sigma_crit(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, z_s=0.8, z_l=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa = power_law.convergence_from_radii(radii=radii)
        sigma_crit = power_law.critical_surface_density_of_lens
        rho = power_law.surface_mass_density_from_radii(radii=radii)

        kappa_via_sigma = rho / sigma_crit

        assert kappa == pytest.approx(kappa_via_sigma, 1e-3)

    def test__convergence_values_correct(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, z_s=0.8, z_l=0.3
        )

        kappa = power_law.convergence_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert kappa == pytest.approx(np.array([1.298, 0.857, 0.672]), 1e-3)

    def test__deflection_angle_values_correct(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, z_s=0.8, z_l=0.3
        )

        alpha = power_law.deflection_angles_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert alpha == pytest.approx(np.array([0.927, 1.224, 1.439]), 1e-3)


class TestHernquist:
    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        rho = Hernquist.surface_mass_density_from_radii(radii=radii)
        kappa = Hernquist.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / Hernquist.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)


class TestNFW:
    def test__f_function_gives_correct_values_given_different_values_of_x(self):
        NFW = profiles.NFW_Keeton(m200=2.5e12, scale_radius=3.2)
        ## x < 1
        assert NFW.f_func(x=np.array([0.25, 0.5, 0.75])) == pytest.approx(
            np.array([2.131, 1.521, 1.202]), 1e-3
        )
        ## x > 1
        assert NFW.f_func(x=np.array([1.25, 1.5, 1.75])) == pytest.approx(
            np.array([0.858, 0.752, 0.670]), 1e-3
        )
        ## x=1
        assert NFW.f_func(x=np.array([1])) == pytest.approx(np.array([1]), 1e-3)

    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        NFW = profiles.NFW_Bartelmann(m200=2.5e12, concentration=3.5, z_l=0.3, z_s=0.8)
        radii = np.arange(0.2, 30, 0.002)

        rho = NFW.surface_mass_density_from_radii(radii=radii)
        kappa = NFW.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / NFW.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)

    def test__convergence_Keeton_from_deflections_and_analytic(self):
        NFW_Keeton = profiles.NFW_Keeton(
            m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Keeton.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Keeton, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        ##lower assertion as deflection angles not as numerically stable
        assert mean_error < 1e-1

    def test__convergence_Hilbert_from_deflections_and_analytic(self):
        NFW_Hilbert = profiles.NFW_Hilbert(
            m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Hilbert.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Hilbert, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-3

    def test__convergence_Bartelmann_from_deflections_and_analytic(self):
        NFW_Bartelmann = profiles.NFW_Bartelmann(
            m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3
        )
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Bartelmann.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Bartelmann, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-3

    def test__convergence_gives_correct_values_given_different_values_of_x(self):
        NFW = profiles.NFW(scale_radius=2.15, kappa_s=0.2)
        ## x < 1
        kappa_x_small = NFW.convergence_from_radii(radii=[1.2, 1.6, 2.0])
        assert kappa_x_small == pytest.approx(np.array([0.2504, 0.1867, 0.1453]), 1e-3)
        ## x > 1
        kappa_x_large = NFW.convergence_from_radii(radii=[2.2, 2.6, 3.0])
        assert kappa_x_large == pytest.approx(np.array([0.1297, 0.1054, 0.0874]), 1e-3)
        ## x=1
        kappa_x_one = NFW.convergence_from_radii(radii=[NFW.scale_radius])
        assert kappa_x_one == pytest.approx(np.array([0]), 1e-3)