import one_d_profiles as profiles
import numpy as np
from scipy import integrate
import pytest


def convergence_via_deflection_angles_from_profile_and_radii(profile, radii):

    alpha = profile.deflection_angles_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa


class TestLensingProfile:
    def test__shear_isothermal_sphere_equals_analytic(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2)

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = power_law.einstein_radius/(2*radii)

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4

    def test__shear_isothermal_sphere_equals_convergence(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2)

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        kappa_isothermal = power_law.convergence_from_radii(radii=radii)

        mean_error = np.average(shear_isothermal - kappa_isothermal)

        assert mean_error < 1e-4

    def test__shear_spherical_power_law_equals_analytic(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2)

        radii = np.arange(0.2, 3, 0.002)

        shear_isothermal = power_law.shear_from_radii(radii=radii)
        shear_isothermal_analytic = np.divide(power_law.slope-1, 2)*(np.divide(power_law.einstein_radius, radii)**(power_law.slope-1))

        mean_error = np.average(shear_isothermal - shear_isothermal_analytic)

        assert mean_error < 1e-4

    def test__mean_power_law_convergence_within_einstein_radius_equal_to_one(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2.3)

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, power_law.einstein_radius)[0] / (np.pi * power_law.einstein_radius**2)

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_power_law_convergence_within_calculated_einstein_radius_equal_to_one(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.6, slope=2.3)

        radii = np.arange(0.2, 3, 0.002)

        einstein_radius = power_law.einstein_radius_in_arcseconds_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * power_law.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (np.pi * einstein_radius**2)

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_NFW_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW(kappa_s=0.2, scale_radius=2.2)
        radii = np.arange(0.2, 3, 0.001)

        einstein_radius = NFW.einstein_radius_in_arcseconds_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * NFW.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
                    np.pi * einstein_radius ** 2)

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__mean_combined_convergence_within_einstein_radius_equal_to_one(self):
        NFW = profiles.NFW(kappa_s=0.2, scale_radius=2.2)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=2.3)
        combined = profiles.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.001)

        einstein_radius = combined.einstein_radius_in_arcseconds_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * combined.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
                    np.pi * einstein_radius ** 2)

        assert av_kappa == pytest.approx(1, 1e-3)


class TestSphericalPowerLaw:
    def test__convergence_from_deflections_and_analytic(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)
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
        isothermal = profiles.SphericalPowerLaw(einstein_radius=1, slope=2)
        radii = np.arange(0.2, 3, 0.002)

        kappa_isothermal = isothermal.convergence_from_radii(radii=radii)

        kappa_isothermal_analytic = np.divide(1*isothermal.einstein_radius, 2*radii)

        assert kappa_isothermal == pytest.approx(kappa_isothermal_analytic, 1e-4)

    def test__convergence_values_correct(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.4, slope=1.6)

        kappa = power_law.convergence_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert kappa == pytest.approx(np.array([1.298, 0.857, 0.672]), 1e-3)

    def test__deflection_angle_values_correct(self):
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.4, slope=1.6)

        alpha = power_law.deflection_angles_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert alpha == pytest.approx(np.array([0.927, 1.224, 1.439]), 1e-3)


class TestNFW:
    def test__f_function_gives_correct_values_given_different_values_of_x(self):
        NFW = profiles.NFW_Keeton(m200=2.5e12, scale_radius=3.2)
        ## x < 1
        assert NFW.f_func(x=np.array([0.25, 0.5, 0.75])) == pytest.approx(np.array([2.131, 1.521, 1.202]), 1e-3)
        ## x > 1
        assert NFW.f_func(x=np.array([1.25, 1.5, 1.75])) == pytest.approx(np.array([0.858, 0.752, 0.670]), 1e-3)
        ## x=1
        assert NFW.f_func(x=np.array([1])) == pytest.approx(np.array([1]), 1e-3)

    def test__convergence_Keeton_from_deflections_and_analytic(self):
        NFW_Keeton = profiles.NFW_Keeton(m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3)
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Keeton.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Keeton, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        ##lower assertion as deflection angles not as numerically stable
        assert mean_error < 1e-1

    def test__convergence_Hilbert_from_deflections_and_analytic(self):
        NFW_Hilbert = profiles.NFW_Hilbert(m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3)
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Hilbert.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Hilbert, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-3

    def test__convergence_Bartelmann_from_deflections_and_analytic(self):
        NFW_Bartelmann = profiles.NFW_Bartelmann(m200=2.5e12, concentration=3.2, z_s=0.8, z_l=0.3)
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

class TestHernquist:





