from one_d_code import one_d_profiles as profiles
from one_d_code import combined_profiles as cp
import numpy as np
from scipy import integrate
import pytest


def convergence_via_deflection_angles_from_profile_and_radii(profile, radii):

    alpha = profile.deflection_angles_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa





class TestCombinedProfile:

    # TODO : This is new

    def test__effective_radius_from_combined_profile(self):

       # combined = cp.CombinedProfile()

      #  with pytest.raises(ValueError):
      #      combined.effective_radius

        hernquist = profiles.Hernquist(effective_radius=1.0, mass=100.0, z_s=0.5, z_l=1.0)

        combined = cp.CombinedProfile(profiles=[hernquist])

        assert combined.effective_radius == 1.0


    def test__convergence_from_deflections_and_analytic(self):
        NFW = profiles.NFW(kappa_s=0.15, scale_radius=2.5)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)
        combined = cp.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.002)
        ## include some interpolation in convergence from deflection angles???
        # so don't need to have such a finely sampled radii grid, this test will fail if > 0.002

        kappa_analytic = combined.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(profile=combined, radii=radii)

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_equals_sum_of_individual_profile_convergence(self):
        NFW = profiles.NFW(kappa_s=0.15, scale_radius=2.5)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)

        combined = cp.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.002)

        kappa_combined = combined.convergence_from_radii(radii=radii)
        kappa_NFW = NFW.convergence_from_radii(radii=radii)
        kappa_pl = power_law.convergence_from_radii(radii=radii)
        kappa_sum = kappa_NFW + kappa_pl

        assert kappa_combined == pytest.approx(kappa_sum, 1e-4)

    def test__deflection_angles_equals_sum_of_individual_profile_deflection_angles(self):
        NFW = profiles.NFW(kappa_s=0.15, scale_radius=2.5)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)

        combined = cp.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.002)

        alpha_combined = combined.deflection_angles_from_radii(radii=radii)
        alpha_NFW = NFW.deflection_angles_from_radii(radii=radii)
        alpha_pl = power_law.deflection_angles_from_radii(radii=radii)

        assert alpha_combined == pytest.approx(alpha_NFW + alpha_pl, 1e-4)

    def test__best_fit_einstein_radius_from_intercept_equal_to_from_kappa(self):
        NFW = profiles.NFW(kappa_s=0.15, scale_radius=2.5)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)
        combined = cp.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(radii=radii, lower_bound=0.5, upper_bound=1.5)

        einstein_radius_c = combined.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.best_fit_power_law_convergence_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.best_fit_power_law_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius_k = np.mean(radii * np.divide(2 * kappa, 3 - slope)**(np.divide(1, slope-1)))

        assert einstein_radius_c[0] == pytest.approx(einstein_radius_k, 1e-3)

    def test__best_fit_einstein_radius_via_deflection_angles_from_intercept_equal_to_from_kappa(self):
        NFW = profiles.NFW(kappa_s=0.15, scale_radius=2.5)
        power_law = profiles.SphericalPowerLaw(einstein_radius=1.8, slope=1.7)
        combined = cp.CombinedProfile(mass_profiles=[NFW, power_law])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(radii=radii, lower_bound=0.5, upper_bound=1.5)

        einstein_radius_c = combined.best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        einstein_radius_k = np.mean(radii * np.divide(2 * kappa, 3 - slope)**(np.divide(1, slope-1)))

        assert einstein_radius_c == pytest.approx(einstein_radius_k, 1e-3)




