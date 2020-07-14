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

    def test__surface_mass_density_values_correct(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, z_s=0.8, z_l=0.3
        )

        sigma = power_law.surface_mass_density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(np.array([3994429111, 2635340405, 2066245712]), 1e-3)

    def test__density_values_correct(self):
        power_law = profiles.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, z_s=0.8, z_l=0.3
        )

        sigma = power_law.density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(np.array([9326310116, 3076534994, 1608110342]), 1e-3)


class TestHernquist:
    def test__convergence_from_deflections_and_analytic(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_analytic = Hernquist.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=Hernquist, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=8.4, z_l=0.3, z_s=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        rho = Hernquist.surface_mass_density_from_radii(radii=radii)
        kappa = Hernquist.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / Hernquist.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)

    def test__convergence_values_correct(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=1.8153, z_l=0.3, z_s=0.8
        )

        kappa = Hernquist.convergence_from_radii(radii=np.array([0.5, 1, 1.5]))

        kappa_s = Hernquist.kappa_s

        assert kappa == pytest.approx(np.array([1.318168568, 0, 0.2219485564]), 1e-3)
        assert kappa_s == pytest.approx(1.758883964, 1e-3)

    def test__deflection_angle_values_correct(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=1.8153, z_l=0.3, z_s=0.8
        )

        alpha = Hernquist.deflection_angles_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert alpha == pytest.approx(np.array([1.2211165, 0, 1.045731397]), 1e-3)

    def test__surface_mass_density_values_correct(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=1.8153, z_l=0.3, z_s=0.8
        )

        sigma = Hernquist.surface_mass_density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(np.array([4053818282, 0, 682832509]), 1e-3)

    def test_density_values_correct(self):
        Hernquist = profiles.Hernquist(
            mass=3.4e10, effective_radius=1.8153, z_l=0.3, z_s=0.8
        )

        rho = Hernquist.density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert rho == pytest.approx(np.array([3206677372, 676408508, 230880770]), 1e-3)


class TestNFW:
    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        NFW = profiles.NFW_Hilbert(m200=2.5e12, z_l=0.3, z_s=0.8)
        radii = np.arange(0.2, 30, 0.002)

        rho = NFW.surface_mass_density_from_radii(radii=radii)
        kappa = NFW.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / NFW.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)

    def test__convergence_from_deflections_and_analytic(self):
        NFW_Hilbert = profiles.NFW_Hilbert(
            m200=2.5e12, z_s=0.8, z_l=0.3
        )
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFW_Hilbert.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=NFW_Hilbert, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-3

    def test__convergence_values_correct(self):
        NFW_Hilbert = profiles.NFW_Hilbert(
            m200=2.5e14, z_s=1.0, z_l=0.5
        )

        kappa = NFW_Hilbert.convergence_from_radii(radii=np.array([132.3960792, 264.79215844891064, 397.1882377]))

        kappa_s = NFW_Hilbert.kappa_s

        assert kappa == pytest.approx(np.array([0.1567668617, 0, 0.04475020184]), 1e-3)
        assert kappa_s == pytest.approx(0.1129027792225471, 1e-3)

    def test__surface_mass_density_values_correct(self):
        NFW_Hilbert = profiles.NFW_Hilbert(
            m200=2.5e14, z_s=1.0, z_l=0.5
        )

        sigma = NFW_Hilbert.surface_mass_density_from_radii(radii=np.array([132.3960792, 264.79215844891064, 397.1882377]))

        assert sigma == pytest.approx(np.array([470603968.6, 0, 134337210.4]), 1e-3)

    def test__density_values_correct(self):
        NFW_Hilbert = profiles.NFW_Hilbert(
            m200=2.5e14, z_s=1.0, z_l=0.5
        )
        rho_s = NFW_Hilbert.rho_s

        rho = NFW_Hilbert.density_from_radii(radii=np.array([132.3960792, 264.79215844891064, 397.1882377]))

        assert rho == pytest.approx(np.array([1137753.844, 319993.2686, 136530.4613]), 1e-3)
        assert rho_s == pytest.approx(1279973.074564, 1e-3)