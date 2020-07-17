import numpy as np
import pytest
from astropy import cosmology
from scipy import integrate


import autogalaxy as ag
import lens1d as l1d

cosmo = cosmology.Planck15


def convergence_via_deflections_from_profile_and_radii(profile, radii):

    alpha = profile.deflections_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa


class TestSphericalPowerLaw:
    def test__convergence_from_deflections_and_analytic(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_analytic = power_law.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflections_from_profile_and_radii(
            profile=power_law, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_isothermal_equal_to_analytic_formula(self):
        isothermal = l1d.SphericalPowerLaw(
            einstein_radius=1, slope=2, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_isothermal = isothermal.convergence_from_radii(radii=radii)

        kappa_isothermal_analytic = np.divide(1 * isothermal.einstein_radius, 2 * radii)

        assert kappa_isothermal == pytest.approx(kappa_isothermal_analytic, 1e-4)

    def test__convergence_equals_surface_mass_density_divided_by_sigma_crit(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=1.7, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa = power_law.convergence_from_radii(radii=radii)
        sigma_crit = power_law.critical_surface_density_of_lens
        rho = power_law.surface_mass_density_from_radii(radii=radii)

        kappa_via_sigma = rho / sigma_crit

        assert kappa == pytest.approx(kappa_via_sigma, 1e-3)

    def test__convergence_values_correct(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, redshift_source=0.8, redshift_lens=0.3
        )

        kappa = power_law.convergence_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert kappa == pytest.approx(np.array([1.298, 0.857, 0.672]), 1e-3)

    def test__deflection_angle_values_correct(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, redshift_source=0.8, redshift_lens=0.3
        )

        alpha = power_law.deflections_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert alpha == pytest.approx(np.array([0.927, 1.224, 1.439]), 1e-3)

    def test__surface_mass_density_values_correct(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, redshift_source=0.8, redshift_lens=0.3
        )

        sigma = power_law.surface_mass_density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(
            np.array([3994429111, 2635340405, 2066245712]), 1e-3
        )

    def test__density_values_correct(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, redshift_source=0.8, redshift_lens=0.3
        )

        sigma = power_law.density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(
            np.array([9326310116, 3076534994, 1608110342]), 1e-3
        )

    def test__compare_to_autogalaxy__sigma_crit_equal_in_desired_units(self):

        cosmo = cosmology.Planck15

        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.4, slope=1.6, redshift_source=0.8, redshift_lens=0.3
        )
        sigma_crit = power_law.critical_surface_density_of_lens

        autolens_sigma_crit = ag.util.cosmology.critical_surface_density_between_redshifts_from(
            redshift_1=0.8,
            redshift_0=0.3,
            unit_length="kpc",
            unit_mass="solMass",
            cosmology=cosmo,
        )

        assert sigma_crit == pytest.approx(autolens_sigma_crit, 1e-4)

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

    def test__shear_equals_analytic(self):
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

    def test__isoyhemral_tangential_eigenvalue_equal_to_zero_at_einstein_radius(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )

        radii = np.arange(0.2, 3, 0.002)

        eigenvalue = power_law.tangential_eigenvalue_from_radii(radii=radii)

        index = np.argmin(np.abs(np.array(radii) - power_law.einstein_radius))

        assert eigenvalue[index] < 1e-4

    def test__convergence_at_einstein_radius_less_than_one(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = power_law.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1


class TestHernquist:
    def test__convergence_from_deflections_and_analytic(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_analytic = Hernquist.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflections_from_profile_and_radii(
            profile=Hernquist, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        rho = Hernquist.surface_mass_density_from_radii(radii=radii)
        kappa = Hernquist.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / Hernquist.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)

    def test__convergence_values_correct(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        kappa = Hernquist.convergence_from_radii(radii=np.array([0.5,  1.5]))

        kappa_s = Hernquist.kappa_s

        assert kappa == pytest.approx(np.array([1.318168568,  0.2219485564]), 1e-3)
        assert kappa_s == pytest.approx(1.758883964, 1e-3)

    def test__deflection_angle_values_correct(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        alpha = Hernquist.deflections_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert alpha == pytest.approx(np.array([1.2211165, 0, 1.045731397]), 1e-3)

    def test__surface_mass_density_values_correct(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        sigma = Hernquist.surface_mass_density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert sigma == pytest.approx(np.array([4053818282, 0, 682832509]), 1e-3)

    def test_density_values_correct(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=1.8153, redshift_lens=0.3, redshift_source=0.8
        )

        rho = Hernquist.density_from_radii(radii=np.array([0.5, 1, 1.5]))

        assert rho == pytest.approx(np.array([3206677372, 676408508, 230880770]), 1e-3)

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

    def test__convergence_at_einstein_radius_less_than_one(self):
        Hernquist = l1d.Hernquist(
            mass=3.4e10, effective_radius=8.4, redshift_lens=0.3, redshift_source=0.8
        )
        radii = np.arange(0.2, 3, 0.002)

        kappa_ein = Hernquist.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1


class TestNFWHilbert:
    def test__convergence_equal_to_surface_mass_density_divided_by_sigma_crit(self):
        NFW = l1d.NFWHilbert(mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=0.8)
        radii = np.arange(0.2, 30, 0.002)

        rho = NFW.surface_mass_density_from_radii(radii=radii)
        kappa = NFW.convergence_from_radii(radii=radii)
        kappa_via_sigma = rho / NFW.critical_surface_density_of_lens

        assert kappa == pytest.approx(kappa_via_sigma, 1e-4)

    def test__convergence_from_deflections_and_analytic(self):
        NFWHilbert = l1d.NFWHilbert(
            mass_at_200=2.5e12, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(1, 5, 0.00002)

        kappa_analytic = NFWHilbert.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflections_from_profile_and_radii(
            profile=NFWHilbert, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-3

    def test__convergence_values_correct(self):
        NFWHilbert = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_source=1.0, redshift_lens=0.5
        )

        kappa = NFWHilbert.convergence_from_radii(
            radii=np.array([132.3960792, 264.79215844891064, 397.1882377])
        )

        kappa_s = NFWHilbert.kappa_s

        assert kappa == pytest.approx(np.array([0.1567668617, 0, 0.04475020184]), 1e-3)
        assert kappa_s == pytest.approx(0.1129027792225471, 1e-3)

    def test__surface_mass_density_values_correct(self):
        NFWHilbert = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_source=1.0, redshift_lens=0.5
        )

        sigma = NFWHilbert.surface_mass_density_from_radii(
            radii=np.array([132.3960792, 264.79215844891064, 397.1882377])
        )

        assert sigma == pytest.approx(np.array([470603968.6, 0, 134337210.4]), 1e-3)

    def test__density_values_correct(self):
        NFWHilbert = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_source=1.0, redshift_lens=0.5
        )
        rho_s = NFWHilbert.rho_s

        rho = NFWHilbert.density_from_radii(
            radii=np.array([132.3960792, 264.79215844891064, 397.1882377])
        )

        assert rho == pytest.approx(
            np.array([1137753.844, 319993.2686, 136530.4613]), 1e-3
        )
        assert rho_s == pytest.approx(1279973.074564, 1e-3)

    def test__compare_to_autogalaxy__sigma_crit_equal_in_desired_units(self):

        cosmo = cosmology.Planck15

        nfw = l1d.NFWHilbert(mass_at_200=2.5e10, redshift_lens=0.3, redshift_source=0.8)
        sigma_crit = nfw.critical_surface_density_of_lens

        autolens_sigma_crit = ag.util.cosmology.critical_surface_density_between_redshifts_from(
            redshift_1=0.8,
            redshift_0=0.3,
            unit_length="kpc",
            unit_mass="solMass",
            cosmology=cosmo,
        )

        assert sigma_crit == pytest.approx(autolens_sigma_crit, 1e-4)

    def test_compare_to_autogalaxy__radius_at_200_equal_in_desired_units(self):

        cosmo = cosmology.Planck15

        ag_nfw = ag.mp.SphericalNFWMCRLudlow(
            mass_at_200=2.5e14, redshift_object=0.5, redshift_source=1.0
        )

        l1d_nfw = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.5, redshift_source=1.0
        )

        scale_radius_kpc = ag_nfw.scale_radius / ag.util.cosmology.arcsec_per_kpc_from(
            redshift=0.5, cosmology=cosmo
        )

        radius_at_200 = ag_nfw.radius_at_200_for_units(
            redshift_object=0.5, redshift_source=1.0, unit_length="kpc"
        )

        assert l1d_nfw.radius_at_200.value == pytest.approx(radius_at_200, 1e-3)

        assert l1d_nfw.scale_radius == pytest.approx(scale_radius_kpc, 1e-3)

        assert l1d_nfw.kappa_s == pytest.approx(ag_nfw.kappa_s, 1e-3)

    def test__compare_to_autogalaxy__nfw_einstein_radius_equal_in_desired_units(self):
        cosmo = cosmology.Planck15

        ## ONLY PASSSES FOR HIGH RESOLUTION AUTOLENS GRID

        ag_nfw = ag.mp.SphericalNFWMCRLudlow(
            mass_at_200=2.5e14, redshift_object=0.3, redshift_source=1.0
        )

        l1d_nfw = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        radii = np.arange(0.01, 5, 0.001)

        einstein_radius = l1d_nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        ag_einstein_radius = ag_nfw.einstein_radius_in_units(
            unit_length="kpc", redshift_object=0.3, cosmology=cosmo
        )

        assert einstein_radius == pytest.approx(ag_einstein_radius, 1e-2)

    def test__mean_NFW_convergence_within_einstein_radius_equal_to_one(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e13, redshift_lens=0.6, redshift_source=1.2)
        radii = np.arange(0.01, 3, 0.001)

        einstein_radius = nfw.einstein_radius_in_kpc_from_radii(radii=radii)

        integrand = lambda r: 2 * np.pi * r * nfw.convergence_from_radii(radii=r)

        av_kappa = integrate.quad(integrand, 0, einstein_radius)[0] / (
            np.pi * einstein_radius ** 2
        )

        assert av_kappa == pytest.approx(1, 1e-3)

    def test__convergence_at_einstein_radius_less_than_one(self):
        nfw = l1d.NFWHilbert(mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=0.8)
        radii = np.arange(0.2, 30, 0.002)

        kappa_ein = nfw.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1
