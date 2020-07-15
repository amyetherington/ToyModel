import numpy as np
import pytest
from astropy import units as u
from scipy.special import gamma

import lens1d as l1d


def convergence_via_deflections_from_profile_and_radii(profile, radii):

    alpha = profile.deflections_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa


def mass_einstein(rho0, g, Rein, r0=1 * u.kpc):
    rho0 = rho0 * u.Msun / u.kpc ** 3
    return (
        (
            (2 * np.pi ** 1.5 * gamma(0.5 * (g - 1)) / ((3 - g) * gamma(0.5 * g)))
            * rho0
            * Rein ** 3
            * (Rein / r0) ** -g
        )
        .to("Msun")
        .value
    )


def mass_dynamical(rho0, g, Reff, r0=1 * u.kpc):
    rho0 = rho0 * u.Msun / u.kpc ** 3
    return (
        ((4 * np.pi * rho0 / (3 - g)) * Reff ** 3 * (Reff / r0) ** -g).to("Msun").value
    )


class TestCombinedProfile:
    def test__effective_radius_from_combined_profile(self):

        hernquist = l1d.Hernquist(
            effective_radius=1.0, mass=100.0, redshift_source=0.5, redshift_lens=1.0
        )

        combined = l1d.CombinedProfile(profiles=[hernquist])

        assert combined.effective_radius == 1.0

    def test__convergence_from_deflections_and_analytic(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )
        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        kappa_analytic = combined.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflections_from_profile_and_radii(
            profile=combined, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_equals_sum_of_individual_profile_convergence(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        kappa_combined = combined.convergence_from_radii(radii=radii)
        kappa_NFW = dm.convergence_from_radii(radii=radii)
        kappa_pl = baryons.convergence_from_radii(radii=radii)
        kappa_sum = kappa_NFW + kappa_pl

        assert kappa_combined == pytest.approx(kappa_sum, 1e-4)

    def test__deflections_equals_sum_of_individual_profile_deflections(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        alpha_combined = combined.deflections_from_radii(radii=radii)
        alpha_NFW = dm.deflections_from_radii(radii=radii)
        alpha_pl = baryons.deflections_from_radii(radii=radii)

        assert alpha_combined == pytest.approx(alpha_NFW + alpha_pl, 1e-4)

    def test__surface_mass_density_equals_sum_of_individual_profiles(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        sigma_combined = combined.surface_mass_density_from_radii(radii=radii)
        sigma_NFW = dm.surface_mass_density_from_radii(radii=radii)
        sigma_pl = baryons.surface_mass_density_from_radii(radii=radii)
        sigma_sum = sigma_NFW + sigma_pl

        assert sigma_combined == pytest.approx(sigma_sum, 1e-4)

    def test__density_equals_sum_of_individual_profiles(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        rho_combined = combined.density_from_radii(radii=radii)
        rho_NFW = dm.density_from_radii(radii=radii)
        rho_pl = baryons.density_from_radii(radii=radii)
        rho_sum = rho_NFW + rho_pl

        assert rho_combined == pytest.approx(rho_sum, 1e-4)


class TestSlopeFromDynamics:
    def test__einstein_mass_from_dynamics_slope_and_normalization_equals_analytic(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=3.2, redshift_lens=0.3, redshift_source=1.0
        )
        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        slope = combined.slope_and_normalisation_via_dynamics(radii=radii)[1]
        rho_0 = combined.slope_and_normalisation_via_dynamics(radii=radii)[0]

        rein = combined.einstein_radius_in_kpc_from_radii(radii=radii)
        mein = combined.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mein_from_dyn = mass_einstein(rho0=rho_0, g=slope, Rein=rein * u.kpc)

        assert mein == pytest.approx(mein_from_dyn, 1e-3)

    def test__three_d_mass_from_dynamics_slope_and_normalization_equals_analytic(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=3.2, redshift_lens=0.3, redshift_source=1.0
        )
        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        slope = combined.slope_and_normalisation_via_dynamics(radii=radii)[1]
        rho_0 = combined.slope_and_normalisation_via_dynamics(radii=radii)[0]

        reff = combined.effective_radius
        mdyn = combined.three_dimensional_mass_enclosed_within_effective_radius

        mdyn_from_dyn = mass_dynamical(rho0=rho_0, g=slope, Reff=reff * u.kpc)

        assert mdyn == pytest.approx(mdyn_from_dyn, 1e-3)


class TestSlopeFromLensing:
    def test__convergence_at_einstein_radius_less_than_one(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=3.2, redshift_lens=0.3, redshift_source=1.0
        )
        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        kappa_ein = combined.convergence_at_einstein_radius_from_radii(radii=radii)

        assert kappa_ein < 1


class TestDarkMatterFraction:
    def test__f_dm_equal_to_one_for_nfw_only_profile(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e12, redshift_lens=0.3, redshift_source=1.0
        )
        combined = l1d.CombinedProfile(profiles=[dm])

        radii = np.arange(0.2, 8, 0.002)

        f_dm_ein = combined.dark_matter_mass_fraction_within_einstein_radius_from_radii(
            radii=radii
        )

        assert f_dm_ein == pytest.approx(1, 1e-4)


class TestBestFit:
    def test__best_fit_einstein_radius_from_intercept_equal_to_from_kappa(self):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(
            radii=radii, lower_bound=0.5, upper_bound=1.5
        )

        einstein_radius_c = combined.inferred_einstein_radius_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.inferred_convergence_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.inferred_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius_k = np.mean(
            radii * np.divide(2 * kappa, 3 - slope) ** (np.divide(1, slope - 1))
        )

        assert einstein_radius_c[0] == pytest.approx(einstein_radius_k, 1e-3)

    def test__best_fit_einstein_radius_via_deflections_from_intercept_equal_to_from_kappa(
        self
    ):
        dm = l1d.NFWHilbert(
            mass_at_200=2.5e14, redshift_lens=0.3, redshift_source=1.0
        )
        baryons = l1d.Hernquist(
            mass=3.4e11, effective_radius=8.4, redshift_lens=0.6, redshift_source=1.2
        )

        combined = l1d.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(
            radii=radii, lower_bound=0.5, upper_bound=1.5
        )

        einstein_radius_c = combined.inferred_einstein_radius_via_deflections_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.inferred_convergence_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.inferred_slope_via_deflections_from_mask_and_radii(
            radii=radii, mask=mask
        )

        einstein_radius_k = np.mean(
            radii * np.divide(2 * kappa, 3 - slope) ** (np.divide(1, slope - 1))
        )

        assert einstein_radius_c == pytest.approx(einstein_radius_k, 1e-3)
