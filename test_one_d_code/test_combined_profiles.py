import numpy as np
import pytest
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profiles
from astropy import units as u
from scipy.special import gamma


def convergence_via_deflection_angles_from_profile_and_radii(profile, radii):

    alpha = profile.deflection_angles_from_radii(radii=radii)

    kappa = 0.5 * ((alpha / radii) + np.gradient(alpha, radii[:]))

    return kappa

def Mein(rho0,g,Rein,r0=1*u.kpc):
    rho0=rho0*u.Msun/u.kpc**3
    return ((2*np.pi**1.5 * gamma(0.5*(g-1)) / ((3-g)*gamma(0.5*g))) * rho0 * Rein**3 * (Rein/r0)**-g).to('Msun').value

def Mdyn(rho0,g,Reff,r0=1*u.kpc):
    rho0 = rho0 * u.Msun / u.kpc ** 3
    return ((4*np.pi*rho0 / (3-g)) * Reff**3 * (Reff/r0)**-g).to('Msun').value

class TestCombinedProfile:

    def test__effective_radius_from_combined_profile(self):

        hernquist = profiles.Hernquist(
            effective_radius=1.0, mass=100.0, z_s=0.5, z_l=1.0
        )

        combined = cp.CombinedProfile(profiles=[hernquist])

        assert combined.effective_radius == 1.0

    def test__convergence_from_deflections_and_analytic(self):
        dm = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )
        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)
        ## include some interpolation in convergence from deflection angles???
        # so don't need to have such a finely sampled radii grid, this test will fail if > 0.002

        kappa_analytic = combined.convergence_from_radii(radii=radii)
        kappa_from_deflections = convergence_via_deflection_angles_from_profile_and_radii(
            profile=combined, radii=radii
        )

        mean_error = np.average(kappa_analytic - kappa_from_deflections)

        assert mean_error < 1e-4

    def test__convergence_equals_sum_of_individual_profile_convergence(self):
        dm = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )

        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        kappa_combined = combined.convergence_from_radii(radii=radii)
        kappa_NFW = dm.convergence_from_radii(radii=radii)
        kappa_pl = baryons.convergence_from_radii(radii=radii)
        kappa_sum = kappa_NFW + kappa_pl

        assert kappa_combined == pytest.approx(kappa_sum, 1e-4)

    def test__deflection_angles_equals_sum_of_individual_profile_deflection_angles(
        self
    ):
        dm = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )

        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)

        alpha_combined = combined.deflection_angles_from_radii(radii=radii)
        alpha_NFW = dm.deflection_angles_from_radii(radii=radii)
        alpha_pl = baryons.deflection_angles_from_radii(radii=radii)

        assert alpha_combined == pytest.approx(alpha_NFW + alpha_pl, 1e-4)

    def test__best_fit_einstein_radius_from_intercept_equal_to_from_kappa(self):
        dm = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )

        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(
            radii=radii, lower_bound=0.5, upper_bound=1.5
        )

        einstein_radius_c = combined.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.best_fit_power_law_convergence_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.best_fit_power_law_slope_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )[0]

        einstein_radius_k = np.mean(
            radii * np.divide(2 * kappa, 3 - slope) ** (np.divide(1, slope - 1))
        )

        assert einstein_radius_c[0] == pytest.approx(einstein_radius_k, 1e-3)

    def test__best_fit_einstein_radius_via_deflection_angles_from_intercept_equal_to_from_kappa(
        self
    ):
        dm = profiles.NFW_Hilbert(m200=2.5e14, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=8.4, z_l=0.6, z_s=1.2
        )

        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 3, 0.002)
        mask = combined.mask_radial_range_from_radii(
            radii=radii, lower_bound=0.5, upper_bound=1.5
        )

        einstein_radius_c = combined.best_fit_power_law_einstein_radius_via_deflection_angles_with_error_from_mask_and_radii(
            radii=radii, mask=mask
        )

        kappa = combined.best_fit_power_law_convergence_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        slope = combined.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(
            radii=radii, mask=mask
        )

        einstein_radius_k = np.mean(
            radii * np.divide(2 * kappa, 3 - slope) ** (np.divide(1, slope - 1))
        )

        assert einstein_radius_c == pytest.approx(einstein_radius_k, 1e-3)

class TestSlopeFromDynamics:
    def test__einstein_mass_from_dynamics_slope_and_normalization_equals_analytic(self):
        dm = profiles.NFW_Hilbert(m200=2.5e12, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=3.2, z_l=0.3, z_s=1.0
        )
        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        slope = combined.slope_and_normalisation_via_dynamics(radii=radii)[1]
        rho_0 = combined.slope_and_normalisation_via_dynamics(radii=radii)[0]

        rein = combined.einstein_radius_in_kpc_from_radii(radii=radii)
        mein = combined.einstein_mass_in_solar_masses_from_radii(radii=radii)

        mein_from_dyn = Mein(rho0=rho_0, g=slope, Rein=rein*u.kpc)


        assert mein == pytest.approx(mein_from_dyn, 1e-3)

    def test__three_d_mass_from_dynamics_slope_and_normalization_equals_analytic(self):
        dm = profiles.NFW_Hilbert(m200=2.5e12, z_l=0.3, z_s=1.0)
        baryons = profiles.Hernquist(
            mass=3.4e11, effective_radius=3.2, z_l=0.3, z_s=1.0
        )
        combined = cp.CombinedProfile(profiles=[dm, baryons])

        radii = np.arange(0.2, 8, 0.002)

        slope = combined.slope_and_normalisation_via_dynamics(radii=radii)[1]
        rho_0 = combined.slope_and_normalisation_via_dynamics(radii=radii)[0]

        reff = combined.effective_radius
        mdyn = combined.three_dimensional_mass_enclosed_within_effective_radius

        mdyn_from_dyn = Mdyn(rho0=rho_0, g=slope, Reff=reff*u.kpc)


        assert mdyn == pytest.approx(mdyn_from_dyn, 1e-3)

