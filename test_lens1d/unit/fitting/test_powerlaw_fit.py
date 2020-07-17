import os
import numpy as np
import pytest
from astropy import cosmology
from scipy import integrate
from astropy import units as u
from scipy.special import gamma

import lens1d as l1d

cosmo = cosmology.Planck15

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


class TestSlopeFromLensing:
    def test__xi_two_equal_to_zero_for_sis(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        fit = l1d.PowerLawFit(profile=power_law, mask=None, radii=radii)

        xi_two = fit.xi_two()

        assert xi_two == pytest.approx(0, 1e-3)

    def test__slope_equal_to_power_law_input(self):
        power_law = l1d.SphericalPowerLaw(
            einstein_radius=1.8, slope=2, redshift_source=0.8, redshift_lens=0.3
        )
        radii = np.arange(0.2, 3, 0.002)

        fit = l1d.PowerLawFit(profile=power_law, mask=None, radii=radii)

        slope_lensing = fit.slope_via_lensing()

        assert slope_lensing == pytest.approx(power_law.slope, 1e-4)


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

        fit = l1d.PowerLawFit(profile=combined, mask=None, radii=radii)

        rho_0, slope = fit.slope_and_normalisation_via_dynamics()

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

        fit = l1d.PowerLawFit(profile=combined, mask=None, radii=radii)

        rho_0, slope = fit.slope_and_normalisation_via_dynamics()

        reff = combined.effective_radius
        mdyn = combined.three_dimensional_mass_enclosed_within_effective_radius

        mdyn_from_dyn = mass_dynamical(rho0=rho_0, g=slope, Reff=reff * u.kpc)

        assert mdyn == pytest.approx(mdyn_from_dyn, 1e-3)

