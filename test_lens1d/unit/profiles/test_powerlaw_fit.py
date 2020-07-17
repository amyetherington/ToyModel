import os
import numpy as np
import pytest
from astropy import cosmology
from scipy import integrate

import lens1d as l1d

cosmo = cosmology.Planck15


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
