import astropy.units as u
import numpy as np
import scipy as sp
from scipy.special import gamma, factorial
from scipy import optimize

from numpy import exp
import timeit


# rho(r) = rho_0 * (r/r_0)^-g


def Mdyn(rho0, g, Reff, r0=1 * u.kpc):
    return ((4 * np.pi * rho0 / (3 - g)) * Reff ** 3 * (Reff / r0) ** -g).to("Msun")


def Mein(rho0, g, Rein, r0=1 * u.kpc):
    return (
        (2 * np.pi ** 1.5 * gamma(0.5 * (g - 1)) / ((3 - g) * gamma(0.5 * g)))
        * rho0
        * Rein ** 3
        * (Rein / r0) ** -g
    ).to("Msun")


rho0 = 2.9e9 * u.Msun / u.kpc ** 3
g = 1.38  # not gamma, to avoid confusion with gamma function
Reff = 3.2 * u.kpc
Rein = 5.0 * u.kpc

# easy to generate Mdyn and Mein for a given rho0 and g.
# i.e.
print("Mdyn =", Mdyn(rho0, g, Reff))
print("Mein =", Mein(rho0, g, Rein))

# to find rho0 and g from Mdyn and Mein (and Reff, Rein and an assumed r0), need to invert the equations for Mdyn and Mein, or just try rho0 and g until you find the right answer!

Mdyn_true = 99379503476.72754 * u.Msun
Mein_true = 378916860347.4141 * u.Msun


# now from these, can we get the input rho0 and g?
st3 = timeit.default_timer()


def vector_residuals(params, Mdyn_true, Mein_true, Reff, Rein):
    log_rho0, g = params
    Reff = 5.68 * u.kpc
    Rein = 3.274999999999997 * u.kpc

    rho0 = 10 ** log_rho0 * u.Msun / u.kpc ** 3

    Mdyn_pred = Mdyn(rho0, g, Reff)
    Mein_pred = Mein(rho0, g, Rein)
    return (
        np.log10((Mdyn_pred / Mdyn_true).to("")),
        np.log10((Mein_pred / Mein_true).to("")),
    )


# guess for [log10(rho0), g]
init_guess = np.array([7, 1.8])

root_finding_data = optimize.root(
    vector_residuals,
    init_guess,
    args=(Mdyn_true, Mein_true, Reff, Rein),
    method="hybr",
    options={"xtol": 0.0001},
)

print(root_finding_data)

print("Found rho0 =", 10 ** root_finding_data.x[0] * u.Msun / u.kpc ** 3)
print("Found gamma =", root_finding_data.x[1])

st4 = timeit.default_timer()
print("RUN TIME : {0}".format(st4 - st3))
