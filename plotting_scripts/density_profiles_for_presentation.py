import numpy as np
import matplotlib.pyplot as plt

fig_path = "/Users/dgmt59/Documents/Plots/one_d_slacs/"

def general_nfw(rho_s, r_s, inner_slope, radii):
    x = np.divide(radii, r_s)
    return np.divide(rho_s, x ** inner_slope * (1 + x) ** (3 - inner_slope))

def isothermal(rho_s, radii):
    return rho_s * np.divide(1, radii) ** 2

def hernquist(rho_s, r_s, radii):
    x = np.array(radii / r_s)
    return np.divide(rho_s, x * (1 + x) ** 3)

radii = np.arange(0.2, 200, 0.001)

baryons = hernquist(rho_s=1.5e8, r_s=20, radii=radii)
isothermal = isothermal(rho_s=1.2e9, radii=radii)
dm = general_nfw(rho_s=1.8e6, r_s=30, radii=radii, inner_slope=1)
dm_cored = general_nfw(rho_s=1e7, r_s=80, radii=radii, inner_slope=1.8)
total = dm+baryons
total_cored = dm_cored+baryons

fig3 = plt.figure(3)
plt.loglog(
    radii, baryons, "--", label="baryons", color="lightcoral"
)
plt.loglog(
    radii, dm, "--", label="dark matter", color="cyan"
)
plt.loglog(
    radii, dm_cored, "--", label="dark matter cored", color="skyblue"
)
plt.loglog(
    radii, total_cored, label="total matter cored",  color="teal"
)
plt.loglog(radii, total, label="total", color="purple")
plt.loglog(radii, isothermal, "-.", label="isothermal", color="grey")
plt.legend()
plt.xlabel("Radius (kpc)", fontsize=14)
plt.ylabel(r"$\rho$", fontsize=14)
plt.savefig(fig_path + "density.png", bbox_inches="tight", dpi=300, transparent=True)

plt.show()
