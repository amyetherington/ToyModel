import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data_1 = pd.read_csv(
    "/Users/dgmt59/PycharmProjects/toy_model/plotting/slacs_like_test_1d_100kpc",
    sep="\s+",
    names=[
        "lens_name",
        "r_ein",
        "r_ein_best",
        "r_eff",
        "f_dm_eff",
        "f_dm_ein",
        "m_dyn",
        "m_ein",
        "straightness",
        "lens_slope",
        "dyn_slope",
        "kappa_slope",
        "kappa_ein_slope",
    ],
    index_col=0,
)
del data_1.index.name


data_2 = pd.read_csv(
    "/Users/dgmt59/PycharmProjects/toy_model/plotting/slacs_like_test_1d_eff",
    sep="\s+",
    names=[
        "lens_name",
        "r_ein",
        "r_ein_best",
        "r_eff",
        "f_dm_eff",
        "f_dm_ein",
        "m_dyn",
        "m_ein",
        "straightness",
        "lens_slope",
        "dyn_slope",
        "kappa_slope",
        "kappa_ein_slope",
    ],
    index_col=0,
)
del data_2.index.name

fig_path = "/Users/dgmt59/Documents/Plots/one_d_stuff/one_d_slacs/"



slacs_path = "{}/../../autolens_slacs_pre_v_1/dataset/slacs_data_table.xlsx".format(
    os.path.dirname(os.path.realpath(__file__))
)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name

print(np.mean(data_2["dyn_slope"]))

print(np.mean(data_2["lens_slope"]))





#fig1, ax = plt.subplots(figsize=(8,8))
#ax.scatter(data_2_2_1['kappa_slope'], data_2_2['kappa_slope'])
#plt.xlabel(r'$slope (500kpc)$', size=14)
#plt.ylabel(r'$slope (50kpc)$', size=14)
#box = ax.get_position()
#ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#lims = [
#    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
#    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
#     ]
#ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0)
#ax.set_aspect('equal')
#ax.set_xlim(lims)
#ax.set_ylim(lims)
#plt.savefig(fig_path +"_slope_comparison.png", bbox_inches="tight", dpi=300)
#plt.show()

#stop

fig3 = plt.figure(3)
plt.scatter(
    data_2["straightness"], data_2["dyn_slope"], label="slope from dynamics", color="cyan", marker="x"
)
plt.scatter(
    data_2["straightness"],
    data_2["lens_slope"],
    label="slope from lensing",
    color="magenta",
marker="x"
)
plt.xlabel("r squared", fontsize=14)
plt.ylabel("slope", fontsize=14)
plt.legend()
plt.savefig(f"{fig_path}slope_v_straightness.png", bbox_inches='tight', dpi=300)


fig4 = plt.figure(4)
plt.scatter(data_2["r_ein"], data_2["dyn_slope"], label="slope from dynamics", color="cyan")
plt.scatter(
    data_2["r_ein"], data_2["lens_slope"], label="slope from lensing", color="magenta"
)
plt.scatter(
    data_2["r_ein"], data_2["kappa_slope"], label="slope from fit to kappa", color="orange"
)
plt.xlabel("r ein", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(f"{fig_path}slope_v_ein.png", bbox_inches='tight', dpi=300)

fig5 = plt.figure(5)
plt.scatter(
    data_2["f_dm_ein"], data_2["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data_2["f_dm_ein"], data_2["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("f_dm ein", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(f"{fig_path}slope_v_fdm_ein.png", bbox_inches='tight', dpi=300)

fig6 = plt.figure(6)
plt.scatter(
    data_2["f_dm_eff"], data_2["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data_2["f_dm_eff"], data_2["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("f_dm eff", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(f"{fig_path}slope_v_fdm_eff.png", bbox_inches='tight', dpi=300)

fig7 = plt.figure(7)
plt.scatter(data_2["m_dyn"], data_2["dyn_slope"], label="slope from dynamics", color="cyan")
plt.scatter(
    data_2["m_dyn"], data_2["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("dynamical mass", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.savefig(f"{fig_path}slope_v_M.png", bbox_inches='tight', dpi=300)

fig8, (ax1) = plt.subplots(figsize=(5,5))
plt.scatter(data_2["dyn_slope"], data_2["lens_slope"], color='cyan')
plt.xlabel("dynamics", fontsize=14)
plt.ylabel("lensing", fontsize=14)
lims1 = [
    np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
    np.max([ax1.get_xlim(), ax1.get_ylim()]),  # max of both axes
     ]
ax1.plot(lims1, lims1, 'k--', alpha=0.75, zorder=0)
ax1.set_aspect('equal')
ax1.set_xlim(lims1)
ax1.set_ylim(lims1)
plt.savefig(f"{fig_path}slope_v_slope.png", bbox_inches='tight', dpi=300)


print(len(data_2["lens_slope"]))

fig9, (ax1) = plt.subplots(figsize=(5,5))
plt.hist(data_2["dyn_slope"], 8, color="orchid", alpha=0.4)
plt.hist(data_2["lens_slope"], 8, color="deepskyblue", alpha=0.4)
plt.show()

