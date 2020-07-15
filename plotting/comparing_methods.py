import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

data = pd.read_csv(
    "/Users/dgmt59/PycharmProjects/toy_model/plotting/slacs_like_test_1d",
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
del data.index.name


slacs_path = "{}/../../autolens_slacs_pre_v_1/dataset/slacs_data_table.xlsx".format(
    os.path.dirname(os.path.realpath(__file__))
)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name


fig1 = plt.figure(1)
plt.scatter(
    data["straightness"], data["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data["straightness"],
    data["lens_slope"],
    label="slope from lensing",
    color="magenta",
)
plt.scatter(
    data["straightness"],
    data["kappa_slope"],
    label="slope from fit to kappa",
    color="orange",
)
plt.xlabel("r squared", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()

fig2 = plt.figure(2)
plt.scatter(data["r_ein"], data["dyn_slope"], label="slope from dynamics", color="cyan")
plt.scatter(
    data["r_ein"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.scatter(
    data["r_ein"], data["kappa_slope"], label="slope from fit to kappa", color="orange"
)
plt.xlabel("r ein", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()


fig3 = plt.figure(3)
plt.scatter(
    data["f_dm_ein"], data["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data["f_dm_ein"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.scatter(
    data["f_dm_ein"],
    data["kappa_slope"],
    label="slope from fit to kappa",
    color="orange",
)
plt.scatter(
    data["f_dm_ein"],
    data["kappa_ein_slope"],
    label="slope from fit to kappa",
    color="purple",
)
plt.xlabel("f_dm ein", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()

fig4 = plt.figure(4)
plt.scatter(
    data["f_dm_eff"], data["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data["f_dm_eff"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.scatter(
    data["f_dm_eff"],
    data["kappa_slope"],
    label="slope from fit to kappa",
    color="orange",
)
plt.xlabel("f_dm eff", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()

fig4 = plt.figure(5)
plt.scatter(data["m_dyn"], data["dyn_slope"], label="slope from dynamics", color="cyan")
plt.scatter(
    data["m_dyn"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.scatter(
    data["m_dyn"], data["kappa_slope"], label="slope from fit to kappa", color="orange"
)
plt.xlabel("dynamical mass", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.show()
