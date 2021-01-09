import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy import cosmology

lens_name = np.array(
    [  #'slacs0008-0004',
        "slacs0330-0020",
        #     'slacs0903+4116',
        'slacs0959+0410',
     #   "slacs1029+0420",
        #   'slacs1153+4612',
        "slacs1402+6321",
        "slacs1451-0239",
    #    'slacs2300+0022',
        "slacs0029-0055",
        "slacs0728+3835",
        "slacs0912+0029",
        "slacs0959+4416",
        #     'slacs1032+5322',
        "slacs1205+4910",
     #   "slacs1416+5136",
        "slacs1525+3327",
        "slacs2303+1422",
        #    'slacs0157-0056',
        "slacs0737+3216",
        "slacs0936+0913",
     #   'slacs1016+3859',
        #    'slacs1103+5322',
        "slacs1213+6708",
     #   "slacs1420+6019",
        "slacs1627-0053",
        "slacs0216-0813",
   #     "slacs0822+2652",
        "slacs0946+1006",
        'slacs1020+1122',
        "slacs1142+1001",
    #    'slacs1218+0830',
        "slacs1430+4105",
        "slacs1630+4520",
        "slacs0252+0039",
        #    'slacs0841+3824',
      #  "slacs0956+5100",
   #     'slacs1023+4230',
        "slacs1143-0144",
        "slacs1250+0523",
        #   'slacs1432+6317',
        "slacs2238-0754",
        "slacs2341+0000",
    ]
)

data = pd.read_csv(
    "/Users/dgmt59/PycharmProjects/toy_model/plotting/slacs_like_test",
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
slacs_new = slacs

for i in range(len(slacs.index)):
     if slacs.index[i] not in data.index:
         slacs_new = slacs_new.drop(slacs.index[i], axis=0)
     else:
         pass

cosmo = cosmology.Planck15

print(np.mean(slacs_new["gamma"]))
print(np.mean(data["dyn_slope"]))

for lens in lens_name:
    kpc_per_arcsec = cosmo.arcsec_per_kpc_proper(z=slacs_new["z_lens"][lens]).value
    print("radius:", data["r_ein"][lens], slacs_new["b_SIE"][lens]/kpc_per_arcsec, data["f_dm_eff"][lens])
    print("slope:", data["dyn_slope"][lens], slacs_new["gamma"][lens])




#fig1, ax = plt.subplots(figsize=(8,8))
#ax.scatter(data_2_1['kappa_slope'], data_2['kappa_slope'])
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
    data["straightness"], data["dyn_slope"], label="slope from dynamics", color="cyan", marker="x"
)
plt.scatter(
    data["straightness"],
    data["lens_slope"],
    label="slope from lensing",
    color="magenta",
marker="x"
)
plt.xlabel("r squared", fontsize=14)
plt.ylabel("slope", fontsize=14)
plt.legend()
#plt.savefig(f"{fig_path}slope_v_straightness.png", bbox_inches='tight', dpi=300)


fig4 = plt.figure(4)
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
#plt.savefig(f"{fig_path}slope_v_ein.png", bbox_inches='tight', dpi=300)

fig5 = plt.figure(5)
plt.scatter(
    data["f_dm_ein"], data["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data["f_dm_ein"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("f_dm ein", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
plt.show()
#plt.savefig(f"{fig_path}slope_v_fdm_ein.png", bbox_inches='tight', dpi=300)

fig6 = plt.figure(6)
plt.scatter(
    data["f_dm_eff"], data["dyn_slope"], label="slope from dynamics", color="cyan"
)
plt.scatter(
    data["f_dm_eff"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("f_dm eff", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
#plt.savefig(f
plt.show()
# "{fig_path}slope_v_fdm_eff.png", bbox_inches='tight', dpi=300)

fig7 = plt.figure(7)
plt.scatter(data["m_dyn"], data["dyn_slope"], label="slope from dynamics", color="cyan")
plt.scatter(
    data["m_dyn"], data["lens_slope"], label="slope from lensing", color="magenta"
)
plt.xlabel("dynamical mass", fontsize=14)
plt.ylabel("Slope", fontsize=14)
plt.legend()
#plt.savefig(f"{fig_path}slope_v_M.png", bbox_inches='tight', dpi=300)

fig8, (ax1) = plt.subplots(figsize=(5,5))
plt.scatter(data["dyn_slope"], data["lens_slope"], color='cyan')
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
plt.show()
#plt.savefig(f"{fig_path}slope_v_slope.png", bbox_inches='tight', dpi=300)


print(len(data["lens_slope"]))

fig9=plt.figure()
plt.hist(data["dyn_slope"], 8, color="orchid", alpha=0.4)
plt.hist(data["lens_slope"], 8, color="deepskyblue", alpha=0.4)
plt.show()

