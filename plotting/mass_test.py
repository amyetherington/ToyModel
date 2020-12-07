import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import autofit as af


fig_path = "/Users/dgmt59/Documents/Plots/one_d_stuff/one_d_slacs/"

slacs_path = "{}/../../autolens_slacs_pre_v_1/dataset/slacs_data_table.xlsx".format(
    os.path.dirname(os.path.realpath(__file__))
)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name

output_path = "/Users/dgmt59/output"

lens_name = np.array(
    [  #'slacs0008-0004',
        "slacs0330-0020",
        #     'slacs0903+4116',
        #    'slacs0959+0410',
     #   "slacs1029+0420",
        #   'slacs1153+4612',
        "slacs1402+6321",
     #   "slacs1451-0239",
        #     'slacs2300+0022',
        "slacs0029-0055",
        "slacs0728+3835",
        "slacs0912+0029",
        "slacs0959+4416",
        #     'slacs1032+5322',
        "slacs1205+4910",
      #  "slacs1416+5136",
        "slacs1525+3327",
        "slacs2303+1422",
        #    'slacs0157-0056',
        "slacs0737+3216",
        "slacs0936+0913",
        #    'slacs1016+3859',
        #    'slacs1103+5322',
        "slacs1213+6708",
     #   "slacs1420+6019",
        "slacs1627-0053",
        "slacs0216-0813",
        "slacs0822+2652",
        "slacs0946+1006",
        #    'slacs1020+1122',
        "slacs1142+1001",
        #    'slacs1218+0830',
        "slacs1430+4105",
        "slacs1630+4520",
        "slacs0252+0039",
        #    'slacs0841+3824',
      #  "slacs0956+5100",
        #   'slacs1023+4230',
      #  "slacs1143-0144",
        "slacs1250+0523",
        #   'slacs1432+6317',
        "slacs2238-0754",
        "slacs2341+0000",
    ]
)

lens_name_check = np.array(["slacs0912+0029"])

for lens in lens_name:
    data = pd.read_csv(
        f"/Users/dgmt59/PycharmProjects/toy_model/plotting/{lens}_mass_test_1d",
        sep="\s+",
        names=[
            "mass",
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
    )

    aggregator_results_path_1 = f"{output_path}/slacs_shu_bspline_clean/{lens}"
    agg = af.Aggregator(directory=str(aggregator_results_path_1), completed_only=True)
    agg_shear_pl = agg.filter(agg.directory.contains("phase[1]_mass[total]_source"),
                              agg.directory.contains("power_law__with_shear"),
                              agg.directory.contains("stochastic"))
    instances_shear = [samps.median_pdf_instance for samps in agg_shear_pl.values("samples")]
    slope = np.asarray([instance.galaxies.lens.mass.slope for instance in instances_shear])
    slope_ue = np.asarray([samps.error_vector_at_upper_sigma(sigma=1.0)[7] for samps in agg_shear_pl.values("samples")])
    slope_le = np.asarray([samps.error_vector_at_lower_sigma(sigma=1.0)[7] for samps in agg_shear_pl.values("samples")])
    slope_dyn = np.asarray([info["slope"] for info in agg_shear_pl.values("info")])
    slope_dyn_err = np.asarray([info["slope_err"] for info in agg_shear_pl.values("info")])
    fig = plt.figure()
    plt.scatter(
        data["mass"], data["lens_slope"], label="predicted slope from lensing", color="magenta"
    )
    plt.scatter(
        data["mass"], data["dyn_slope"], label="predicted slope from dynamics", color="cyan"
    )
    plt.xlabel(r"$log[M/M_\odot]$", fontsize=14)
    plt.ylabel(r"$\gamma$", fontsize=14)
    plt.axhline(y=slope_dyn, color='cyan')
    plt.axhspan(slope_dyn-slope_dyn_err,slope_dyn+slope_dyn_err, color='cyan', alpha=0.2)
    plt.axhline(y=slope, color='magenta')
    plt.axhspan(slope-slope_le, slope+slope_ue, color='magenta', alpha=0.2)
    plt.legend()
    plt.savefig(f"{fig_path}slope_v_mass_{lens}.png", bbox_inches='tight', dpi=300)
    plt.show()

    fig = plt.figure()
    plt.scatter(
        data["mass"], data["r_eff"], label="effective_radius", color="magenta"
    )
    plt.scatter(
        data["mass"], data["r_ein"], label="einstein_radius", color="cyan"
    )
    plt.show()

    fig = plt.figure()
    plt.scatter(
        data["mass"], data["m_dyn"], label="effective_radius", color="magenta"
    )
    plt.scatter(
        data["mass"], data["m_ein"], label="einstein_radius", color="cyan"
    )
    plt.show()

    fig = plt.figure()
    plt.scatter(
        data["m_dyn"], data["lens_slope"], label="effective_radius", color="magenta"
    )
    plt.scatter(
        data["m_dyn"], data["dyn_slope"], label="einstein_radius", color="cyan"
    )
    plt.show()

    fig = plt.figure()
    plt.scatter(
        data["mass"], data["r_eff"]*1.33, label="effective_radius", color="magenta"
    )
    plt.scatter(
        data["mass"], data["r_ein"], label="einstein_radius", color="cyan"
    )
    plt.show()

