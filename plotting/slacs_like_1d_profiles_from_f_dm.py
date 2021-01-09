import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import lens1d as l1d

slacs_path = "{}/../../autolens_slacs_pre_v_1/dataset/slacs_data_table.xlsx".format(
    os.path.dirname(os.path.realpath(__file__))
)
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name

fig_path = "/Users/dgmt59/Documents/Plots/one_d_stuff/one_d_slacs/"




lens_name = np.array(
    [  #'slacs0008-0004',
        "slacs0330-0020",
        #     'slacs0903+4116',
        'slacs0959+0410',
     #   "slacs1029+0420",
        #   'slacs1153+4612',
        "slacs1402+6321",
        "slacs1451-0239",
        'slacs2300+0022',
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
        'slacs1016+3859',
        #    'slacs1103+5322',
        "slacs1213+6708",
     #   "slacs1420+6019",
        "slacs1627-0053",
        "slacs0216-0813",
   #     "slacs0822+2652",
        "slacs0946+1006",
        'slacs1020+1122',
        "slacs1142+1001",
        'slacs1218+0830',
        "slacs1430+4105",
        "slacs1630+4520",
        "slacs0252+0039",
        #    'slacs0841+3824',
      #  "slacs0956+5100",
        'slacs1023+4230',
        "slacs1143-0144",
        "slacs1250+0523",
        #   'slacs1432+6317',
        "slacs2238-0754",
        "slacs2341+0000",
    ]
)


radii = np.arange(0.01, 100, 0.001)

f = open("slacs_like_test_f_dm", "w")

for i in range(len(lens_name)):

    baryons = l1d.Hernquist(
        mass=10 ** (slacs["log[M*/M]_chab"][lens_name[i]]),
        effective_radius=slacs["R_eff"][lens_name[i]],
        redshift_lens=slacs["z_lens"][lens_name[i]],
        redshift_source=slacs["z_source"][lens_name[i]],
    )
    
    true_profile = l1d.CombinedProfile.from_hernquist_and_dark_matter_fraction_within_effective_radius(
        hernquist=baryons, dark_matter_fraction=slacs["f_dm_chab"][lens_name[i]]
    )

    mask_einstein_radius = true_profile.mask_einstein_radius_from_radii(width=5, radii=radii)
    
    fit_mask = l1d.PowerLawFit(profile=true_profile, radii=radii, mask=mask_einstein_radius)
    
    fit_no_mask = l1d.PowerLawFit(profile=true_profile, radii=radii, mask=None)

    einstein_radius = true_profile.einstein_radius_in_kpc_from_radii(radii=radii)

    effective_radius = true_profile.effective_radius

    einstein_mass = true_profile.einstein_mass_in_solar_masses_from_radii(radii=radii)

    three_d_mass = true_profile.three_dimensional_mass_enclosed_within_effective_radius

    straightness = fit_no_mask.r_squared_value()

    lensing_slope = fit_no_mask.slope_via_lensing()

    dynamics_slope = fit_no_mask.slope_and_normalisation_via_dynamics()[1]

    kappa_fit_slope = fit_no_mask.slope_with_error()[0]

    einstein_radius_best_fit = fit_no_mask.einstein_radius_with_error()[
        0
    ]

    kappa_ein_fit_slope = fit_mask.slope_with_error()[0]

    f_dm_eff = true_profile.dark_matter_mass_fraction_within_effective_radius

    f_dm_ein = true_profile.dark_matter_mass_fraction_within_einstein_radius_from_radii(radii=radii)

    f.write(
        str(lens_name[i])
        + " "
        + str(einstein_radius)
        + " "
        + str(einstein_radius_best_fit)
        + " "
        + str(effective_radius)
        + " "
        + str(f_dm_eff)
        + " "
        + str(f_dm_ein)
        + " "
        + str(three_d_mass)
        + " "
        + str(einstein_mass)
        + " "
        + str(straightness)
        + " "
        + str(lensing_slope)
        + " "
        + str(dynamics_slope)
        + " "
        + str(kappa_fit_slope)
        + " "
        + str(kappa_ein_fit_slope)
        + " "
        + "\n"
    )

#    kappa_baryons = baryons.convergence_from_radii(radii=radii)
#    kappa_DM = DM.convergence_from_radii(radii=radii)
#    kappa_total = true_profile.convergence_from_radii(radii=radii)
#    kappa_dynamics = fit_no_mask.convergence_via_dynamics()
#    kappa_lensing = fit_no_mask.convergence_via_lensing()
#    kappa_best_fit = fit_no_mask.convergence()

#    fig1 = plt.figure(1)
#    plt.loglog(
#        radii, kappa_baryons, "--", label="baryons", alpha=0.5, color="lightcoral"
#    )
#    plt.loglog(
#        radii, kappa_DM, "--", label="dark matter", alpha=0.5, color="lightskyblue"
#    )
#    plt.axvline(x=einstein_radius, color="grey", alpha=0.5)
#    plt.axvline(x=effective_radius, color="darkslategrey", alpha=0.5)
#    plt.loglog(
#        radii, kappa_best_fit, "-.", label="best fit kappa", color="navy", alpha=0.8
#    )
#    plt.loglog(
#        radii, kappa_lensing, "-.", label="kappa via lensing", color="blue", alpha=0.8
#    )
#    plt.loglog(
#        radii, kappa_dynamics, "-.", label="kappa via dyn", color="cyan", alpha=0.8
#    )
#    plt.loglog(radii, kappa_total, label="total", color="plum")
#    plt.legend()
#    plt.xlabel("Radius (kpc)", fontsize=14)
#    plt.ylabel("Convergence", fontsize=14)
#    plt.savefig(fig_path + lens_name[i] + "_100kpc_eff.png", bbox_inches="tight", dpi=300, transparent=True)
#    plt.close()
f.close()