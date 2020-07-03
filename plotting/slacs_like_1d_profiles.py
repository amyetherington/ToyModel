import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from one_d_code import combined_profiles as cp
from one_d_code import one_d_profiles as profiles

slacs_path = '{}/../../autolens_slacs/dataset/slacs_data_table.xlsx'.format(os.path.dirname(os.path.realpath(__file__)))
slacs = pd.read_excel(slacs_path, index_col=0)
del slacs.index.name

print(slacs)

lens_name = np.array([#'slacs0008-0004',
                      'slacs0330-0020',
                 #     'slacs0903+4116',
                  #    'slacs0959+0410',
                      'slacs1029+0420',
                   #   'slacs1153+4612',
                      'slacs1402+6321',
                      'slacs1451-0239',
                      'slacs2300+0022',
                      'slacs0029-0055',
                      'slacs0728+3835',
                      'slacs0912+0029',
                      'slacs0959+4416',
                 #     'slacs1032+5322',
                      'slacs1205+4910',
                      'slacs1416+5136',
                      'slacs1525+3327',
                      'slacs2303+1422',
                  #    'slacs0157-0056',
                      'slacs0737+3216',
                      'slacs0936+0913',
                      'slacs1016+3859',
                  #    'slacs1103+5322',
                      'slacs1213+6708',
                      'slacs1420+6019',
                      'slacs1627-0053',
                      'slacs0216-0813',
                      'slacs0822+2652',
                      'slacs0946+1006',
                  #    'slacs1020+1122',
                      'slacs1142+1001',
                  #    'slacs1218+0830',
                      'slacs1430+4105',
                      'slacs1630+4520',
                      'slacs0252+0039',
                  #    'slacs0841+3824',
                      'slacs0956+5100',
                   #   'slacs1023+4230',
                      'slacs1143-0144',
                      'slacs1250+0523',
                   #   'slacs1432+6317',
                      'slacs2238-0754',
                      'slacs2341+0000'])

radii = np.arange(0.01, 500, 0.001)

f = open('slacs_like_test_1d', 'a+')

for i in range(len(lens_name)):
    print(lens_name[i])
    baryons = profiles.Hernquist(mass=10**slacs["log[M*/M]_chab"][lens_name[i]], effective_radius=slacs["R_eff"][lens_name[i]],
                                 z_l=slacs["z_lens"][lens_name[i]], z_s=slacs["z_source"][lens_name[i]])
    DM = profiles.NFW_Hilbert(m200=slacs["M200"][lens_name[i]], z_l=slacs["z_lens"][lens_name[i]],
                              z_s=slacs["z_source"][lens_name[i]])
    true_profile = cp.CombinedProfile(profiles=[baryons, DM])

    no_mask = true_profile.mask_radial_range_from_radii(lower_bound=0, upper_bound=1, radii=radii)
    mask_einstein_radius = true_profile.mask_radial_range_from_radii(lower_bound=0.8, upper_bound=1.2, radii=radii)

    einstein_radius = true_profile.einstein_radius_in_kpc_from_radii(radii=radii)

    einstein_mass = true_profile.einstein_mass_in_solar_masses_from_radii(radii=radii)

    three_d_mass = true_profile.three_dimensional_mass_enclosed_within_effective_radius

    straightness = true_profile.power_law_r_squared_value(radii=radii, mask=no_mask)

    lensing_slope = true_profile.slope_via_lensing(radii=radii)

    dynamics_slope = true_profile.slope_and_normalisation_via_dynamics(radii=radii)[1]

    kappa_fit_slope = true_profile.best_fit_power_law_slope_via_deflection_angles_from_mask_and_radii(mask=no_mask, radii=radii)

    einstein_radius_best_fit = true_profile.best_fit_power_law_einstein_radius_with_error_from_mask_and_radii(mask=no_mask, radii=radii)[0]

    kappa_ein_fit_slope = true_profile.best_fit_power_law_slope_with_error_from_mask_and_radii(mask=mask_einstein_radius, radii=radii)[0]

    f.write(str(lens_name[i]) + ' ' +
            str(einstein_radius) + ' ' +
            str(einstein_radius_best_fit) + ' ' +
            str(three_d_mass) + ' ' +
            str(einstein_mass) + ' ' +
            str(straightness) + ' ' +
            str(lensing_slope) + ' ' +
            str(dynamics_slope) + ' ' +
            str(kappa_fit_slope) + ' ' +
            str(kappa_ein_fit_slope) + ' ' + '\n')
