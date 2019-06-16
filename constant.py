'''
Constants, conversions, and units.
In CGS, unless otherwise noted.

@author: Andrew Wetzel <arwetzel@gmail.com>
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
from scipy import constants as scipy_const

# physical constants ----------
grav = scipy_const.gravitational_constant * 1e3  # ~6.67384e-8 [cm^3 / g / s^2]
speed_light = scipy_const.speed_of_light * 1e2  # [cm / s]
boltzmann = scipy_const.k * 1e7  # [erg / K]
electron_volt = scipy_const.electron_volt * 1e7  # [erg]
proton_mass = scipy_const.proton_mass * 1e3  # [gram]
electron_mass = scipy_const.electron_mass * 1e3  # [gram]

amu_mass = scipy_const.m_u * 1e3  # 1/12 of carbon mass [gram]
hydrogen_mass = 1.008 * amu_mass
helium_mass = 4.002602 * amu_mass
carbon_mass = 12.0107 * amu_mass
nitrogen_mass = 14.0067 * amu_mass
oxygen_mass = 15.9994 * amu_mass
neon_mass = 20.1797 * amu_mass
magnesium_mass = 24.305 * amu_mass
silicon_mass = 28.0855 * amu_mass
sulphur_mass = 32.065 * amu_mass
calcium_mass = 40.078 * amu_mass
iron_mass = 55.847 * amu_mass

# astrophysical constants ----------
year = scipy_const.Julian_year  # Julian [sec]
parsec = scipy_const.parsec * 1e2  # ~3.0857e18 [cm]
au = scipy_const.astronomical_unit * 1e2  # [cm]

sun_mass = 1.98892e33  # [gram]
sun_luminosity = 3.842e33  # [erg]
sun_magnitude = 4.76  # bolometric (varies with filter but good for sdss r-band)
sun_radius = 6.957e10  # [cm]

# solar element abundances (photosphere values from Asplund et al 2009) ----------
# massfraction := metal mass / total mass
# abundance := metal number / hydrogen number

sun_metal_mass_fraction = 0.0134
sun_helium_mass_fraction = 0.2485
sun_hydrogen_mass_fraction = 0.7381

# old values from Anders & Grevesse 1989
#sun_metal_mass_fraction = 0.02
#sun_helium_mass_fraction = 0.274
#sun_hydrogen_mass_fraction = 0.706

hydrogen_factor = sun_hydrogen_mass_fraction / hydrogen_mass

sun_composition = {
    'hydrogen': {'massfraction': sun_hydrogen_mass_fraction},

    'helium': {'abundance': 10 ** (10.93 - 12), 'massfraction': sun_helium_mass_fraction},

    'metals': {'massfraction': sun_metal_mass_fraction},
    'total': {'massfraction': sun_metal_mass_fraction},

    'carbon': {'abundance': 10 ** (8.43 - 12),
               'massfraction': 10 ** (8.43 - 12) * carbon_mass * hydrogen_factor},

    'nitrogen': {'abundance': 10 ** (7.83 - 12),
                 'massfraction': 10 ** (7.83 - 12) * nitrogen_mass * hydrogen_factor},

    'oxygen': {'abundance': 10 ** (8.69 - 12),
               'massfraction': 10 ** (8.69 - 12) * oxygen_mass * hydrogen_factor},

    'neon': {'abundance': 10 ** (7.93 - 12),
             'massfraction': 10 ** (7.93 - 12) * neon_mass * hydrogen_factor},

    'magnesium': {'abundance': 10 ** (7.60 - 12),
                  'massfraction': 10 ** (7.60 - 12) * magnesium_mass * hydrogen_factor},

    'silicon': {'abundance': 10 ** (7.51 - 12),
                'massfraction': 10 ** (7.51 - 12) * silicon_mass * hydrogen_factor},

    'sulphur': {'abundance': 10 ** (7.12 - 12),
                'massfraction': 10 ** (7.12 - 12) * sulphur_mass * hydrogen_factor},

    'calcium': {'abundance': 10 ** (6.34 - 12),
                'massfraction': 10 ** (6.34 - 12) * calcium_mass * hydrogen_factor},

    'iron': {'abundance': 10 ** (7.50 - 12),
             'massfraction': 10 ** (7.50 - 12) * iron_mass * hydrogen_factor},
}

del(hydrogen_factor)

# dictionaries to convert between element name and symbol
element_symbol_from_name = {
    'hydrogen': 'h',
    'helium': 'he',
    'carbon': 'c',
    'nitrogen': 'n',
    'oxygen': 'o',
    'neon': 'ne',
    'magnesium': 'mg',
    'silicon': 'si',
    'sulphur': 's',
    'calcium': 'ca',
    'iron': 'fe',
}
element_name_from_symbol = {v: k for k, v in element_symbol_from_name.items()}

# store abundances accessible by element symbol as well
for name in element_symbol_from_name:
    sun_composition[element_symbol_from_name[name]] = sun_composition[name]

# conversions ----------
# metric
micro = 1e-6
milli = 1e-3
centi = 1e-2
kilo = 1e3
mega = 1e6
giga = 1e9

centi_per_kilo = 1e5
kilo_per_centi = 1 / centi_per_kilo

centi_per_mega = 1e8
mega_per_centi = 1 / centi_per_mega

kilo_per_mega = 1e3
mega_per_kilo = 1 / kilo_per_mega

# mass
gram_per_sun = sun_mass
sun_per_gram = 1 / gram_per_sun

gram_per_proton = proton_mass
proton_per_gram = 1 / proton_mass

gram_per_hydrogen = hydrogen_mass
hydrogen_per_gram = 1 / hydrogen_mass

proton_per_sun = sun_mass / proton_mass
sun_per_proton = 1 / proton_per_sun

hydrogen_per_sun = sun_mass / hydrogen_mass
sun_per_hydrogen = 1 / hydrogen_per_sun

# time
sec_per_yr = year
yr_per_sec = 1 / sec_per_yr

sec_per_Gyr = sec_per_yr * 1e9
Gyr_per_sec = 1 / sec_per_Gyr

# length
cm_per_pc = parsec
pc_per_cm = 1 / cm_per_pc

cm_per_kpc = cm_per_pc * 1e3
kpc_per_cm = 1 / cm_per_kpc

cm_per_Mpc = cm_per_pc * 1e6
Mpc_per_cm = 1 / cm_per_Mpc

km_per_pc = cm_per_pc * kilo_per_centi
pc_per_km = 1 / km_per_pc

km_per_kpc = cm_per_pc * 1e-2
kpc_per_km = 1 / km_per_kpc

km_per_Mpc = cm_per_pc * 10
Mpc_per_km = 1 / km_per_Mpc

# energy
erg_per_ev = electron_volt
ev_per_erg = 1 / erg_per_ev

erg_per_kev = erg_per_ev * 1e3
kev_per_erg = 1 / erg_per_kev

kelvin_per_ev = scipy_const.electron_volt / scipy_const.k
ev_per_kelvin = 1 / kelvin_per_ev

# angle
degree_per_radian = 180 / scipy_const.pi
radian_per_degree = 1 / degree_per_radian

arcmin_per_degree = 60
degree_per_arcmin = 1 / arcmin_per_degree

arcsec_per_arcmin = 60
arcmin_per_arcsec = 1 / arcsec_per_arcmin

arcsec_per_degree = arcmin_per_degree * arcsec_per_arcmin
degree_per_arcsec = degree_per_arcmin * arcmin_per_arcsec

arcsec_per_radian = arcsec_per_arcmin * arcmin_per_degree * degree_per_radian
radian_per_arcsec = 1 / arcsec_per_radian

arcmin_per_radian = arcmin_per_degree * degree_per_radian
radian_per_arcmin = 1 / arcmin_per_radian

deg2_per_sky = 4 * scipy_const.pi * degree_per_radian ** 2

# cosmological constant parameters ----------
# hubble parameter = H_0 [h/sec]
hubble_parameter_0 = 100 * Mpc_per_km
# hubble time = 1 / H_0 ~ 9.7779 [Gyr/h]
hubble_time = 1 / 100 * Gyr_per_sec * km_per_Mpc
# hubble distance = c / H_0 ~ 2,997,925 [kpc/h]
hubble_distance = speed_light / 100 * kilo_per_centi * kilo_per_mega
# critical density at z = 0:  3 * H_0 ^ 2 / (8 * pi * G) ~ 277.5 [M_sun/h / (kpc/h)^3]
density_critical_0 = (3 * 100 ** 2 / (8 * scipy_const.pi * grav) *
                      centi_per_kilo ** 2 / Mpc_per_cm / sun_mass) / kilo_per_mega ** 3

# gravitational constant in various units ----------
# [pc^3 / M_sun / s^2]
grav_pc_msun_sec = grav * pc_per_cm ** 3 * gram_per_sun
# [kpc^3 / M_sun / s^2]
grav_kpc_msun_sec = grav * kpc_per_cm ** 3 * gram_per_sun
# [pc^3 / M_sun / yr^2]
grav_pc_msun_yr = grav * pc_per_cm ** 3 * gram_per_sun * sec_per_yr ** 2
# [kpc^3 / M_sun / yr^2]
grav_kpc_msun_yr = grav * kpc_per_cm ** 3 * gram_per_sun * sec_per_yr ** 2
# [kpc^3 / M_sun / Gyr^2]
grav_kpc_msun_Gyr = grav * kpc_per_cm ** 3 * gram_per_sun * sec_per_Gyr ** 2
