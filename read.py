'''
Reader for Gizmo simulation snapshot files at z = 0.
Designed for use with the Latte suite of FIRE-2 simulations of Milky Way-mass galaxies,
though usable with any Gizmo simulation snapshot files at z = 0.

This is a light-weight version of the more feature-rich Gizmo reader/analysis package that I develop
https://bitbucket.org/awetzel/gizmo_analysis

@author: Andrew Wetzel <arwetzel@gmail.com>


Units

Unless otherwise noted, this reader converts all quantities to the following units
(and combinations thereof):
* mass in [M_sun]
* position, distance, radius in [kpc physical]
* velocity in [km / s]
* time, age in [Gyr]
* elemental abundance in [(linear) mass fraction]
* metallicity in log10(mass_fraction / mass_fraction_solar), assuming Asplund et al 2009 for Solar


Reading a snapshot

Within a simulation directory, read all particles in a snapshot at redshift 0 via:
    part = read.Read.read_snapshot()
part is a dictionary, with a key for each particle species. So, access star particle dictionary via:
    part['star']
part['star'] is dictionary, with each property as a key. For example:
    part['star']['mass']
returns a numpy array of masses, one for each star particle, while
    part['star']['position']
returns a numpy array of positions, of dimension particle_number x 3.


Particle species

The available particle species in a cosmological simulation are:
    part['dark'] : dark matter at the highest resolution
    part['dark.2'] : dark matter at lower resolution (outside of the zoom-in region)
    part['gas'] : gas
    part['star'] : stars


Default/stored particle properties

Access these via:
    part[species_name][property_name]
For example:
    part['star']['position']

All particle species have the following properties:
    'id' : ID (indexing starts at 0)
    'position' : 3-D position wrt galaxy center, aligned with galaxy principal axes [kpc physical]
    'velocity' : 3-D velocity wrt galaxy center, aligned with galaxy principal axes [km / s]
    'mass' : mass [M_sun]
    'potential' : potential (computed using all particles in the simulation) [km^2 / s^2 physical]

Star and gas particles also have:
    'massfraction' : fraction of the mass that is in different elemental abundances,
        stored as an array for each particle, with indexes as follows:
        0 = all metals (everything not H, He)
        1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
    these also are stored as metallicity := log10(mass_fraction / mass_fraction_solar)
        where mass_fraction_solar is from Asplund et al 2009
    'metallicity.total' : everything not H, He
    'metallicity.he' : Helium
    'metallicity.c' : Carbon
    'metallicity.n' : Nitrogen
    'metallicity.o' : Oxygen
    'metallicity.ne' : Neon
    'metallicity.mg' : Magnesium
    'metallicity.si' : Silicon
    'metallicity.s' : Sulfur
    'metallicity.ca' : Calcium
    'metallicity.fe' : Iron

Star particles also have:
    'form.scalefactor' : expansion scale-factor when the star particle formed [0 to 1]
    'age' : current age (t_now - t_form) [Gyr]

Gas particles also have:
    'density' : [M_sun / kpc^3]
    'temperature' : [K]
    'electron.fraction' : free-electron number per proton, averaged over mass of particle
    'hydrogen.neutral.fraction' : fraction of hydrogen that is neutral (not ionized)
    'sfr' : instantaneous star formation rate [M_sun / yr]
    'smooth.length' : smoothing/kernel length, stored as Plummer-equivalent
        (for consistency with force softening) [kpc physical]


Further documenation

Gizmo users guide
comprenehsive documentation of the Gizmo code and contents of simulation snapshots:
http://www.tapir.caltech.edu/~phopkins/Site/GIZMO_files/gizmo_documentation.html

Gizmo source code (publicly available version)
https://bitbucket.org/phopkins/gizmo-public

Hopkins 2015
describes the Gizmo code and MFM hydrodynamics method.
http://adsabs.harvard.edu/abs/2015MNRAS.450...53H

Hopkins et al 2018
describes the FIRE-2 physics model
https://ui.adsabs.harvard.edu/#abs/2017arXiv170206148H

Wetzel et al 2016
introduces the Latte suite of FIRE-2 simulations of Milky Way-mass galaxies
https://ui.adsabs.harvard.edu/#abs/2016ApJ...827L..23W)

Sanderson et al 2018
describes the Ananke framework for generating synthetic Gaia surveys from m12f, m12i, and m12m
of the Latte suite and presents the properties of these simulations in detail
https://arxiv.org/abs/1806.10564v1

FIRE project website
https://fire.northwestern.edu


Citation

If you use any of the Latte FIRE-2 simulations, please including the following citation:

"The Latte suite of FIRE-2 cosmological zoom-in baryonic simulations of Milky Way-mass galaxies
(Wetzel et al 2016), part of the Feedback In Realistic Environments (FIRE) simulation project,
were run using the Gizmo gravity plus hydrodynamics code in meshless finite-mass (MFM) mode
(Hopkins 2015) and the FIRE-2 physics model (Hopkins et al 2018)."
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatibility
import collections
import glob
import h5py
import numpy as np
from scipy import integrate, interpolate
# local ----
from . import center, constant, coordinate


# store particles as dictionary class
class DictClass(dict):
    pass


class ReadClass():
    '''
    Read Gizmo snapshot.
    '''

    def __init__(self):
        '''
        Set properties for snapshot files.
        '''
        self.snapshot_name_base = 'snap*[!txt]'  # avoid accidentally reading snapshot indices file
        self.file_extension = '.hdf5'

        self.gas_eos = 5 / 3  # gas equation of state

        # create ordered dictionary to convert particle species name to its id,
        # set all possible species, and set the order in which to read species
        self.species_dict = collections.OrderedDict()
        # dark-matter species
        self.species_dict['dark'] = 1  # dark matter at highest resolution
        self.species_dict['dark.2'] = 2  # dark matter at all lower resolutions
        # baryon species
        self.species_dict['gas'] = 0
        self.species_dict['star'] = 4

        self.species_all = tuple(self.species_dict.keys())
        self.species_read = list(self.species_all)

        # use to translate between element name and index in element table
        self.element_dict = {}
        self.element_dict['total'] = 0
        self.element_dict['he'] = 1
        self.element_dict['c'] = 2
        self.element_dict['n'] = 3
        self.element_dict['o'] = 4
        self.element_dict['ne'] = 5
        self.element_dict['mg'] = 6
        self.element_dict['si'] = 7
        self.element_dict['s'] = 8
        self.element_dict['ca'] = 9
        self.element_dict['fe'] = 10

    def read_snapshot(
        self, species='all', properties='all', directory='.', particle_subsample_factor=None):
        '''
        Read properties for input particle species from simulation snapshot file[s].
        Return particle catalog as a dictionary class.

        Parameters
        ----------
        species : string or list : name[s] of particle species:
            'all' = all species in file
            'star' = stars
            'gas' = gas
            'dark' = dark matter at highest resolution
            'dark.2' = dark matter at lower resolution
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, list subset from among read_particles.property_dict
                for example: ['mass', 'position', 'velocity']
        directory : string : directory of snapshot file[s]
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory

        Returns
        -------
        part : dictionary class : catalog of particles at snapshot
        '''
        snapshot_index = 600  # corresponds to z = 0

        # parse input species to read
        if species == 'all' or species == ['all'] or not species:
            # read all species in snapshot
            species = self.species_all
        else:
            # read subsample of species in snapshot
            if np.isscalar(species):
                species = [species]  # ensure is list
            # check if input species names are valid
            for spec_name in list(species):
                if spec_name not in self.species_dict:
                    species.remove(spec_name)
                    print('! not recognize input species = {}'.format(spec_name))
        self.species_read = list(species)

        # read header from snapshot file
        header = self.read_header(snapshot_index, directory)

        # read particles from snapshot file[s]
        part = self.read_particles(snapshot_index, directory, properties, header)

        # assign auxilliary information to particle dictionary class
        # store header dictionary
        part.info = header
        for spec_name in part:
            part[spec_name].info = part.info

        # get and store cosmological parameters
        part.Cosmology = CosmologyClass(
            header['omega_lambda'], header['omega_matter'], hubble=header['hubble'])
        for spec_name in part:
            part[spec_name].Cosmology = part.Cosmology

        # store information about snapshot time
        time = part.Cosmology.get_time(header['redshift'], 'redshift')
        part.snapshot = {
            'index': snapshot_index,
            'redshift': header['redshift'],
            'scalefactor': header['scalefactor'],
            'time': time,
            'time.lookback': part.Cosmology.get_time(0) - time,
            'time.hubble': constant.Gyr_per_sec / part.Cosmology.get_hubble_parameter(0),
        }
        for spec_name in part:
            part[spec_name].snapshot = part.snapshot

        # adjust properties for each species
        self.adjust_particle_properties(part, header, particle_subsample_factor)

        # assign galaxy center position and velocity, principal axes rotation vectors
        self.read_galaxy_center_coordinates(part, directory)
        # alternately can assign these on the fly
        #center.assign_center(part)
        #center.assign_principal_axes(part)

        # adjust coordinates to be relative to galaxy center position and velocity
        # and aligned with principal axes
        self.adjust_particle_coordinates(part)

        return part

    def read_header(self, snapshot_index=600, directory='.'):
        '''
        Read header from snapshot file.

        Parameters
        ----------
        snapshot_index : int : index (number) of snapshot file
        directory : directory of snapshot

        Returns
        -------
        header : dictionary class : header dictionary
        '''
        # convert name in snapshot's header dictionary to custom name preference
        header_dict = {
            # 6-element array of number of particles of each type in file
            'NumPart_ThisFile': 'particle.numbers.in.file',
            # 6-element array of total number of particles of each type (across all files)
            'NumPart_Total': 'particle.numbers.total',
            'NumPart_Total_HighWord': 'particle.numbers.total.high.word',
            # mass of each particle species, if all particles are same
            # (= 0 if they are different, which is usually true)
            'MassTable': 'particle.masses',
            'Time': 'time',  # [Gyr/h]
            'BoxSize': 'box.length',  # [kpc/h comoving]
            'Redshift': 'redshift',
            # number of output files per snapshot
            'NumFilesPerSnapshot': 'file.number.per.snapshot',
            'Omega0': 'omega_matter',
            'OmegaLambda': 'omega_lambda',
            'HubbleParam': 'hubble',
            'Flag_Sfr': 'has.star.formation',
            'Flag_Cooling': 'has.cooling',
            'Flag_StellarAge': 'has.star.age',
            'Flag_Metals': 'has.metals',
            'Flag_Feedback': 'has.feedback',
            'Flag_DoublePrecision': 'has.double.precision',
            'Flag_IC_Info': 'has.ic.info',
            # level of compression of snapshot file
            'CompactLevel': 'compression.level',
            'Compactify_Version': 'compression.version',
            'ReadMe': 'compression.readme',
        }

        header = {}  # dictionary to store header information

        if directory[-1] != '/':
            directory += '/'

        file_name = self.get_snapshot_file_name(directory, snapshot_index)

        print('reading header from:\n  {}'.format(file_name.replace('./', '')))
        print()

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            header_in = file_in['Header'].attrs  # load header dictionary

            for prop_in in header_in:
                prop = header_dict[prop_in]
                header[prop] = header_in[prop_in]  # transfer to custom header dict

        # convert header quantities
        header['scalefactor'] = float(header['time'])
        del(header['time'])
        header['box.length/h'] = float(header['box.length'])
        header['box.length'] /= header['hubble']  # convert to [kpc comoving]

        print('snapshot contains the following number of particles:')
        # keep only species that have any particles
        read_particle_number = 0

        species_read = list(self.species_read)
        for species_name in species_read:
            if species_name not in self.species_all:
                species_read.append(species_name)

        for spec_name in species_read:
            spec_id = self.species_dict[spec_name]
            print('  {:6s} (id = {}): {} particles'.format(
                  spec_name, spec_id, header['particle.numbers.total'][spec_id]))

            if header['particle.numbers.total'][spec_id] > 0:
                read_particle_number += header['particle.numbers.total'][spec_id]
            elif spec_name in self.species_read:
                self.species_read.remove(spec_name)

        if read_particle_number <= 0:
            raise ValueError('snapshot file[s] contain no particles of species = {}'.format(
                             self.species_read))

        print()

        return header

    def read_particles(self, snapshot_index=600, directory='.', properties='all', header=None):
        '''
        Read particles from snapshot file[s].

        Parameters
        ----------
        snapshot_index : int : index (number) of snapshot file
        directory : directory of snapshot
        properties : string or list : name[s] of particle properties to read - options:
            'all' = all species in file
            otherwise, list subset from among read_particles.property_dict
                for example: ['mass', 'position', 'velocity']
        header : dict : snapshot header dictionary

        Returns
        -------
        part : dictionary class : catalog of particles at snapshot
        '''
        # convert name in snapshot's particle dictionary to custon name preference
        # if comment out any prop, will not read it
        property_dict = {
            ## all particles ----------
            'ParticleIDs': 'id',  # indexing starts at 0
            'Coordinates': 'position',
            'Velocities': 'velocity',
            'Masses': 'mass',
            'Potential': 'potential',

            ## star/gas particles ----------
            ## mass fraction of individual elements ----------
            ## 0 = all metals (everything not H, He)
            ## 1 = He, 2 = C, 3 = N, 4 = O, 5 = Ne, 6 = Mg, 7 = Si, 8 = S, 9 = Ca, 10 = Fe
            'Metallicity': 'massfraction',

            ## star particles ----------
            ## 'time' when star particle formed
            ## for cosmological runs, = scale-factor; for non-cosmological runs, = time [Gyr/h]
            'StellarFormationTime': 'form.scalefactor',

            ## gas particles ----------
            'InternalEnergy': 'temperature',
            'Density': 'density',
            # stored in snapshot file as maximum distance to neighbor (radius of compact support)
            # but here convert to Plummer-equivalent length (for consistency with force softening)
            #'SmoothingLength': 'smooth.length',
            # average free-electron number per proton, averaged over mass of gas particle
            'ElectronAbundance': 'electron.fraction',
            # fraction of hydrogen that is neutral (not ionized)
            'NeutralHydrogenAbundance': 'hydrogen.neutral.fraction',
            #'StarFormationRate': 'sfr',  # [M_sun / yr]
        }

        part = DictClass()  # dictionary class to store properties for particle species

        if directory[-1] != '/':
            directory += '/'

        # parse input list of properties to read
        if properties == 'all' or properties == ['all'] or not properties:
            properties = list(property_dict.keys())
        else:
            if np.isscalar(properties):
                properties = [properties]  # ensure is list
            # make safe list of properties to read
            properties_temp = []
            for prop in list(properties):
                prop = str.lower(prop)
                if 'massfraction' in prop or 'metallicity' in prop:
                    prop = 'massfraction'  # this has several aliases, so ensure default name
                for prop_in in property_dict:
                    if prop in [str.lower(prop_in), str.lower(property_dict[prop_in])]:
                        properties_temp.append(prop_in)
            properties = properties_temp
            del(properties_temp)

        if 'InternalEnergy' in properties:
            # need helium mass fraction and electron fraction to compute temperature
            for prop in np.setdiff1d(['ElectronAbundance', 'Metallicity'], properties):
                properties.append(prop)

        if not header:
            header = self.read_header(snapshot_index, directory)

        file_name = self.get_snapshot_file_name(directory, snapshot_index)

        # open snapshot file
        with h5py.File(file_name, 'r') as file_in:
            part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

            # initialize arrays to store each prop for each species
            for spec_name in self.species_read:
                spec_id = self.species_dict[spec_name]
                part_number_tot = header['particle.numbers.total'][spec_id]

                # add species to particle dictionary
                part[spec_name] = DictClass()

                # check if snapshot file happens not to have particles of this species
                if part_numbers_in_file[spec_id] > 0:
                    part_in = file_in['PartType' + str(spec_id)]
                else:
                    # this scenario should occur only for multi-file snapshot
                    if header['file.number.per.snapshot'] == 1:
                        raise ValueError('no {} particles in snapshot file'.format(spec_name))

                    # need to read in other snapshot files until find one with particles of species
                    for file_i in range(1, header['file.number.per.snapshot']):
                        file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))
                        # try each snapshot file
                        file_in_i = h5py.File(file_name_i, 'r')
                        part_numbers_in_file_i = file_in_i['Header'].attrs['NumPart_ThisFile']
                        if part_numbers_in_file_i[spec_id] > 0:
                            # found one
                            part_in = file_in_i['PartType' + str(spec_id)]
                            break
                    else:
                        # tried all files and still did not find particles of species
                        raise ValueError('no {} particles in any snapshot file'.format(spec_name))

                props_print = []
                ignore_flag = False  # whether ignored any properties in the file
                for prop_in in part_in.keys():
                    if prop_in in properties:
                        prop = property_dict[prop_in]

                        # determine shape of property array
                        if len(part_in[prop_in].shape) == 1:
                            prop_shape = part_number_tot
                        elif len(part_in[prop_in].shape) == 2:
                            prop_shape = [part_number_tot, part_in[prop_in].shape[1]]

                        # determine data type to store
                        prop_in_dtype = part_in[prop_in].dtype

                        # initialize to -1's
                        part[spec_name][prop] = np.zeros(prop_shape, prop_in_dtype) - 1

                        if prop == 'id':
                            # initialize so calling an un-itialized value leads to error
                            part[spec_name][prop] -= part_number_tot

                        if prop_in in property_dict:
                            props_print.append(property_dict[prop_in])
                        else:
                            props_print.append(prop_in)
                    else:
                        ignore_flag = True

                if ignore_flag:
                    props_print.sort()
                    print('reading {} properties:\n  {}'.format(spec_name, props_print))

                # special case: particle mass is fixed and given in mass array in header
                if 'Masses' in properties and 'Masses' not in part_in:
                    prop = property_dict['Masses']
                    part[spec_name][prop] = np.zeros(part_number_tot, dtype=np.float32)

        ## read properties for each species ----------
        # initial particle indices to assign to each species from each file
        part_indices_lo = np.zeros(len(self.species_read), dtype=np.int64)

        print()
        if header['file.number.per.snapshot'] == 1:
            print('reading particles from:  {}'.format(file_name.strip('./')))
        else:
            print('reading particles from:')

        # loop over all files at given snapshot
        for file_i in range(header['file.number.per.snapshot']):
            # open i'th of multiple files for snapshot
            file_name_i = file_name.replace('.0.', '.{}.'.format(file_i))

            # open snapshot file
            with h5py.File(file_name_i, 'r') as file_in:
                if header['file.number.per.snapshot'] > 1:
                    print('  ' + file_name_i.split('/')[-1])

                part_numbers_in_file = file_in['Header'].attrs['NumPart_ThisFile']

                # read particle properties
                for spec_i, spec_name in enumerate(self.species_read):
                    spec_id = self.species_dict[spec_name]
                    if part_numbers_in_file[spec_id] > 0:
                        part_in = file_in['PartType' + str(spec_id)]

                        part_index_lo = part_indices_lo[spec_i]
                        part_index_hi = part_index_lo + part_numbers_in_file[spec_id]

                        # check if mass of species is fixed, according to header mass array
                        if 'Masses' in properties and header['particle.masses'][spec_id] > 0:
                            prop = property_dict['Masses']
                            part[spec_name][prop][
                                part_index_lo:part_index_hi] = header['particle.masses'][spec_id]

                        for prop_in in part_in.keys():
                            if prop_in in properties:
                                prop = property_dict[prop_in]
                                if len(part_in[prop_in].shape) == 1:
                                    part[spec_name][prop][part_index_lo:part_index_hi] = (
                                        part_in[prop_in])
                                elif len(part_in[prop_in].shape) == 2:
                                    prop_in = part_in[prop_in]
                                    part[spec_name][prop][part_index_lo:part_index_hi, :] = prop_in

                        part_indices_lo[spec_i] = part_index_hi  # set indices for next file

        print()

        return part

    def adjust_particle_properties(self, part, header, particle_subsample_factor=None):
        '''
        Adjust properties for each species, including unit conversions, and sub-sampling.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        header : dict : header dictionary
        particle_subsample_factor : int : factor to periodically subsample particles, to save memory
        '''
        # sub-sample particles for smaller memory
        if particle_subsample_factor is not None and particle_subsample_factor > 1:
            print('periodically subsampling all particles by factor = {}\n'.format(
                  particle_subsample_factor))
            for spec_name in part:
                for prop in part[spec_name]:
                    part[spec_name][prop] = part[spec_name][prop][::particle_subsample_factor]

        for spec_name in part:
            if 'position' in part[spec_name]:
                # convert to [kpc comoving]
                part[spec_name]['position'] /= header['hubble']

            if 'velocity' in part[spec_name]:
                # convert to [km / s]
                part[spec_name]['velocity'] *= np.sqrt(header['scalefactor'])

            if 'mass' in part[spec_name]:
                # convert to [M_sun]
                part[spec_name]['mass'] *= 1e10 / header['hubble']

            if 'potential' in part[spec_name]:
                # convert to [km^2 / s^2 physical]
                part[spec_name]['potential'] /= header['scalefactor']

            if 'form.scalefactor' in part[spec_name]:
                # assign current age, relative to formation [Gyr]
                # initialize (to ensure same precision)
                part[spec_name]['age'] = np.zeros(part[spec_name]['form.scalefactor'].size,
                                                  part[spec_name]['form.scalefactor'].dtype)
                part[spec_name]['age'][:] = part.snapshot['time'] - part.Cosmology.get_time(
                    part[spec_name]['form.scalefactor'], 'scalefactor')

            if 'massfraction' in part[spec_name]:
                # assign metallicity := log10(mass_fraction / mass_fraction_solar) for each species
                for element_name in self.element_dict:
                    element_index = self.element_dict[element_name]
                    # initialize (to ensure same precision)
                    part[spec_name]['metallicity.' + element_name] = np.zeros(
                        part[spec_name]['massfraction'][:, element_index].size,
                        part[spec_name]['massfraction'][:, element_index].dtype)
                    part[spec_name]['metallicity.' + element_name][:] = np.log10(
                        part[spec_name]['massfraction'][:, element_index] /
                        constant.sun_composition[element_name]['massfraction'])

            if 'density' in part[spec_name]:
                # convert to [M_sun / kpc^3 physical]
                part[spec_name]['density'] *= (
                    1e10 / header['hubble'] / (header['scalefactor'] / header['hubble']) ** 3)

            if 'smooth.length' in part[spec_name]:
                # convert to [pc physical]
                part[spec_name]['smooth.length'] *= 1000 * header['scalefactor'] / header['hubble']
                # convert to Plummer softening:
                #   factor of 2.8 is valid for using cubic spline (our default)
                # alternately, to convert to Gaussian scale length, divide by 2
                part[spec_name]['smooth.length'] /= 2.8

            if 'temperature' in part[spec_name]:
                # convert from [(km / s) ^ 2] to [Kelvin]
                # ignore small corrections from elements beyond He
                helium_mass_fracs = part[spec_name]['massfraction'][:, 1]
                ys_helium = helium_mass_fracs / (4 * (1 - helium_mass_fracs))
                mus = (1 + 4 * ys_helium) / (1 + ys_helium + part[spec_name]['electron.fraction'])
                molecular_weights = mus * constant.proton_mass
                part[spec_name]['temperature'] *= (
                    constant.centi_per_kilo ** 2 * (self.gas_eos - 1) * molecular_weights /
                    constant.boltzmann)
                del(helium_mass_fracs, ys_helium, mus, molecular_weights)

    def adjust_particle_coordinates(self, part):
        '''
        Adjust particle positions and velocities to be relative to the galaxy center,
        and aligned with the galaxy's principal axes.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        '''
        print('adjusting particle coordinates to be relative to galaxy center')
        print('  and aligned with the principal axes\n')

        for spec_name in part:
            if 'velocity' in part[spec_name]:
                # convert to be relative to galaxy center [km / s]
                part[spec_name]['velocity'] = coordinate.get_velocity_differences(
                    part[spec_name]['velocity'], part.center_velocity,
                    part[spec_name]['position'], part.center_position,
                    part.info['box.length'], part.snapshot['scalefactor'],
                    part.snapshot['time.hubble'])

                # convert to be aligned with galaxy principal axes
                part[spec_name]['velocity'] = coordinate.get_coordinates_rotated(
                    part[spec_name]['velocity'], part.principal_axes_vectors)

            if 'position' in part[spec_name]:
                # convert to be relative to galaxy center [kpc physical]
                part[spec_name]['position'] = coordinate.get_distances(
                    part[spec_name]['position'], part.center_position,
                    part.info['box.length'], part.snapshot['scalefactor'])

                # convert to be aligned with galaxy principal axes
                part[spec_name]['position'] = coordinate.get_coordinates_rotated(
                    part[spec_name]['position'], part.principal_axes_vectors)

    def read_galaxy_center_coordinates(self, part, directory='.'):
        '''
        Read pre-computed galaxy center position and velocity,
        as well as principal axes rotation vectors.
        Generally, this should be a file named:  m12*_res7100_center.txt
        Append to particle catalog dictionary class.

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        directory : string : directory of file
        '''
        file_name_base = 'm*_center.txt'  # base name (with wildcard) of expected file

        if directory[-1] != '/':
            directory += '/'

        file_name_find = directory + file_name_base

        try:
            file_names = self.get_file_names(file_name_find)
            file_name = file_names[0]
            print('reading galaxy center coordinates and principal axes from:  {}'.format(
                  file_name.strip('./')))
            # read from file
            data_in = np.loadtxt(file_name, np.float64, comments='#', delimiter=',')

            # append to particle dictionary class
            part.center_position = data_in[0]
            for spec_name in part:
                part[spec_name].center_position = part.center_position

            print('  center position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
                  part.center_position[0], part.center_position[1], part.center_position[2]))

            part.center_velocity = data_in[1].astype(np.float32)
            for spec_name in part:
                part[spec_name].center_velocity = part.center_velocity

            print('  center velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
                  part.center_velocity[0], part.center_velocity[1], part.center_velocity[2]))

            part.principal_axes_vectors = data_in[2:5]
            for spec_name in part:
                part[spec_name].principal_axes_vectors = part.principal_axes_vectors

        except ValueError:
            print('! cannot find file with name base:  {}'.format(file_name_find))
            print('  that contains galaxy center coordinates and principal axes')
            print('  instead will compute these from the particle catalog')
            center.assign_center(part)
            center.assign_principal_axes(part)

        print()

    def read_lsr_coordinates(self, part, directory='.', lsr_index=0):
        '''
        Read pre-computed solar / LSR (local standard of rest) position and velocity,
        in galactocentric coordinates, aligned with galaxy's principal axes.
        Generally, this should be a file named:  m12*_res7100_LSR*.txt
        Append to particle catalog dictionary class.

        Then you can convert coordinates to be relative to LSR via:
           positions(wrt LSR) = part[spec_name]['position'] - part.lsr_position
           velocities(wrt LSR) = part[spec_name]['velocity'] - part.lsr_velocity

        Parameters
        ----------
        part : dictionary class : catalog of particles at snapshot
        directory : string : directory of file
        lsr_index : int : index of LSR coordinate to read (currently 0, 1, or 2)
        '''
        # base name (with wildcard) of expected file
        file_name_base = 'm12*_res7100_LSR{}.txt'.format(lsr_index)

        if directory[-1] != '/':
            directory += '/'

        file_name_find = directory + file_name_base

        try:
            file_names = self.get_file_names(file_name_find)
            file_name = file_names[0]
            print('reading LSR coordinates from:\n  {}'.format(file_name.strip('./')))
            # read from file
            data_in = np.loadtxt(file_name, np.float64, comments='#', delimiter=',')

            # append to particle dictionary class
            part.lsr_position = data_in[0:3]
            for spec_name in part:
                part[spec_name].lsr_position = part.lsr_position

            print('  LSR_{} position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
                  lsr_index, part.lsr_position[0], part.lsr_position[1], part.lsr_position[2]))

            part.lsr_velocity = data_in[3:6].astype(np.float32)
            for spec_name in part:
                part[spec_name].lsr_velocity = part.lsr_velocity

            print('  LSR_{} velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
                  lsr_index, part.lsr_velocity[0], part.lsr_velocity[1], part.lsr_velocity[2]))

        except ValueError:
            print('! cannot find file with name base:  {}'.format(file_name_find))
            print('  that contains LSR coordinates')

        print()

    def get_snapshot_file_name(self, directory, snapshot_index):
        '''
        Get name (with relative path) of file to read in.
        If multiple files per snapshot, get name of 0th one.

        Parameters
        ----------
        directory : string : directory to check for files
        snapshot_index : int : index of snapshot

        Returns
        -------
        path_file_name : string : file name (with relative path)
        '''
        if directory[-1] != '/':
            directory += '/'

        path_names, file_indices = self.get_file_names(
            directory + self.snapshot_name_base, (int, float))

        if snapshot_index < 0:
            snapshot_index = file_indices[snapshot_index]  # allow negative indexing of snapshots
        elif snapshot_index not in file_indices:
            raise ValueError('cannot find snapshot index = {} in: {}'.format(
                             snapshot_index, path_names))

        path_name = path_names[np.where(file_indices == snapshot_index)[0][0]]

        if self.file_extension in path_name:
            # got actual file, so good to go
            path_file_name = path_name
        else:
            # got snapshot directory with multiple files, return only 0th one
            path_file_names = self.get_file_names(path_name + '/' + self.snapshot_name_base)
            if len(path_file_names) and '.0.' in path_file_names[0]:
                path_file_name = path_file_names[0]
            else:
                raise ValueError('cannot find 0th snapshot file in ' + path_file_names)

        return path_file_name

    def get_file_names(self, file_name_base, number_type=None, sort_reverse=False):
        '''
        Get all file name[s] (with full path) with given name base
        [and number in each file name, if number_type defined], sorted.

        Parameters
        ----------
        file_name_base : string : base name of file, with full/relative path, using * as wildcard
        number_type : dtype : type of number to get in file name (get final one in name)
            options: None, int, float, (int, float), numbers.Real
        sort_reverse : boolean : whether to return list of file names and numbers in reverse order

        Returns
        -------
        path_file_names : string : name[s] of file[s], with full path
        [file_numbers] : int[s] and/or float[s] : number in file name
        '''
        # get all file names matching string in directory
        path_file_names = glob.glob(file_name_base)
        path_file_names.sort()
        if not path_file_names:
            raise ValueError('found no files with base name: ' + file_name_base)

        if number_type:
            # for file names with number, get final number of given type in each file name
            file_numbers = []
            for path_file_name in path_file_names:

                file_name = path_file_name
                if '/' in file_name:
                    file_name = file_name.split('/')[-1]
                if '.hdf5' in file_name:
                    file_name = file_name.replace('.hdf5', '')

                file_numbers_t = self.get_numbers_in_string(file_name, scalarize=False)

                for file_number_t in reversed(file_numbers_t):
                    if isinstance(file_number_t, number_type):
                        file_numbers.append(file_number_t)
                        break
                else:
                    raise ValueError(
                        'no number of type {} in file: {}'.format(number_type, path_file_name))

            file_numbers = np.array(file_numbers)

            if sort_reverse:
                path_file_names = path_file_names[::-1]
                file_numbers = file_numbers[::-1]

            return path_file_names, file_numbers

        else:
            if len(path_file_names) > 1 and sort_reverse:
                path_file_names = path_file_names[::-1]

            return path_file_names

    def get_numbers_in_string(self, string, scalarize=False):
        '''
        Get list of int and float numbers in string.

        Parameters
        ----------
        string : string
        scalarize : boolean : whether to return scalar value if only one number

        Returns
        -------
        numbers : int[s] and/or float[s]
        '''
        numbers = []
        number = ''

        for ci, char in enumerate(string):
            if char.isdigit():
                number += char
            elif char == '.':
                if (number and ci > 0 and string[ci - 1].isdigit() and len(string) > ci + 1 and
                        string[ci + 1].isdigit()):
                    number += char

            if number and ((not char.isdigit() and not char == '.') or ci == len(string) - 1):
                if '.' in number:
                    numbers.append(float(number))
                else:
                    numbers.append(int(number))
                number = ''

        if scalarize and len(numbers) == 1:
            numbers = numbers[0]

        return numbers


Read = ReadClass()


class CosmologyClass(dict):
    '''
    Class to store cosmological parameters and cosmology functions.

    Parameters
    ----------
    dictionary class (to allow this class to store as if a dictionary)
    '''

    def __init__(
        self, omega_lambda=0.702, omega_matter=0.272, omega_baryon=0.0455, hubble=0.702,
        sigma_8=0.807, n_s=0.961, w=-1.0):
        '''
        Store cosmology parameters.
        Default values are from AGORA simulation box, used to select Latte halos.

        Parameters
        ----------
        omega_lambda : float : Omega_lambda(z = 0)
        omega_matter : float : Omega_matter(z = 0)
        omega_baryon : float : Omega_baryon(z = 0)
        hubble : float : dimensionless hubble constant (at z = 0)
        sigma_8 : float : sigma_8(z = 0)
        n_s : float : index (slope) of primordial power spectrum
        w : float : dark energy equation of state
        '''
        self['omega_lambda'] = omega_lambda
        self['omega_matter'] = omega_matter
        self['omega_baryon'] = omega_baryon
        self['omega_curvature'] = 1 - self['omega_matter'] - self['omega_lambda']
        self['omega_dm'] = self['omega_matter'] - self['omega_baryon']
        self['baryon.fraction'] = self['omega_baryon'] / self['omega_matter']
        self['hubble'] = hubble
        self['sigma_8'] = sigma_8
        self['n_s'] = n_s
        self['w'] = w
        self.TimeScalefactorSpline = self.get_time_v_scalefactor_spline()

    def get_hubble_parameter(self, redshifts):
        '''
        Get Hubble parameter[s] [sec ^ -1] at redshift[s].

        Parameters
        ----------
        redshifts : float or array : redshift[s]
        '''
        return (constant.hubble_parameter_0 * self['hubble'] *
                (self['omega_matter'] * (1 + redshifts) ** 3 + self['omega_lambda'] +
                 self['omega_curvature'] * (1 + redshifts) ** 2) ** 0.5)

    def get_time_v_scalefactor_spline(self, scalefactor_limits=[0.01, 1.01], number=500):
        '''
        Make and store spline to get time [Gyr] from scale-factor.
        Use scale-factor (as opposed to redshift) because it is more stable:
        time maps ~linearly onto scale-factor.

        Parameters
        ----------
        scalefactor_limits : list : min and max limits of scale-factor
        number : int : number of spline points within limits
        '''
        scalefactors = np.linspace(scalefactor_limits[0], scalefactor_limits[1], number)

        times = np.zeros(number)
        for a_i, a in enumerate(scalefactors):
            times[a_i] = self.get_time(a, 'scalefactor')

        return interpolate.splrep(scalefactors, times)

    def get_time(self, values, value_kind='scalefactor'):
        '''
        Get time[s] [Gyr] at redshift[s] or scale-factor[s].
        Big Bang corresponds to time = 0.

        Parameters
        ----------
        values : float or array : redshift[s] or scale-factor[s]
        value_kind : string : 'redshift' or 'scalefactor'

        Returns
        -------
        times : float or array : time[s] [Gyr]
        '''

        def get_dt(scalefactor, self):
            return (self['omega_matter'] / scalefactor + self['omega_lambda'] * scalefactor ** 2 +
                    self['omega_curvature']) ** -0.5

        if not np.isscalar(values):
            values = np.asarray(values)

        if value_kind == 'scalefactor':
            scalefactors = values
        elif value_kind == 'redshift':
            scalefactors = 1 / (1 + values)
        else:
            raise ValueError('not recognize value_kind = {}'.format(value_kind))

        if np.isscalar(scalefactors):
            times = constant.hubble_time / self['hubble'] * integrate.quad(
                get_dt, 1e-10, scalefactors, (self))[0]
        else:
            times = interpolate.splev(scalefactors, self.TimeScalefactorSpline)

        return times
