'''
Compute galaxy center, principal axes, and LSR coordinates from Gizmo snapshot,
for use with Latte suite of FIRE-2 simulations of MW-mass galaxies.

@author: Andrew Wetzel <arwetzel@gmail.com>

Units
    mass in [M_sun]
    position, distance, radius in [kpc physical]
    velocity in [km / s]
    time, age in [Gyr]
'''

# system ----
from __future__ import absolute_import, division, print_function  # python 2 compatability
import numpy as np
# local ----
from . import coordinate

# assumed positions of Sun / LSR (local standard of rest), in cylindical coordinates
sun_radius = 8.2  # radius [kpc]
sun_height = 0  # vertical height [kpc]
sun_phi_0 = 1 / 4 * (2 * np.pi)  # initial angle [radian]
#sun_phi_0 = 0  # initial angle [radian]
#sun_phi_0 = 1 / 12 * (2 * np.pi)  # initial angle [radian]
#sun_phi_0 = 1 / 6 * (2 * np.pi)  # initial angle [radian]

# generate positions evenly spaced in azimuthal angle
sun_position_number = 3
sun_positions = [
    [sun_radius, sun_height, sun_phi_0 + i / sun_position_number * (2 * np.pi)]
    for i in range(sun_position_number)
]


def assign_center(part):
    '''
    Assign center position [kpc comoving] and velocity [km / s] to galaxy.

    Parameters
    ----------
    part : dictionary class : catalog of particles
    '''
    spec_name = 'star'  # particle species to use to compute center
    velocity_radius_max = 15  # compute average velocity using particles within this radius [kpc]

    print('assigning center of galaxy:')

    # assign to particle dictionary class
    part.center_position = coordinate.get_center_position_zoom(
        part[spec_name]['position'], part[spec_name]['mass'], part.info['box.length'])
    # assign to each species dictionary class
    for spec_name in part:
        part[spec_name].center_position = part.center_position

    print('  center position [kpc] = {:.3f}, {:.3f}, {:.3f}'.format(
        part.center_position[0], part.center_position[1], part.center_position[2]))

    if 'velocity' in part[spec_name]:
        # assign to particle dictionary class
        part.center_velocity = coordinate.get_center_velocity(
            part[spec_name]['velocity'], part[spec_name]['mass'], part[spec_name]['position'],
            part.center_position, velocity_radius_max, part.info['box.length'])
        # assign to each species dictionary class
        for spec_name in part:
            part[spec_name].center_velocity = part.center_velocity

        print('  center velocity [km/s] = {:.1f}, {:.1f}, {:.1f}'.format(
            part.center_velocity[0], part.center_velocity[1], part.center_velocity[2]))

    print()


def assign_principal_axes(part, radius_max=sun_radius, age_limits=[0, 1]):
    '''
    Assign principal axes (rotation vectors defined by moment of inertia tensor) to galaxy.

    Parameters
    ----------
    part : dictionary class : catalog of particles
    distance_max : float : maximum radius to select particles [kpc physical]
    age_limits : float : min and max limits of age to select star particles [Gyr]
    '''
    spec_name = 'star'  # particle species to use to compute MOI tensor and principal axes

    if spec_name not in part or not len(part[spec_name]['position']):
        print('! catalog not contain star particles, so cannot assign principal axes')
        return

    print('assigning principal axes:')
    print('  using {} particles at radius < {} kpc'.format(spec_name, radius_max))
    print('  using {} particles with age = {} Gyr'.format(spec_name, age_limits))

    if ('form.scalefactor' not in part[spec_name] or
            not len(part[spec_name]['form.scalefactor'])):
        raise ValueError('! catalog not contain ages for {} particles'.format(spec_name))

    # get particles within age limits
    if age_limits is not None and len(age_limits):
        try:
            ages = part[spec_name]['age']
        except Exception:
            ages = part[spec_name].prop('age')
        part_indices = np.where(
            (ages >= min(age_limits)) * (ages < max(age_limits)))[0]
    else:
        part_indices = np.arange(part[spec_name]['mass'])

    # store galaxy center
    center_position = part[spec_name].center_position
    center_velocity = part[spec_name].center_velocity

    # compute radii wrt galaxy center [kpc physical]
    radius_vectors = coordinate.get_distances(
        part[spec_name]['position'][part_indices], center_position,
        part.info['box.length'], part.snapshot['scalefactor'])

    # keep only particles within radius_max
    radius2s = np.sum(radius_vectors ** 2, 1)
    masks = (radius2s < radius_max ** 2)

    radius_vectors = radius_vectors[masks]
    part_indices = part_indices[masks]

    # compute rotation vectors for principal axes (defined via moment of inertia tensor)
    rotation_vectors, _eigen_values, axes_ratios = coordinate.get_principal_axes(
        radius_vectors, part[spec_name]['mass'][part_indices], print_results=False)

    # test if need to flip principal axis to ensure that v_phi is defined as moving
    # clockwise as seen from + Z (silly Galactocentric convention)
    velocity_vectors = coordinate.get_velocity_differences(
        part[spec_name]['velocity'][part_indices], center_velocity)
    velocity_vectors_rot = coordinate.get_coordinates_rotated(velocity_vectors, rotation_vectors)
    radius_vectors_rot = coordinate.get_coordinates_rotated(radius_vectors, rotation_vectors)
    velocity_vectors_cyl = coordinate.get_velocities_in_coordinate_system(
        velocity_vectors_rot, radius_vectors_rot, 'cartesian', 'cylindrical')
    if np.median(velocity_vectors_cyl[:, 2]) > 0:
        rotation_vectors[0] *= -1  # flip v_phi
    else:
        rotation_vectors[0] *= -1  # consistency for m12f
        rotation_vectors[1] *= -1

    # store in particle catalog
    part.principal_axes_vectors = rotation_vectors
    part.principal_axes_ratios = axes_ratios
    for spec_name in part:
        part[spec_name].principal_axes_vectors = rotation_vectors
        part[spec_name].principal_axes_ratios = axes_ratios

    print('  axis ratios:  min/maj = {:.3f}, min/med = {:.3f}, med/maj = {:.3f}'.format(
          axes_ratios[0], axes_ratios[1], axes_ratios[2]))
    print()


def print_center_coordinates_principal_axes(part, radius_max=sun_radius, age_limits=[0, 1]):
    '''
    Print galaxy center coordinates and principal axes rotation vectors.

    Parameters
    ----------
    part : dictionary class : catalog of particles
    distance_max : float : maximum radius to select particles [kpc physical]
    age_limits : float : min and max limits of age to select star particles [Gyr]
    '''
    spec_name = 'star'  # particle species to use to compute center and principal axes

    # store galaxy center
    center_position = part[spec_name].center_position
    center_velocity = part[spec_name].center_velocity

    if 'simulation.name' in part.info:
        print('# {}'.format(part.info['simulation.name']).replace(' r7100', '_res7100'))
    print('# center position(x, y, z) [kpc]\n{:.5f}, {:.5f}, {:.5f}'.format(
          center_position[0], center_position[1], center_position[2]))
    print('# center velocity(v_x, v_y, v_z) [km/s]\n{:.5f}, {:.5f}, {:.5f}'.format(
          center_velocity[0], center_velocity[1], center_velocity[2]))
    print('# principal axes rotation vectors')
    print('# defined via {} particles with r < {:.1f} kpc, age = [{:.1f}, {:.1f}] Gyr'.format(
          spec_name, radius_max, age_limits[0], age_limits[1]))
    for rotation_vector in part.principal_axes_vectors:
        print('{:.12f}, {:.12f}, {:.12f}'.format(
            rotation_vector[0], rotation_vector[1], rotation_vector[2]))
    print()


def get_lsr_positions_velocities(
    part, sun_positions=sun_positions, lsr_radius=0.2, age_limits=[],
    coordinate_system='cartesian'):
    '''
    Get positions and velocities (in galactocentric cartestian coordinates) of the LSR around
    input solar positions.

    Parameters
    ----------
    part : dictionary class : catalog of particles
    sun_positions : list : list of solar positions, in cylindrical coordinates [kpc]
    lsr_radius : float : radius around solar position over which to compute the LSR [kpc]
    age_limits : list : min and max stellar age to use in computing LSR [Gyr]
    coordinate_system : string : which coordinates to get positions in:
        'cartesian' (default), 'cylindrical', 'spherical'
    '''
    spec_name = 'star'  # particle species to use to compute LSR

    assert coordinate_system in ('cartesian', 'cylindrical', 'spherical')

    # store galaxy center
    center_position = part[spec_name].center_position
    center_velocity = part[spec_name].center_velocity
    principal_axes_vectors = part[spec_name].principal_axes_vectors

    lsr_positions = []
    lsr_velocities = []
    for pos_i, sun_position in enumerate(sun_positions):
        print('# {}: LSR {}'.format(
            part.info['simulation.name'].replace(' r7100', '_res7100'), pos_i))

        # convert from cylindrical to cartesian
        lsr_position = coordinate.get_positions_in_coordinate_system(
            sun_position, 'cylindrical', 'cartesian')

        # get total/scalar distances from galaxy center [kpc physical]
        distance_vectors = coordinate.get_distances(
            part[spec_name]['position'], center_position, part.info['box.length'],
            part.snapshot['scalefactor'])
        distance_vectors = coordinate.get_coordinates_rotated(
            distance_vectors, principal_axes_vectors)
        distances = coordinate.get_distances(
            distance_vectors, lsr_position, part.info['box.length'], total_distance=True)

        # keep only particles within lsr_radius of input position
        part_indices = np.where(distances < lsr_radius)[0]

        # get particles within age limits
        if age_limits is not None and len(age_limits):
            try:
                ages = part[spec_name]['age'][part_indices]
            except Exception:
                ages = part[spec_name].prop('age', part_indices)
            masks = ((ages >= min(age_limits)) * (ages < max(age_limits)))
            part_indices = part_indices[masks]

        # get velocity vectors aligned with principal axes, in cartesian coordinates [km / s]
        velocity_vectors = coordinate.get_velocity_differences(
            part[spec_name]['velocity'][part_indices], center_velocity,
            part[spec_name]['position'][part_indices], center_position,
            part.info['box.length'], part.snapshot['scalefactor'], part.snapshot['time.hubble'])
        velocity_vectors = coordinate.get_coordinates_rotated(
            velocity_vectors, principal_axes_vectors)

        # convert to cylindrical coordinates
        if coordinate_system in ('cylindrical', 'spherical'):
            lsr_position = sun_position  # convert back to cylindrical
            velocity_vectors = coordinate.get_velocities_in_coordinate_system(
                velocity_vectors, distance_vectors[part_indices], 'cartesian', coordinate_system)

        # get median velocity within lsr_radius
        lsr_velocity = np.median(velocity_vectors, 0)

        #lsr_velocity = np.zeros(velocity_vectors.shape[1], velocity_vectors.dtype)
        #for dimen_i in range(velocity_vectors.shape[1]):
        #    lsr_velocity[dimen_i] = percentile_weighted(
        #        velocity_vectors[:, dimen_i], 50, part[spec_name].prop('mass', part_indices))

        print('# coordinates of a local standard of rest (LSR)')
        print('# positions and velocities are:')
        print('#   relative to the galaxy center')
        print('#   aligned with the galaxys principal axes')
        if coordinate_system is 'cylindrical':
            print('#   in cylindrical units (R, Z, phi), (v_R, v_Z, v_phi)')
        else:
            print('#   in cartesian units (x, y, z), (v_x, v_y, v_z)')
        print('# velocity is median of {} {} particles within {:.0f} pc of position'.format(
            part_indices.size, spec_name, 1000 * lsr_radius))
        print('# {} particle nearest to position has array index = {}, dist = {:.3f} kpc'.format(
            spec_name, np.nanargmin(distances), min(distances)))
        print('# position [kpc] velocity [km/s]')
        print('{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}'.format(
            lsr_position[0], lsr_position[1], lsr_position[2],
            lsr_velocity[0], lsr_velocity[1], lsr_velocity[2]))
        print()

        lsr_positions.append(lsr_position)
        lsr_velocities.append(lsr_velocity)

    return lsr_positions, lsr_velocities
