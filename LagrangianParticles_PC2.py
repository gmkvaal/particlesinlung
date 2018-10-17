# __authors__ = ('Mikael Mortensen <mikaem@math.uio.no>',
#                'Miroslav Kuchta <mirok@math.uio.no>')
# __date__ = '2014-19-11'
# __copyright__ = 'Copyright (C) 2011' + __authors__
# __license__  = 'GNU Lesser GPL version 3 or any later version'
'''
This module contains functionality for Lagrangian tracking of particles with
DOLFIN
'''

import dolfin as df
import numpy as np
import copy
from mpi4py import MPI as pyMPI
from collections import defaultdict
import csv
import os

# Disable printing
__DEBUG__ = True

comm = pyMPI.COMM_WORLD

# collisions tests return this value or -1 if there is no collision
__UINT32_MAX__ = np.iinfo('uint32').max

class Particle:
    __slots__ = ['position', 'properties']
    'Lagrangian particle with position and some other passive properties.'
    def __init__(self, x):
        self.position = x
        self.properties = {}


    def send(self, dest):
        'Send particle to dest.'
        comm.Send(self.position, dest=dest)
        comm.send(self.properties, dest=dest)

    def recv(self, source):
        'Receive info of a new particle sent from source.'
        comm.Recv(self.position, source=source)
        self.properties = comm.recv(source=source)


class CellWithParticles(df.Cell):
    'Dolfin cell with list of particles that it contains.'
    def __init__(self, mesh, cell_id, particle):
        # Initialize parent -- create Cell with id on mesh
        df.Cell.__init__(self, mesh, cell_id)
        # Make an empty list of particles that I carry
        self.particles = []
        self += particle
        # Make the cell aware of its neighbors; neighbor cells are cells
        # connected to this one by vertices
        tdim = mesh.topology().dim()

        neighbors = sum((vertex.entities(tdim).tolist() for vertex in df.vertices(self)), [])
        neighbors = set(neighbors) - set([cell_id])   # Remove self
        self.neighbors = map(lambda neighbor_index: df.Cell(mesh, neighbor_index),
                             neighbors)

    def __add__(self, particle):
        'Add single particle to cell.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            self.particles.append(particle)
            return self
        else:
            return self.__add__(Particle(particle))

    def __len__(self):
        'Number of particles in cell.'
        return len(self.particles)


class CellParticleMap(dict):
    'Dictionary of cells with particles.'
    def __add__(self, ins):
        '''
        Add ins to map:
            ins is either (mesh, cell_id, particle) or
                          (mesh, cell_id, particle, particle_properties)
        '''
        assert isinstance(ins, tuple) and len(ins) in (3, 4)
        # If the cell_id is in map add the particle
        if ins[1] in self:
            self[ins[1]] += ins[2]
        # Other wise create new cell
        else:
            self[ins[1]] = CellWithParticles(ins[0], ins[1], ins[2])
        # With particle_properties, update properties of the last added particle
        if len(ins) == 4:
            self[ins[1]].particles[-1].properties.update(ins[3])

        return self

    def pop(self, cell_id, i):
        'Remove i-th particle from the list of particles in cell with cell_id.'
        # Note that we don't check for cell_id being a key or cell containg
        # at least i particles.
        particle = self[cell_id].particles.pop(i)

        # If the cell is empty remove it from map
        if len(self[cell_id]) == 0:
            del self[cell_id]

        return particle

    def total_number_of_particles(self):
        'Total number of particles in all cells of the map.'
        return sum(map(len, self.itervalues()))


class LagrangianParticles:
    'Particles moved by the velocity field in V.'
    def __init__(self, V, save_particles_path, stuck_particles_path, deposition_criterion, g, rho_p, d_p,
                    rho_f, nu_f):

        self.save_particles_path = save_particles_path
        self.stuck_particles_path = stuck_particles_path

        self.g = g
        self.rho_p = rho_p
        self.d_p = d_p
        self.rho_f = rho_f
        self.nu_f = nu_f
        self.mu_f = self.nu_f*self.rho_f
        self.deposition_criterion = deposition_criterion
        self.tau =  (self.rho_p*self.d_p**2) / (18*self.mu_f)
        self.K1 = 18*self.mu_f / (self.rho_p*self.d_p**2)
        self.K2 = 3*self.mu_f/(4*self.rho_p*self.d_p**2)
        self.__debug = __DEBUG__

        self.V = V
        self.mesh = V.mesh()
        self.bmesh = df.BoundaryMesh(self.mesh, "exterior")
        self.dofmap = self.V.dofmap()
        #self.mesh.init(2, 2)  # Cell-cell connectivity for neighbors of cell Removed
        self.tree = self.mesh.bounding_box_tree()  # Tree for isection comput.
        self.boundary_tree = self.bmesh.bounding_box_tree()
        # Allocate some variables used to look up the velocity
        # Velocity is computed as U_i*basis_i where i is the dimension of
        # element function space, U are coefficients and basis_i are element
        # function space basis functions. For interpolation in cell it is
        # advantageous to compute the resctriction once for cell and only
        # update basis_i(x) depending on x, i.e. particle where we make
        # interpolation. This updaea mounts to computing the basis matrix
        self.dim = self.mesh.topology().dim()
        self.u_new = np.zeros(self.dim)
        self.u_old = np.zeros(self.dim)
        self.x_new = np.zeros(self.dim)
        self.mesh.init(0, self.dim)

        self.element = V.dolfin_element()
        self.num_tensor_entries = 1
        for i in range(self.element.value_rank()):
            self.num_tensor_entries *= self.element.value_dimension(i)
        # For VectorFunctionSpace CG1 this is 3
        self.coefficients = np.zeros(self.element.space_dimension())

        # For VectorFunctionSpace CG1 this is 3x3compute
        self.basis_matrix = np.zeros((self.element.space_dimension(),
                                      self.num_tensor_entries))

        # Allocate a dictionary to hold all particles
        self.particle_map = CellParticleMap()

        # Allocate some MPI stuff
        self.num_processes = comm.Get_size()
        self.myrank = comm.Get_rank()
        self.all_processes = range(self.num_processes)
        self.other_processes = range(self.num_processes)
        self.other_processes.remove(self.myrank)
        self.my_escaped_particles = np.zeros(1, dtype='I')
        self.tot_escaped_particles = np.zeros(self.num_processes, dtype='I')
        # Dummy particle for receiving/sending at [0, 0, ...]
        self.particle0 = Particle(np.zeros(self.mesh.geometry().dim()))


    def __iter__(self):
        '''Iterate over all particles.'''
        for cwp in self.particle_map.itervalues():
            for particle in cwp.particles:
                yield particle


    def add_particles(self, list_of_particles, properties_d):
        '''Add particles and search for their home on all processors.
           Note that list_of_particles must be same on all processes. Further
           every len(properties[property]) must equal len(list_of_particles).
        '''
        if properties_d is not None:
            n = len(list_of_particles)
            assert all(len(sub_list) == n
                       for sub_list in properties_d.itervalues())
            # Dictionary that will be used to feed properties of single
            # particles
            properties = properties_d.keys()
            particle_properties = dict((key, 0) for key in properties)


            has_properties = True
        else:
            has_properties = False

        pmap = self.particle_map
        my_found = np.zeros(len(list_of_particles), 'I')
        all_found = np.zeros(len(list_of_particles), 'I')

        for i, particle in enumerate(list_of_particles):

            c = self.locate(particle)
            if not (c == -1 or c == __UINT32_MAX__):
                my_found[i] = True
                if not has_properties:
                    pmap += self.mesh, c, particle
                else:
                    # Get values of properties for this particle
                    for key in properties:
                        particle_properties[key] = properties_d[key][i]
                    pmap += self.mesh, c, particle, particle_properties
        # All particles must be found on some process
        comm.Reduce(my_found, all_found, root=0)

        if self.myrank == 0:

            missing = np.where(all_found == 0)[0]
            n_missing = len(missing)

            if self.__debug:
                for i in missing:

                    #print 'Missing', list_of_particles[i].position
                    self.save_stuck_particle(list_of_particles[i].position)

                n_duplicit = len(np.where(all_found > 1)[0])
                #print 'There are %d duplicit particles' % n_duplicit



    def drag_model_1(self, u_p, u_f, u_f_1, forces, dt):
        u_new = np.zeros(len(u_p))
        for i in range(len(u_p[:])):
            Re_p = self.d_p*abs(u_f_1[i] - u_p[i])/self.nu_f

            if Re_p <= 0.1:
                u_new[i] = (u_p[i] + dt*forces[i] + dt*self.K1*(u_f[i] - 0.5*u_p[i])) / (1 + 0.5*dt*self.K1)

            elif Re_p > 0.1 and Re_p < 1.0:
                Cd = 22.73/Re_p + 0.0903/Re_p**2 + 3.69
                u_new[i] = (u_p[i] + dt*forces[i] + dt*self.K2*Cd*Re_p*(u_f[i] - 0.5*u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 1 and Re_p < 10:
                Cd = 29.1667/Re_p - 3.8889/Re_p**2 + 1.222
                u_new[i] = (u_p[i] + dt*forces[i] + dt*self.K2*Cd*Re_p*(u_f[i] - 0.5*u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 10 and Re_p < 100:
                Cd = 46.5/Re_p - 116.67/Re_p**2 + 0.6167
                u_new[i] = (u_p[i] + dt*forces[i] + dt*self.K2*Cd*Re_p*(u_f[i] - 0.5*u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 100 and Re_p < 1000:
                Cd = 98.33/Re_p - 2778/Re_p**2 + 0.3644
                u_new[i] = (u_p[i] + dt*forces[i] + dt*self.K2*Cd*Re_p*(u_f[i] - 0.5*u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

        return u_new[:]


    def drag_model_2(self, u_p, u_f, u_f_1, forces, dt):
        u_new = np.zeros(len(u_p))

        for i in range(len(u_p[:])):
            Re_p = self.d_p*abs(u_f[i] - u_p[i])/self.nu_f

            if Re_p <= 0.1:
                u_new[i] = ( u_p[i] + dt*forces[i] + 0.5*dt*self.K1*(u_f[i] + u_f_1[i] - u_p[i]) ) / (1 + 0.5*dt*self.K1)

            elif Re_p > 0.1 and Re_p < 1.0:
                Cd = 22.73/Re_p + 0.0903/Re_p**2 + 3.69
                u_new[i] = (u_p[i] + dt*forces[i] + 0.5*dt*self.K2*Cd*Re_p*(u_f[i] + u_f_1[i] - u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 1 and Re_p < 10:
                Cd = 29.1667/Re_p - 3.8889/Re_p**2 + 1.222
                u_new[i] = (u_p[i] + dt*forces[i] + 0.5*dt*self.K2*Cd*Re_p*(u_f[i] + u_f_1[i] - u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 10 and Re_p < 100:
                Cd = 46.5/Re_p - 116.67/Re_p**2 + 0.6167
                u_new[i] = (u_p[i] + dt*forces[i] + 0.5*dt*self.K2*Cd*Re_p*(u_f[i] + u_f_1[i] - u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

            elif Re_p > 100 and Re_p < 1000:
                Cd = 98.33/Re_p - 2778/Re_p**2 + 0.3644
                u_new[i] = (u_p[i] + dt*forces[i] + 0.5*dt*self.K2*Cd*Re_p*(u_f[i] + u_f_1[i] - u_p[i])) / (1 + 0.5*dt*self.K2*Cd*Re_p)

        return u_new[:]



    def step(self, u, dt):
        'Move particles by forward Euler x += u*dt'
        start = df.Timer('shift')

        #print self.particle_map.total_number_of_particles()

        for cwp in self.particle_map.itervalues():

            # Restrict once per cell

            u.restrict(self.coefficients,
                       self.element,
                       cwp,
                       cwp.get_vertex_coordinates(),
                       cwp)


            for particle in cwp.particles:


                x = particle.position
                self.element.evaluate_basis_all(self.basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())


                u_f = np.dot(self.coefficients, self.basis_matrix)[:]
                u_p = particle.properties['u_p']
                particle.properties['u_1'] = u_f
                particle.properties['x1'] = 1*x[:]
                particle.properties['up1'] = 1*u_p

                forces = self.g

                u_new = self.drag_model_1(u_p, u_f, u_f, forces, dt)
                x[:] = x[:] + dt*0.5*(u_new + u_p[:])


                self.element.evaluate_basis_all(self.basis_matrix,
                                                x,
                                                cwp.get_vertex_coordinates(),
                                                cwp.orientation())
                # Fluid velocity
                u_f = np.dot(self.coefficients, self.basis_matrix)[:]
                u_p = particle.properties['u_p']
                u_f_1 = particle.properties['u_1']
                x1 = particle.properties['x1']

                forces = self.g
                u_new = self.drag_model_2(u_p, u_f, u_f_1, forces, dt)
                x[:] =  x1[:] + dt*0.5*(u_new[:] + u_p[:])
                particle.properties['u_p'] = u_new[:]

                #cell_index, distance = self.boundary_tree.compute_closest_entity(df.Point(x) )
                #particle.properties['dist'] = distance

                #if u_p[0] > 0.2 + 1e-8:

                #    return 1

        # Recompute the map
        stop_shift = start.stop()
        start =df.Timer('relocate')
        info = self.relocate()
        stop_reloc = start.stop()
        # We return computation time per process
        #return (stop_shift, stop_reloc)
        #return u_new[:], x[:], Re_p
        #return 0

    def relocate(self):
        # Relocate particles on cells and processors
        p_map = self.particle_map

        # Map such that map[old_cell] = [(new_cell, particle_id), ...]
        # Ie new destination of particles formerly in old_cell
        props = []
        depo_in_cell = []
        depo_in_cell_index = []
        new_cell_map = defaultdict(list)

        for cwp in p_map.itervalues():
            for i, particle in enumerate(cwp.particles):
                point = df.Point(*particle.position)
                if not cwp.contains(point):
                    found = False
                    # Check neighbor cells
                    #for neighbor in df.cells(cwp):
                    for neighbor in cwp.neighbors:
                        if neighbor.contains(point):
                            new_cell_id = neighbor.index()
                            found = True
                            #print "found in neighbor"
                            break
                    # Do a completely new search if not found by now
                    if not found:
                        new_cell_id = self.locate(particle)

                    # Record to map
                    new_cell_map[cwp.index()].append((new_cell_id, i))
		"""
                #Appending particles that are within the deposition criteria to destruction list
                if particle.properties['dist'] <= self.deposition_criterion:
                    depo_in_cell.append((cwp.index(), i, particle.properties['x1']))
                    depo_in_cell_index.append(cwp.index())

        if len(depo_in_cell) >= 1:
            ids = []; cells = []; positions = []
            # Reading backwards in case more that one particle in one cell
            for index, cell_id, pos in sorted(depo_in_cell,
                                              key=lambda t: t[1],
                                              reverse=True):

                ids.append(index); cells.append(cell_id); positions.append(pos)
                self.save_stuck_particle(pos)
                p_map.pop(index, cell_id)
	"""
        # Rebuild locally the particles that end up on the process. Some
        # have cell_id == -1, i.e. they are on other process
        list_of_escaped_particles = []
        for old_cell_id, new_data in new_cell_map.iteritems():

            # We iterate in reverse because normal order would remove some
            # particle this shifts the whole list!
            for (new_cell_id, i) in sorted(new_data,
                                           key=lambda t: t[1],
                                           reverse=True):
                if not old_cell_id in depo_in_cell_index:
                    particle = p_map.pop(old_cell_id, i)

                # Not adding particles that have a certain distance to the wall
                # back to the particle map
                #if particle.properties['dist'] <= self.deposition_criterion:
                #    self.save_stuck_particle(particle.properties["x1"])
                if new_cell_id == -1 or new_cell_id == __UINT32_MAX__ :
                    list_of_escaped_particles.append(particle)
                else:
                    p_map += self.mesh, new_cell_id, particle
        # Create a list of how many particles escapes from each processor
        self.my_escaped_particles[0] = len(list_of_escaped_particles)
        # Make all processes aware of the number of escapees
        comm.Allgather(self.my_escaped_particles, self.tot_escaped_particles)

        # Send particles to root
        if self.myrank != 0:
            for particle in list_of_escaped_particles:
                particle.send(0)

        # Receive the particles escaping from other processors
        if self.myrank == 0:
            for proc in self.other_processes:
                for i in range(self.tot_escaped_particles[proc]):
                    self.particle0.recv(proc)
                    list_of_escaped_particles.append(copy.deepcopy(self.particle0))

        # Put all travelling particles on all processes, then perform new search
        travelling_particles = comm.bcast(list_of_escaped_particles, root=0)
        self.add_particles(travelling_particles, properties_d=None)



    def total_number_of_particles(self):
        'Return number of particles in total and on process.'
        num_p = self.particle_map.total_number_of_particles()
        tot_p = comm.allreduce(num_p)
        return (tot_p, num_p)

    def locate(self, particle):
        'Find mesh cell that contains particle.'
        assert isinstance(particle, (Particle, np.ndarray))
        if isinstance(particle, Particle):
            # Convert particle to point
            point = df.Point(*particle.position)
            return self.tree.compute_first_entity_collision(point)
        else:
            return self.locate(Particle(particle))


    def save_position(self, particles, timestep):
        # adding timestamp to path
        filename = os.path.split(self.save_particles_path)[1]
        extention = filename.split(".")[-1]
        filename = filename.split(".")[0]
        filename = "%s%s.%s" % (filename,timestep,extention)
        filename = os.path.join(os.path.split(self.save_particles_path)[0], filename)
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(particles)


    def save_velo(self, velo, timestep):
        # adding timestamp to path
        filepath = os.path.split(self.save_particles_path)[0]
        filename = os.path.join(filepath, "velo%s" % timestep)
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(velo)


    def save_stuck_particle(self, stuck_particles):
        with open(self.stuck_particles_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(stuck_particles)



    def scatter(self, timestep, cp_velo):
        "Gather all particles on process 1 and save positions"

        p_map = self.particle_map

        # Inform root about number of particles on other processes
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves send particles to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)

        # Master recieves particles
        else:
            received_particles = []
            received_particles_velo = []
            [received_particles.append(p.position)
                           for cwp in p_map.itervalues()
                           for p in cwp.particles]

            [received_particles_velo.append(p.properties['u_p'])
                           for cwp in p_map.itervalues()
                           for p in cwp.particles]

            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    received_particles.append(copy.copy(self.particle0.position))
                    received_particles_velo.append(copy.copy(self.particle0.properties['u_p']))

            self.save_position(received_particles, timestep)
            if timestep % cp_velo == 0:
                self.save_velo(received_particles_velo, timestep)


    def scatter_plot(self, timestep, skip=1):
        'Scatter plot of all particles on process 0'


        p_map = self.particle_map

        #import matplotlib.colors as colors
        #import matplotlib.cm as cmx
        #cmap = cmx.get_cmap('jet')
        #cnorm = colors.Normalize(vmin=0, vmax=self.num_processes)
        #scalarMap = cmx.ScalarMappable(norm=cnorm, cmap=cmap)


        # ax = fig.gca()

        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()

        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        # Slaves should send to master
        if self.myrank > 0:
            for cwp in p_map.itervalues():
                for p in cwp.particles:
                    p.send(0)
        else:
            # Receive on master
            received = defaultdict(list)
            received[0] = [copy.copy(p.position)
                           for cwp in p_map.itervalues()
                           for p in cwp.particles]
            for proc in self.other_processes:
                # Receive all_particles[proc]
                for j in range(all_particles[proc]):
                    self.particle0.recv(proc)
                    received[proc].append(copy.copy(self.particle0.position))


            for proc in received:
                # Plot only if there is something to plot
                particles = received[proc]

                if len(particles) > 0:
                    xy = np.array(particles)
                    self.save_position(xy, timestep)
                """
                if len(particles) > 0:
                    xy = np.array(particles)
                    #print xy
                    if len(xy) > 1:
                        n = len(xy[1])
                        m = len(xy)
                        coords = np.zeros([m,n+1])
                        coords[:,:-1] = xy
                        coords[:,-1] = 1
                        #coords = coords.round(3)
                        #print coords
                        self.save_position(coords, timestep)

                    ax.scatter(xy[::skip, 0], xy[::skip, 1],
                               label='%d' % proc,
                               c=scalarMap.to_rgba(proc),
                               edgecolor='none')
            ax.legend(loc='best')
            ax.axis([0, 1, 0, 1])"""


    def bar(self, fig):
        'Bar plot of particle distribution.'
        ax = fig.gca()

        p_map = self.particle_map
        all_particles = np.zeros(self.num_processes, dtype='I')
        my_particles = p_map.total_number_of_particles()
        # Root learns about count of particles on all processes
        comm.Gather(np.array([my_particles], 'I'), all_particles, root=0)

        if self.myrank == 0 and self.num_processes > 1:
            ax.bar(np.array(self.all_processes)-0.25, all_particles, 0.5)
            ax.set_xlabel('proc')
            ax.set_ylabel('number of particles')
            ax.set_xlim(-0.25, max(self.all_processes)+0.25)
            return np.sum(all_particles)
        else:
            return None


# Simple initializers for particle positions

from math import pi, sqrt
from itertools import product

comm = pyMPI.COMM_WORLD


class RandomGenerator(object):
    '''
    Fill object by random points.
    '''
    def __init__(self, domain, rule):
        '''
        Domain specifies bounding box for the shape and is used to generate
        points. The rule filter points of inside the bounding box that are
        axctually inside the shape.
        '''
        assert isinstance(domain, list)
        self.domain = domain
        self.rule = rule
        self.dim = len(domain)
        self.rank = comm.Get_rank()

    def generate(self, N, method='full'):
        'Genererate points.'
        assert len(N) == self.dim
        assert method in ['full', 'tensor']

        if self.rank == 0:
            # Generate random points for all coordinates
            if method == 'full':
                n_points = np.product(N)
                points = np.random.rand(n_points, self.dim)
                for i, (a, b) in enumerate(self.domain):
                    points[:, i] = a + points[:, i]*(b-a)
            # Create points by tensor product of intervals
            else:
                # Values from [0, 1) used to create points between
                # a, b - boundary
                # points in each of the directiosn
                shifts_i = np.array([np.random.rand(n) for n in N])
                # Create candidates for each directions
                points_i = (a+shifts_i[i]*(b-a)
                            for i, (a, b) in enumerate(self.domain))
                # Cartesian product of directions yield n-d points
                points = (np.array(point) for point in product(*points_i))


            # Use rule to see which points are inside
            points_inside = np.array(filter(self.rule, points))
        else:
            points_inside = None

        points_inside = comm.bcast(points_inside, root=0)

        return points_inside


class RandomRectangle(RandomGenerator):
    def __init__(self, ll, ur):
        ax, ay = ll.x(), ll.y()
        bx, by = ur.x(), ur.y()
        assert ax < bx and ay < by
        RandomGenerator.__init__(self, [[ax, bx], [ay, by]], lambda x: True)


class RandomCircle(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2) < radius)

class RandomRing(RandomGenerator):
    def __init__(self, center, radius1, radius2):
        assert radius1 > 0
        domain = [[center[0]-radius2, center[0]+radius2],
                  [center[1]-radius2, center[1]+radius2]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: radius1 < sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2) < radius2)

class RandomSphere(RandomGenerator):
    def __init__(self, center, radius):
        assert radius > 0
        domain = [[center[0]-radius, center[0]+radius],
                  [center[1]-radius, center[1]+radius],
                  [center[2]-radius, center[2]+radius]]
        RandomGenerator.__init__(self, domain,
                                 lambda x: sqrt((x[0]-center[0])**2 +
                                                (x[1]-center[1])**2 +
                                                (x[2]-center[2])**2) < radius
				)


class RandomSphere3D(object):
    def __init__(self, center, R, N_points):
        self.center = center
        self.R = R
        self.N_points = N_points

    def generate(self):
        points = np.zeros(shape=(self.N_points,3))
        for i in range(len(points)):
            r = self.R*np.random.random_sample()
            theta = 2*np.pi*np.random.random_sample()
            phi = np.pi*np.random.random_sample()
            points[i,0] = r*np.cos(theta)*np.sin(theta) + self.center[0]
            points[i,1] = r*np.sin(theta)*np.sin(theta) + self.center[1]
            points[i,2] = r*np.cos(phi) + self.center[2]
        return points


class RandomCircle3D(object):
    def __init__(self, center, R, N_points):
        self.center = center
        self.R = R
        self.N_points = N_points

    def generate_y_plane(self):
        points = np.zeros(shape=(self.N_points,3))
        r = self.R * np.sqrt(np.random.uniform(0.0, 1.0, self.N_points))
        theta = np.random.uniform(0.0, 2.0*np.pi, self.N_points)
        points[:,0] = r*np.cos(theta) + self.center[0]
        points[:,1] = self.center[1]
        points[:,2] = r*np.sin(theta) + self.center[2]
        return points

class RandomCircle2D(object):
    def __init__(self, center, R, N_points):
        self.center = center
        self.R = R
        self.N_points = N_points

    def generate(self):
        points = np.zeros(shape=(self.N_points,2))
        for i in range(len(points)):
            r = self.R*np.random.random_sample()
            theta = 2*np.pi*np.random.random_sample()
            points[i,0] = r*np.cos(theta) + self.center[0]
            points[i,1] = r*np.sin(theta) + self.center[1]
        return points
