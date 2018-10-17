from ..NSfracStep import *
import numpy as np
from hashlib import sha1
#from h5py import *
#from pyvtk import *
#from fenicstools import Probes
from Probe import *
from LagrangianParticles_PC2 import *

from os import path, getcwd, listdir, remove, system, makedirs
import cPickle

set_log_active(False)

def mesh(meshfile, ** NS_namespace):
    f_ = HDF5File(mpi_comm_world(), meshfile, "r")
    m = Mesh()
    f_.read(m, "/mesh", False)

    return m


### If restarting from previous solution then read in parameters ###
def update(commandline_kwargs, NS_parameters, **NS_namespace):
    if commandline_kwargs.has_key("restart_folder"):
        restart_folder = commandline_kwargs["restart_folder"]
        f = open(path.join(restart_folder, 'params.dat'), 'r')
        NS_parameters.update(cPickle.load(f))
        NS_parameters['restart_folder'] = restart_folder
	NS_parameters['t'] = 0
        NS_parameters['tstep'] = 0
        NS_parameters['T'] =  1
        NS_parameters['start_particles'] = 1
        NS_parameters['stop_particles'] = 5000
        NS_parameters['plot_particles'] = 100
        NS_parameters['plot_t'] = 1e10
	#NS_parameters['folder'] = commandline_kwargs["folder"]
        globals().update(NS_parameters)
        if MPI.rank(mpi_comm_world()) == 0:
            print "restarting from", restart_folder
            print NS_parameters

    else:
        NS_parameters.update(
            use_krylov_solvers = True,
            center = [],
            inlet_normal = [],
            nu = 1.7e-5,
            T = 10,
            dt = 0.01,
            plot_interval = 1e10,
            plot_t = 100,
            save_step = 10,
            checkpoint=1000,
            velocity_degree=1,
            print_intermediate_info = 250,
            name = "lungtest",
            counter = 0,
            wall_ID = [16,37],
            outlet_ID = 38,
            inlet_ID = 1,
            check_flux = 10,
            check_CFL = 10,
            folder = "results",
            U = 1.86,
            dump_stats = 1000,
            meshfile = "dist_03_1_1d2_noBL_meter.xml",
            start_particles = 1,
            stop_particles = 5000,
            rsp_path = "lol",
            rsp_step = 1000,
            restart_particles = [False],
	        plot_particles = 100
            )


class _HDF5Link:
    """Helper class for creating links in HDF5-files."""
    cpp_link_module = None
    def __init__(self):
        cpp_link_code = '''
        #include <hdf5.h>
        void link_dataset(const MPI_Comm comm,
                          const std::string hdf5_filename,
                          const std::string link_from,
                          const std::string link_to, bool use_mpiio)
        {
            hid_t hdf5_file_id = HDF5Interface::open_file(comm, hdf5_filename, "a", use_mpiio);
            herr_t status = H5Lcreate_hard(hdf5_file_id, link_from.c_str(), H5L_SAME_LOC,
                                link_to.c_str(), H5P_DEFAULT, H5P_DEFAULT);
            dolfin_assert(status != HDF5_FAIL);

            HDF5Interface::close_file(hdf5_file_id);
        }
        '''

        self.cpp_link_module = compile_extension_module(cpp_link_code, additional_system_headers=["dolfin/io/HDF5Interface.h"])

    def link(self, hdf5filename, link_from, link_to):
        "Create link in hdf5file."
        use_mpiio = MPI.size(mpi_comm_world()) > 1
        self.cpp_link_module.link_dataset(mpi_comm_world(), hdf5filename, link_from, link_to, use_mpiio)


def save_hdf5(fullname, field_name, data, timestep, hdf5_link):
        # Create "good enough" hash. This is done to avoid data corruption when restarted from
        # different number of processes, different distribution or different function space
        local_hash = sha1()
        local_hash.update(str(data.function_space().mesh().num_cells()))
        local_hash.update(str(data.function_space().ufl_element()))
        local_hash.update(str(data.function_space().dim()))
        local_hash.update(str(MPI.size(mpi_comm_world())))

        # Global hash (same on all processes), 10 digits long
        global_hash = MPI.sum(mpi_comm_world(), int(local_hash.hexdigest(), 16))
        global_hash = str(int(global_hash%1e10)).zfill(10)

        # Open HDF5File
        if not path.isfile(fullname):
            datafile = HDF5File(mpi_comm_world(), fullname, 'w')
        else:
            datafile = HDF5File(mpi_comm_world(), fullname, 'a')

        # Write to hash-dataset if not yet done
        if not datafile.has_dataset(global_hash) or not datafile.has_dataset(global_hash+"/"+field_name):
            datafile.write(data, str(global_hash)+"/"+field_name)

        if not datafile.has_dataset("Mesh"):
            datafile.write(data.function_space().mesh(), "Mesh")

        # Write vector to file
        datafile.write(data.vector(), field_name+str(timestep)+"/vector")

        # HDF5File.close is broken in 1.4, but fixed in dev.
        if dolfin_version() != "1.4.0":
            datafile.close()
        del datafile

        # Link information about function space from hash-dataset
        hdf5filename = str(global_hash)+"/"+field_name+"/%s"
        field_name_current = "%s%s" % (field_name, str(timestep)) +"/%s"
        for l in ["x_cell_dofs", "cell_dofs", "cells"]:
            hdf5_link(fullname, hdf5filename % l, field_name_current % l)



def create_bcs(V, Q, mesh, meshfile, inlet_ID, outlet_ID, wall_ID, ** NS_namespace):
    D = 3   # Dimentions
    fd = MeshFunction("size_t", mesh, D-1, mesh.domains())

    p0 = project(Expression("x[0]", degree=1), V, solver_type="cg", preconditioner_type="none")
    p1 = project(Expression("x[1]", degree=1), V, solver_type="cg", preconditioner_type="none")
    p2 = project(Expression("x[2]", degree=1), V, solver_type="cg", preconditioner_type="none")

    def boundary_data(inlet_ID):
        # Compute inlet area
        inlet_normal = []
        center = []

        dsi = ds(inlet_ID, domain=mesh, subdomain_data=fd)
        inlet_area = assemble(Constant(1)*dsi)

        # Compute center of inlet coordinate
        for p in [p0, p1, p2]:
            center.append(assemble(p*dsi) / inlet_area)

        # Compute inlet normal vector
        n = FacetNormal(mesh)
        ni = np.array([assemble(n[i]*dsi) for i in xrange(D)])
        n_len = np.sqrt(sum([ni[i]**2 for i in xrange(D)])) # Should always be 1!?
        i_n = -ni / n_len
        for i in range(len(i_n)):
            inlet_normal.append(i_n[i])


        if MPI.rank(mpi_comm_world()) == 0:
            print "-----Inlet / Outlet info-----"
            print "ID = %d" % inlet_ID
            print "Area = %.3e" % inlet_area
            print "Center = (%.3e,%.3e,%.3e)" % (center[0],center[1],center[2])
            print "Normal = (%.3e,%.3e,%.3e)" % (inlet_normal[0],inlet_normal[1],inlet_normal[2])

        return center, inlet_normal, inlet_area


    class IP(Expression):
        # Generate a parabolic inlet profile
        def eval(self, values, x):

            r = np.sqrt((center[0]-x[0])**2+(center[1]-x[1])**2+(center[2]-x[2])**2)
            inlet_prof = 2*U*(1 - r**2/R**2)

            values[:] = inlet_prof

        def shape_value(self):
            return (1,)

    noslip = Constant(0)
    bcx = []; bcy = []; bcz = []; bcp = []


    if inlet_ID == 29:
        outlet_flux = {30:2.8, 31:3.2, 32:8.0, 33:3.8, 34:7.1, 35:9.3, 36:7.6, 37:6.2, 38:6.1, 39:5.9}
        inlet_flux = {29:60}


    if inlet_ID == 28:
        outlet_flux = {29:2.8, 30:3.2, 31:8.0, 32:3.8, 33:7.1, 34:9.3, 35:7.6, 36:6.2, 37:6.1, 38:5.9}
        inlet_flux = {28:60}


    total_flux = 0

    # Inlet condition
    center, inlet_normal, inlet_area = boundary_data(inlet_ID)
    U = (10**(-3)/60.0)*inlet_flux[inlet_ID] / inlet_area
    if MPI.rank(mpi_comm_world()) == 0:
        print "U = %.3e" % U
    R_square = inlet_area/np.pi
    parabolic = Expression("2*%s*(1 - (pow((%s-x[0]),2)  + pow((%s-x[1]),2) + pow((%s-x[2]),2) ) / %s)" % \
                                (U,center[0],center[1],center[2],R_square), degree=1 )

    ip = parabolic
    bcx.append(DirichletBC(V, ip*inlet_normal[0], fd, inlet_ID))
    bcy.append(DirichletBC(V, ip*inlet_normal[1], fd, inlet_ID))
    bcz.append(DirichletBC(V, ip*inlet_normal[2], fd, inlet_ID))

    # Zero pressure at one outlet
    bcp.append(DirichletBC(Q,noslip,fd,outlet_ID[-1]))

    # No slip on walls
    for key in wall_ID:
        bcx.append(DirichletBC(V, noslip, fd, key))
        bcy.append(DirichletBC(V, noslip, fd, key))
        bcz.append(DirichletBC(V, noslip, fd, key))

    # Assigning outlet conditions and converting to m3 / sec
    for key in outlet_ID[:-1]:
        center, inlet_normal, inlet_area = boundary_data(key)
        U = (10**(-3)/60.0)*outlet_flux[key] / inlet_area
        if MPI.rank(mpi_comm_world()) == 0:
            print "U = %.3e" % U
        R_square = inlet_area/pi
        ip = Expression("2*%s*(1 - (pow((%s-x[0]),2)  + pow((%s-x[1]),2) + pow((%s-x[2]),2) ) / %s)" % \
                            (U,center[0],center[1],center[2],R_square), degree=1 )

        bcx.append(DirichletBC(V, -ip*inlet_normal[0], fd, key))
        bcy.append(DirichletBC(V, -ip*inlet_normal[1], fd, key))
        bcz.append(DirichletBC(V, -ip*inlet_normal[2], fd, key))

        total_flux += outlet_flux[key]

    if MPI.rank(mpi_comm_world()) == 0:
        print ""
        print "----- TOTAL FLUX = %.3f" % (total_flux + outlet_flux[outlet_ID[-1]])
        print ""

    return dict(u0=bcx,
                u1=bcy,
                u2=bcz,
                p=bcp)


def pre_solve_hook(V, Q, velocity_degree, mesh, newfolder, **NS_namesepace):
    n = FacetNormal(mesh)
    normal = FacetNormal(mesh)

    eval_dict = {}
    probe_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "points_lung"))
    A1A2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "A1A2.npy"))
    B1B2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "B1B2.npy"))
    C1C2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "C1C2.npy"))
    D1D2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "D1D2.npy"))
    E1E2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "E1E2.npy"))
    F1F2_points = np.load(path.join(path.dirname(path.abspath(__file__)), "..", "..", "TracheaLinesPoints", "F1F2.npy"))

    eval_dict["centerline_u_x_probes"] = Probes(probe_points.flatten(), V)
    eval_dict["centerline_u_y_probes"] = Probes(probe_points.flatten(), V)
    eval_dict["centerline_u_z_probes"] = Probes(probe_points.flatten(), V)
    eval_dict["centerline_p_probes"] = Probes(probe_points.flatten(), Q)

    eval_dict["A1A2_u_x_probes"] = Probes(A1A2_points.flatten(), V)
    eval_dict["A1A2_u_y_probes"] = Probes(A1A2_points.flatten(), V)
    eval_dict["A1A2_u_z_probes"] = Probes(A1A2_points.flatten(), V)

    eval_dict["B1B2_u_x_probes"] = Probes(B1B2_points.flatten(), V)
    eval_dict["B1B2_u_y_probes"] = Probes(B1B2_points.flatten(), V)
    eval_dict["B1B2_u_z_probes"] = Probes(B1B2_points.flatten(), V)

    eval_dict["C1C2_u_x_probes"] = Probes(C1C2_points.flatten(), V)
    eval_dict["C1C2_u_y_probes"] = Probes(C1C2_points.flatten(), V)
    eval_dict["C1C2_u_z_probes"] = Probes(C1C2_points.flatten(), V)

    eval_dict["D1D2_u_x_probes"] = Probes(D1D2_points.flatten(), V)
    eval_dict["D1D2_u_y_probes"] = Probes(D1D2_points.flatten(), V)
    eval_dict["D1D2_u_z_probes"] = Probes(D1D2_points.flatten(), V)

    eval_dict["E1E2_u_x_probes"] = Probes(E1E2_points.flatten(), V)
    eval_dict["E1E2_u_y_probes"] = Probes(E1E2_points.flatten(), V)
    eval_dict["E1E2_u_z_probes"] = Probes(E1E2_points.flatten(), V)

    eval_dict["F1F2_u_x_probes"] = Probes(F1F2_points.flatten(), V)
    eval_dict["F1F2_u_y_probes"] = Probes(F1F2_points.flatten(), V)
    eval_dict["F1F2_u_z_probes"] = Probes(F1F2_points.flatten(), V)


    # Save file info
    hdf5_link = _HDF5Link().link

    dp_list = [1.0, 2.5, 4.0, 10.0, 20.0]

    if MPI.rank(mpi_comm_world()) == 0:
        counter = 1
        to_check = path.join(newfolder, "%s")
        while path.isdir(to_check % str(counter)):
            counter += 1

        if not path.exists(path.join(to_check % str(counter), "VTK")):
            makedirs(path.join(to_check % str(counter), "VTK"))
	    makedirs(path.join(to_check % str(counter), "Probes"))
            for d_p in dp_list:
                makedirs(path.join(to_check % str(counter), "Particles_dp%s" % d_p))
    else:
        counter = 0

    counter = MPI.max(mpi_comm_world(), counter)
    p_path = path.join(newfolder, str(counter))

    rho_p = 914
    rho_f = 1.1455
    nu_f = 1.7e-5
    mu_f = nu_f*rho_f
    lam = 0.070e-6
    g = np.array([0, 0, -9.81])
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    uv = Function(Vv)

    save_p_path_dict = {}
    save_s_path_dict = {}
    lp_dict = {}
    for d_p in dp_list:
        save_particles_path = path.join(p_path, "Particles_dp%s" % d_p, "particlos.particles")
        stuck_particles_path = path.join(p_path, "Particles_dp%s" % d_p, "stuck_dp%s" % d_p)
        dp = d_p * 1e-6
        deposition_criterion = dp/2
        if MPI.rank(mpi_comm_world()) == 0:
             print "creating instance for particle with diameter", dp
        lp_dict["%s" % d_p] = LagrangianParticles(Vv, save_particles_path, stuck_particles_path, deposition_criterion, g, rho_p, dp,
    	                        rho_f, nu_f)

    if MPI.rank(mpi_comm_world()) == 0:
             print "number of particle instances", len(lp_dict)




    return dict(hdf5_link=hdf5_link, eval_dict=eval_dict, lp_dict=lp_dict, uv=uv, p_path=p_path)

def read_particles(ifile):
    particle_positions = []
    with open(ifile, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            for row in reader:
                floats = []
                [floats.append(float(i)) for i in row]
                particle_positions.append(np.array(floats))
    return particle_positions


def temporal_hook(mesh, u_, p_, uv, dt, tstep, plot_t, hdf5_link, eval_dict, dump_stats, newfolder, lp_dict, p_path, rsp_path, rsp_step, **NS_namespace):

        start_particles = 1
        stop_particles = 5000
        plot_particles = 100


    if restart_particles[0]:
        if MPI.rank(mpi_comm_world()) == 0:
            print "restarting particles from step", rsp_step, "and path", rsp_path
        for dp in dp_list:
            ifile = path.join(rsp_path, "Particles_dp%s" % dp, "particlos%s.particles" % rsp_step)
            particle_positions = read_particles(ifile)
            ifile = path.join(rsp_path, "Particles_dp%s" % dp, "velo%s" % rsp_step)
            start_velo = read_particles(ifile)

            start_velo = np.zeros(shape=(len(particle_positions),len(particle_positions[0])))
            start_distance = np.zeros(len(particle_positions)) + 1
            properties = {'u_p':start_velo, 'dist': start_distance, 'x1': particle_positions}
            lp_dict["%s" % dp].add_particles(particle_positions,properties)

        restart_particles[0]

	if MPI.rank(mpi_comm_world()) == 0:
	    print "tstep", tstep

        if tstep >= start_particles and tstep <= stop_particles:
            particle_positions = RandomCircle3D([0.0, 0.0, 0.0], 0.01/2 - 0.01*0.1, 5).generate_y_plane()
            start_velo = np.zeros(shape=(len(particle_positions),len(particle_positions[0])))
            start_distance = np.zeros(len(particle_positions)) + 1
            properties = {'u_p':start_velo, 'dist': start_distance, 'x1': particle_positions}

            for lp_instance in lp_dict.itervalues():
                lp_instance.add_particles(particle_positions,properties)

        if tstep >= start_particles:
            for lp_instance in lp_dict.itervalues():
                assign(uv.sub(0), u_[0])
                assign(uv.sub(1), u_[1])
                assign(uv.sub(2), u_[2])
                lp_instance.step(uv, dt)
		if tstep % plot_particles == 0:
		    lp_instance.scatter(tstep, 1000)






        # Sample at probes
        eval_dict["centerline_u_x_probes"](u_[0])
        eval_dict["centerline_u_y_probes"](u_[1])
        eval_dict["centerline_u_z_probes"](u_[2])
        eval_dict["centerline_p_probes"](p_)

        eval_dict["A1A2_u_x_probes"](u_[0])
        eval_dict["A1A2_u_y_probes"](u_[1])
        eval_dict["A1A2_u_z_probes"](u_[2])

        eval_dict["B1B2_u_x_probes"](u_[0])
        eval_dict["B1B2_u_y_probes"](u_[1])
        eval_dict["B1B2_u_z_probes"](u_[2])

        eval_dict["C1C2_u_x_probes"](u_[0])
        eval_dict["C1C2_u_y_probes"](u_[1])
        eval_dict["C1C2_u_z_probes"](u_[2])

        eval_dict["D1D2_u_x_probes"](u_[0])
        eval_dict["D1D2_u_y_probes"](u_[1])
        eval_dict["D1D2_u_z_probes"](u_[2])

        eval_dict["E1E2_u_x_probes"](u_[0])
        eval_dict["E1E2_u_y_probes"](u_[1])
        eval_dict["E1E2_u_z_probes"](u_[2])

        eval_dict["F1F2_u_x_probes"](u_[0])
        eval_dict["F1F2_u_y_probes"](u_[1])
        eval_dict["F1F2_u_z_probes"](u_[2])

        # Dump probes
        if tstep % dump_stats == 0:
            arr_u_x = eval_dict["centerline_u_x_probes"].array()
            arr_u_y = eval_dict["centerline_u_y_probes"].array()
            arr_u_z = eval_dict["centerline_u_z_probes"].array()
            arr_p = eval_dict["centerline_p_probes"].array()

            A1A2_u_x = eval_dict["A1A2_u_x_probes"].array()
            A1A2_u_y = eval_dict["A1A2_u_y_probes"].array()
            A1A2_u_z = eval_dict["A1A2_u_z_probes"].array()

            B1B2_u_x = eval_dict["B1B2_u_x_probes"].array()
            B1B2_u_y = eval_dict["B1B2_u_y_probes"].array()
            B1B2_u_z = eval_dict["B1B2_u_z_probes"].array()

            C1C2_u_x = eval_dict["C1C2_u_x_probes"].array()
            C1C2_u_y = eval_dict["C1C2_u_y_probes"].array()
            C1C2_u_z = eval_dict["C1C2_u_z_probes"].array()

            D1D2_u_x = eval_dict["D1D2_u_x_probes"].array()
            D1D2_u_y = eval_dict["D1D2_u_y_probes"].array()
            D1D2_u_z = eval_dict["D1D2_u_z_probes"].array()

            E1E2_u_x = eval_dict["E1E2_u_x_probes"].array()
            E1E2_u_y = eval_dict["E1E2_u_y_probes"].array()
            E1E2_u_z = eval_dict["E1E2_u_z_probes"].array()

            F1F2_u_x = eval_dict["F1F2_u_x_probes"].array()
            F1F2_u_y = eval_dict["F1F2_u_y_probes"].array()
            F1F2_u_z = eval_dict["F1F2_u_z_probes"].array()

            if MPI.rank(mpi_comm_world()) == 0:
                filepath = path.join(p_path, "Probes")
                arr_u_x.dump(path.join(filepath, "u_x_%s.probes" % str(tstep)))
                arr_u_y.dump(path.join(filepath, "u_y_%s.probes" % str(tstep)))
                arr_u_z.dump(path.join(filepath, "u_z_%s.probes" % str(tstep)))
                arr_p.dump(path.join(filepath, "p_%s.probes" % str(tstep)))

                A1A2_u_x.dump(path.join(filepath, "A1A2_u_x_%s.probes" % str(tstep)))
                A1A2_u_y.dump(path.join(filepath, "A1A2_u_y_%s.probes" % str(tstep)))
                A1A2_u_z.dump(path.join(filepath, "A1A2_u_z_%s.probes" % str(tstep)))

                B1B2_u_x.dump(path.join(filepath, "B1B2_u_x_%s.probes" % str(tstep)))
                B1B2_u_y.dump(path.join(filepath, "B1B2_u_y_%s.probes" % str(tstep)))
                B1B2_u_y.dump(path.join(filepath, "B1B2_u_y_%s.probes" % str(tstep)))

                C1C2_u_x.dump(path.join(filepath, "C1C2_u_x_%s.probes" % str(tstep)))
                C1C2_u_y.dump(path.join(filepath, "C1C2_u_y_%s.probes" % str(tstep)))
                C1C2_u_y.dump(path.join(filepath, "C1C2_u_y_%s.probes" % str(tstep)))

                D1D2_u_x.dump(path.join(filepath, "D1D2_u_x_%s.probes" % str(tstep)))
                D1D2_u_y.dump(path.join(filepath, "D1D2_u_y_%s.probes" % str(tstep)))
                D1D2_u_z.dump(path.join(filepath, "D1D2_u_z_%s.probes" % str(tstep)))

                E1E2_u_x.dump(path.join(filepath, "E1E2_u_x_%s.probes" % str(tstep)))
                E1E2_u_y.dump(path.join(filepath, "E1E2_u_y_%s.probes" % str(tstep)))
                E1E2_u_z.dump(path.join(filepath, "E1E2_u_z_%s.probes" % str(tstep)))

                F1F2_u_x.dump(path.join(filepath, "F1F2_u_x_%s.probes" % str(tstep)))
                F1F2_u_y.dump(path.join(filepath, "F1F2_u_y_%s.probes" % str(tstep)))
                F1F2_u_z.dump(path.join(filepath, "F1F2_u_z_%s.probes" % str(tstep)))

            MPI.barrier(mpi_comm_world())

            eval_dict["centerline_u_x_probes"].clear()
            eval_dict["centerline_u_y_probes"].clear()
            eval_dict["centerline_u_z_probes"].clear()
            eval_dict["centerline_p_probes"].clear()

            eval_dict["A1A2_u_x_probes"].clear()
            eval_dict["A1A2_u_y_probes"].clear()
            eval_dict["A1A2_u_z_probes"].clear()

            eval_dict["B1B2_u_x_probes"].clear()
            eval_dict["B1B2_u_y_probes"].clear()
            eval_dict["B1B2_u_z_probes"].clear()

            eval_dict["C1C2_u_x_probes"].clear()
            eval_dict["C1C2_u_y_probes"].clear()
            eval_dict["C1C2_u_z_probes"].clear()

            eval_dict["D1D2_u_x_probes"].clear()
            eval_dict["D1D2_u_y_probes"].clear()
            eval_dict["D1D2_u_z_probes"].clear()

            eval_dict["E1E2_u_x_probes"].clear()
            eval_dict["E1E2_u_y_probes"].clear()
            eval_dict["E1E2_u_z_probes"].clear()

            eval_dict["F1F2_u_x_probes"].clear()
            eval_dict["F1F2_u_y_probes"].clear()
            eval_dict["F1F2_u_z_probes"].clear()
