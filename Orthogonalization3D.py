import multiprocessing
import numpy as np
from scipy import linalg

import integrals3D
import orthogonalization
import params

np.set_printoptions(precision=8, suppress=True)
n_states = params.n_states

###############################################################################################################

energies_list = np.loadtxt(params.basis_file())
mask = np.ones(params.n_l[0], dtype=bool)
energies_list = energies_list[mask]

# lista rmax
r_max_list = integrals3D.rmax_value(energies_list)


# calculate SPP
if not params.AllMatricesCalculated:
    # normalize functions
    norm = integrals3D.norm_matrix(n_states, r_max_list)
    spp = integrals3D.spp_3d_integrate(n_states, norm, r_max_list)
    A = integrals3D.A_3d_integrate(n_states, norm, r_max_list)
    np.savetxt(params.dir('matrices') +  str(params.n_states) + 'A.dat', A, fmt="%6f")
    print spp


def solve(i_min, i_max):
    d_vals = np.arange(params.d_min + i_min * params.delta_d, params.d_min + i_max * params.delta_d, params.delta_d)
    i = 0
    for d in d_vals:
        i += 1
        print ("d = " + str(d) + " " + str(i) + "/" + str(len(d_vals)))
        if not params.AllMatricesCalculated:
            spm = integrals3D.spm_3d_integrate(n_states, d, norm, r_max_list)
            app = -A + d * spp
            amm = A + d * spp
            apm = np.zeros(n_states, n_states)
            # rpp = integrals3D.rpp_3d_integrate(n_states, d, norm, params.b, r_max_list)
            # rmm = integrals3D.rmm_3d_integrate(n_states, d, norm, params.b, r_max_list)
            # rpm = integrals3D.rpm_3d_integrate(n_states, d, norm, params.b, r_max_list)

            S = orthogonalization.calculate_matrix(spp, spm, spm.T, spp)
            np.savetxt(
                params.dir('matrices') + str(params.n_states) + 'S_d=' + str(d) + '.dat', S, fmt="%6f")
            A = 2 * params.alpha * d * orthogonalization.calculate_matrix(app, apm, apm.T, amm)
            np.savetxt(
                params.dir('matrices')+ str(params.n_states) + 'A_d=' + str(d) + '.dat', A, fmt="%6f")
            # R = - orthogonalization.calculate_matrix(rpp, rpm, rpm.T, rmm)
            # np.savetxt(
            #     params.dir('matrices') + str(params.n_states) + 'R_d=' + str(d) + '.dat', R, fmt="%6f")
            H, E = orthogonalization.calculate_H(S, 2 * params.alpha * d * A, R, d, energies_list)
            np.savetxt(
                params.dir('matrices') + str(params.n_states) + 'H_d=' + str(d) + '.dat', H, fmt="%6f")
        if params.AllMatricesCalculated:
            S = np.loadtxt(params.dir('matrices') + str(params.n_states) + 'S_d=' + str(d) + '.dat')
            R = np.loadtxt(params.dir('matrices') + str(params.n_states) + 'R_d=' + str(d) + '.dat')
            A = np.loadtxt(params.dir('matrices') + str(params.n_states) + 'A_d=' + str(d) + '.dat')
            H, E = orthogonalization.calculate_H(S, 2 * params.alpha * d * A, R, d, energies_list)
        # S = S.round(3)
        # H = H.round(3)
        inv_sqrt_s = linalg.inv(linalg.sqrtm(S))
        # inv_sqrt_s = inv_sqrt_s.round(5)
        H2 = inv_sqrt_s.dot(H.dot(inv_sqrt_s))
        # H2 = H2.round(3)
        # np.savetxt(
        #     params.directory + 'Results/Matrices/k=' + str(params.kappa) + '/H_k=' + str(params.kappa) + '_d=' + str(
        #         d) + '.dat',
        #     H2, fmt="%3f")
        # H2 = np.loadtxt(params.directory + 'Results/Matrices/k=' + str(params.kappa) + '/H_k=' + str(params.kappa) + '_d=' + str(
        #         d) + '.dat').view(complex).reshape(-1)
        eig = linalg.eigvals(H2)
        np.savetxt(params.dir('eigenvalues') + str(params.n_states) + 'eigvals_d=' + str(d) + '.dat',
                   np.real(eig), fmt="%3f")


# ********************************
num_workers = multiprocessing.cpu_count()
n_d = int((params.d_max - params.d_min) / params.delta_d)
n_per_process = n_d / (num_workers - 1) +1
n_d_last = n_d - (num_workers -1) * n_per_process
print n_d
print('n_per_process:', n_per_process)
print ('n_dlast: ', n_d_last)

processes = []

for x in range(num_workers - 1):
    processes.append(multiprocessing.Process(target=solve, args=(x * n_per_process, (x + 1) * n_per_process)))
    print(
        "process " + str(x) + "  d_min= " + str(
            params.d_min + x * n_per_process * params.delta_d) + ",  d_max = " + str(
            params.d_min + ((x + 1) * n_per_process) * params.delta_d))
processes.append(multiprocessing.Process(target=solve, args=(
    (num_workers - 1) * n_per_process, (num_workers-1) * n_per_process + n_d_last)))
print(
    "process " + str(num_workers - 1) + "  d_min= " + str(
        params.d_min + (num_workers - 1) * n_per_process * params.delta_d) + ",  d_max = " + str(
        params.d_min + (num_workers * n_per_process ) * params.delta_d))
d_Max = params.d_min + (num_workers * n_per_process ) * params.delta_d
if __name__ == '__main__':
    for x in range(num_workers):
        processes[x].start()