import multiprocessing
import numpy as np
from scipy import linalg

import integrals3D
import orthogonalization
import params

np.set_printoptions(precision=3, suppress=True)
n_states = params.n_states

###############################################################################################################

energies_list = np.loadtxt(params.directory_numerov + 'E_levels_l=0_b=' + str(params.b) + '_1.dat')
mask = np.ones(params.n_l[0], dtype=bool)
energies_list = energies_list[mask]
for l_value in range(1, params.l_max + 1):
    l_list = np.loadtxt(params.directory_numerov + 'E_levels_l=' + str(l_value) + '_b=' + str(params.b) + '_1.dat')
    mask = np.ones(params.n_l[l_value], dtype=bool)
    energies_l = l_list[mask]
    energies_list = np.concatenate((energies_list, energies_l))
print energies_list

l = np.zeros((params.n_l[0]), dtype=int)
for l_value in range(1, params.l_max + 1):
    ll = l_value * np.ones((params.n_l[l_value]), dtype=int)
    l = np.concatenate((l, ll))


# lista rmax
r_max_list = integrals3D.rmax_value(energies_list)

# wyniki z numerova
numerov = integrals3D.numerov_results(l, energies_list)
numerov_x = numerov[0]
numerov_y = numerov[1]
# print type(numerov_y)

# normalize functions
norm = integrals3D.norm_matrix(n_states, r_max_list, numerov_x, numerov_y)

# calculate SPP
spp = integrals3D.spp_1d_integrate(n_states, norm, r_max_list, numerov_x, numerov_y)


def solve(i_min, i_max):
    d_vals = np.arange(params.d_min + i_min * params.delta_d, params.d_min + i_max * params.delta_d, params.delta_d)
    i = 0
    for d in d_vals:
        i += 1
        print ("d = " + str(d) + " " + str(i) + "/" + str(len(d_vals)))
        spm = integrals3D.spm_1d_integrate(n_states, d, norm, r_max_list, numerov_x, numerov_y)
        print 'spm calculated'
        app = integrals3D.app_1d_integrate(n_states, d, norm, r_max_list, numerov_x, numerov_y)
        amm = integrals3D.amm_1d_integrate(n_states, d, norm, r_max_list, numerov_x, numerov_y)
        apm = integrals3D.apm_1d_integrate(n_states, d, norm, r_max_list, numerov_x, numerov_y)
        print 'A calculated'
        rpp = params.kappa * integrals3D.rpp_1d_integrate(n_states, d, norm, params.b, r_max_list, numerov_x, numerov_y)
        rmm = params.kappa * integrals3D.rmm_1d_integrate(n_states, d, norm, params.b, r_max_list, numerov_x, numerov_y)
        rpm = params.kappa * integrals3D.rpm_1d_integrate(n_states, d, norm, params.b, r_max_list, numerov_x, numerov_y)
        print 'R calculated'

        S = orthogonalization.calculate_matrix(spp, spm, spm.T, spp)
        np.savetxt(
            params.directory + 'ResultsR4/Matrices/' + str(params.n_states) + 'S_d=' + str(d) + '.dat', S, fmt="%3f")
        A = orthogonalization.calculate_matrix(app, apm, apm.T, amm)
        np.savetxt(
            params.directory + 'ResultsR4/Matrices/' + str(params.n_states) + 'A_d=' + str(d) + '.dat', A, fmt="%3f")
        R = orthogonalization.calculate_matrix(rpp, rpm, rpm.T, rmm)
        np.savetxt(
            params.directory + 'ResultsR4/Matrices/' + str(params.n_states) + 'R_d=' + str(d) + '.dat', R, fmt="%3f")
        H, E = orthogonalization.calculate_H(S, A, R, d, energies_list)
        np.savetxt(
            params.directory + 'ResultsR4/Matrices/' + str(params.n_states) + 'H_d=' + str(d) + '.dat', H, fmt="%3f")

        S = np.loadtxt(params.directory + 'ResultsR4/Matrices/' + str(params.n_states) + 'S_d=' + str(d) + '.dat')
        # S = S.round(3)
        H = H.round(4)
        inv_sqrt_s = linalg.inv(linalg.sqrtm(S))
        inv_sqrt_s = inv_sqrt_s.round(4)
        H2 = inv_sqrt_s.dot(H.dot(inv_sqrt_s))
        H2 = H2.round(4)
        # np.savetxt(
        #     params.directory + 'Results/Matrices/k=' + str(params.kappa) + '/H_k=' + str(params.kappa) + '_d=' + str(
        #         d) + '.dat',
        #     H2, fmt="%3f")
        # H2 = np.loadtxt(params.directory + 'Results/Matrices/k=' + str(params.kappa) + '/H_k=' + str(params.kappa) + '_d=' + str(
        #         d) + '.dat').view(complex).reshape(-1)
        eig = linalg.eigvals(H2)
        eigvecs = linalg.eig(H2)[1]
        eigvecsMultiplied = []
        for vector in eigvecs:
            vectorN = inv_sqrt_s.dot(vector)
            # print vectorN
            eigvecsMultiplied.append(vectorN)
        np.savetxt(params.directory + 'ResultsR4/Eigenvalues/' + str(params.n_states) + 'eigvals_d=' + str(d) + '.dat',
                   np.real(eig), fmt="%3f")
        np.savetxt(params.directory + 'ResultsR4/Eigenvectors/' + str(params.n_states) + 'eigvecs_d=' + str(d) + '.dat',
                   np.real(eigvecsMultiplied), fmt="%3f")


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
