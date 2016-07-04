import numpy as np

import matplotlib.pyplot as plt

import params

d_vals = np.arange(params.d_min + params.delta_d, params.d_max, params.delta_d)
n = np.loadtxt( params.directory + 'ResultsR4/Eigenvalues/' + str(params.n_states) + 'eigvals_d=' + str(params.d_min) + '.dat')

# print  n
for d in d_vals:
    n_d = np.loadtxt(
        params.directory + 'ResultsR4/Eigenvalues/' + str(params.n_states) + 'eigvals_d=' + str(d) + '.dat')
    n = np.concatenate((n, n_d), axis=1)

n_vals = np.reshape(n, (d_vals.shape[0] + 1 , n_d.shape[0]))
n_vals=np.transpose(n_vals)
d_vals = np.arange(params.d_min, params.d_max, params.delta_d)
plt.xlabel('$d$')
plt.ylabel('$E$')
plt.title('Poziomy energetyczne, ' + ' baza: $' + str(params.n_states) + '$')
for row in n_vals:
    plt.plot(d_vals, row )

plt.savefig(params.directory_my + 'ResultsR4/Plots/' + str(params.n_states) + '.png')
plt.savefig(params.directory_my + 'ResultsR4/Plots/' + str(params.n_states) + '.pdf')

plt.show()

# ni = np.loadtxt(params.directory + 'Roots/baza' + str(params.kappa_basis) + '_' + str(params.basis) + '.dat', dtype=float)
# x = np.arange(-params.x_max+params.d_max, params.x_max - params.d_max, 0.1)
# plt.figure()
# plt.xlabel("$x$")
# plt.title("Funkcje bazowe, $\\kappa = " + str(params.kappa) + "$, baza: "+ str(params.basis))
# for ni_v in ni:
#     norm = integrals_analytical.normalize(ni_v, 20)
#     vals = []
#
#     for val in x:
#         vals.append(norm * integrals_analytical.f(val, ni_v) + ni_v)
#     # vals = integrals_analytical.normalize(ni_v, 10) * vals
#     plt.plot(x, vals)
# plt.savefig(params.directory_my + 'Results/Plots/Baza_kappa=' + str(params.kappa) + '_' + str(n_d.shape[0] / 2) + '.pdf')
# plt.savefig(params.directory_my + 'Results/Plots/Baza_kappa=' + str(params.kappa) + '_' + str(n_d.shape[0] / 2) + '.png')
# plt.show()
