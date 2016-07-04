import params
import matplotlib.pyplot as plt
import numpy as np
# import integrals_analytical
import integrals1D
from numpy import linalg as la
from scipy import integrate

# get eigenvectors
# get list of eigenvalues
xVals = np.arange(-10, 10, 0.01)
# lista norm funkcji falowych bazy
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
# wyniki z numerova
numerov = integrals1D.numerov_results(l, energies_list)
numerov_x = numerov[0]
numerov_y = numerov[1]
r_max_list = integrals1D.rmax_value(energies_list)
norms = integrals1D.norm_list(params.n_states, r_max_list, numerov_x, numerov_y)

d = params.d
eigenVals = np.loadtxt(
        params.directory + 'ResultsR4/Eigenvalues/' + str(params.n_states) + 'eigvals_d=' + str(d) + '.dat')
# eigenVals = np.loadtxt(params.directory_my + 'Results/Eigenvalues/k=' + str(params.kappa) + '/NSeigenVals_baza' + str(
#     params.basis) + '_k=' + str(params.kappa) + '_d=' + str(d) + '.dat')

# eigenVecs = np.real(np.loadtxt(params.directory_my + 'Results/Eigenvectors/k=' + str(params.kappa) + '/eigenVecs_baza' + str(
#     params.basis) + '_k=' + str(params.kappa) + '_d=' + str(d) + '.dat'))

eigenVecs = np.real(np.loadtxt(
        params.directory + 'ResultsR4/Eigenvectors/' + str(params.n_states) + 'eigvecs_d=' + str(d) + '.dat'))



#trzeba wywalic te ktore odpowiadaja zlej wartosci wlasnej, tzn zepolonej
plt.figure()
# plt.ylim(0, 16)
plt.title("Funkcje falowe, baza: $" + str(params.n_states) + "$, b: $" + str(params.b) + " $ ")
plt.xlabel("$x$")
plt.ylabel("$E$")
eigValInd = -1
for eigenValue in eigenVals:
    eigValInd += 1
    functionVals = []
    xV = []
    eigenVector =eigenVecs[eigValInd]/la.norm(eigenVecs[eigValInd])
    for x in xVals:
        psiXValue = 0
        vecElemIndex = -1
        for vecElem in eigenVector:
            vecElemIndex += 1
            if vecElemIndex < params.n_states:
                psiXValue += vecElem * norms[vecElemIndex] * integrals1D.f(x + d, numerov_x[vecElemIndex], numerov_y[vecElemIndex])
            else:
                psiXValue += vecElem * norms[vecElemIndex -params.n_states] * integrals1D.f(x - d, numerov_x[vecElemIndex - params.n_states], numerov_y[vecElemIndex-params.n_states])
        functionVals.append(psiXValue)
        xV.append(x)
    #normalizacja funkcji falowej
    norm = integrate.quad(lambda xx: np.interp(xx, xV, functionVals)*np.interp(xx, xV, functionVals), -params.x_max, params.x_max )[0]
    functionNormalized = []
    for val in functionVals:
        functionNormalized.append(1.0/np.sqrt(norm) * val + eigValInd)


    # print functionVals
    # print xVals.shape
    xt = np.arange(-10, 10, 2)
    plt.xticks(np.append(xt, (d, -d)))
    plt.axvline(x=d, linewidth=0.1, color='black', linestyle='dashed')
    plt.axvline(x=-d, linewidth=0.1, color='black', linestyle='dashed')
    plt.plot(xV, functionNormalized , label = 'E = ' + str(eigenValue))

# plt.savefig(params.directory_my + 'Results/Plots/psi_kappa=' + str(params.kappa) + '_baza=' + str(params.basis) + '_d=' + str(d) + '.png')
# plt.savefig(params.directory_my + 'Results/Plots/psi_kappa=' + str(params.kappa) + '_baza=' + str(params.basis) + '_d=' + str(d) + '.pdf')

plt.legend()
plt.show()