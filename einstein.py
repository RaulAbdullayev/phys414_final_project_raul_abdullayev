import numpy
from matplotlib import pyplot
from scipy.integrate import solve_ivp
from scipy.sparse import diags


# calculates derivatives of mass, pressure, and of the time dilation factor based on
# Tolman - Oppenheimer - Volkoff equations.
def tom_equations_ode_sys(r, system, k):
    # extract mass, time dilation factor, and pressure from the system variable
    mass = system[0]
    pressure = system[2]
    rho = numpy.sqrt(pressure / k)  # calculate current density based on pressure from polytropic EOS, n = 1

    mass_derivative = 4 * numpy.pi * r**2 * rho

    if r == 0:  # otherwise, general expression is undefined
        dilation_derivative = 0
        mass_baryonic_derivative = 0
    else:
        dilation_derivative = 2 * ((mass + 4 * numpy.pi * r**3 * pressure) / (r * (r - 2 * mass)))
        mass_baryonic_derivative = 4 * numpy.pi * 1 / numpy.sqrt(1 - 2 * mass / r) * r ** 2 * rho

    pressure_derivative = -1 / 2 * (rho + pressure) * dilation_derivative

    return [mass_derivative, dilation_derivative, pressure_derivative, mass_baryonic_derivative]


# returns 0 when pressure of the star at a given r becomes 0, i.e., surface is reached
# this will be used by solve_ivp() as a stopping signal
def pressure_root(r, system, k):
    return system[2]


pressure_root.terminal = True  # ensures that pressure_root() = 0 is a stopping signal
pressure_root.direction = -1.00


# evolves TOM equations for a star of a given central density, returns its radius and mass
def evolve_tom_equations(rho_center, k):
    mass_initial = 0  # mass at the origin
    dilation_initial = 0  # dilation factor at the origin, simplified assumption
    pressure_initial = k * rho_center ** 2  # pressure at the origin based on central density, from polytropic EoS
    mass_baryonic_initial = 0

    initial_conditions = [mass_initial, dilation_initial, pressure_initial, mass_baryonic_initial]
    radius_bound = 100  # upper boundary on the star radius. In practice, routine will stop when pressure = 0

    solution = solve_ivp(tom_equations_ode_sys, t_span=[0, radius_bound], y0=initial_conditions, events=pressure_root,
                         args=[k])
    radius = solution.t[-1]  # last step value corresponds to the radius
    mass = solution.y[0, - 1]  # y[0] is the mass vector, so access final mass value
    mass_baryonic = solution.y[3, -1]

    return radius, mass, mass_baryonic


# return the matrix for discretized first derivative for n points
def first_derivative_matrix(n):
    main_diagonal = numpy.zeros(n)
    main_diagonal[0] = -1  # staggered stencil, since no BCs on the mass
    main_diagonal[-1] = 1
    upper_diagonal = numpy.ones(n-1)
    lower_diagonal = -1 * numpy.ones(n-1)

    diagonals = [main_diagonal, upper_diagonal, lower_diagonal]
    diagonal_coefficients = [0, 1, -1]

    return diags(diagonals, diagonal_coefficients).toarray()


def einstein_a_b_c():
    rho_center_vec = numpy.linspace(1e-3, 9e-3)  # central densities on the order of 10^-3 (geometric units)
    radius_data = numpy.zeros_like(rho_center_vec)
    mass_data = numpy.zeros_like(rho_center_vec)
    mass_baryonic_data = numpy.zeros_like(rho_center_vec)

    k = 100

    # evolve TOM equations for stars with all of the above densities
    for i in range(rho_center_vec.shape[0]):
        radius_data[i], mass_data[i], mass_baryonic_data[i] = evolve_tom_equations(rho_center_vec[i], k)
        radius_data[i] *= 1.477  # convert radii to km, based on G * Msolar / c^2 = 1477 m

    # plot M - R data
    pyplot.plot(radius_data, mass_data)
    pyplot.xlabel("Radius (in km)")
    pyplot.ylabel("Mass (in solar masses)")
    pyplot.title("Mass - Radius Plot for Neutron Stars")
    pyplot.show()

    fractional_binding_energies = (mass_baryonic_data - mass_data) / mass_data
    # plot fractional binding energy vs
    pyplot.plot(radius_data, fractional_binding_energies)
    pyplot.xlabel("Radius (in km)")
    pyplot.ylabel("Fractional binding energy")
    pyplot.title("NS Fractional Energy as a Function of Radius")
    pyplot.show()

    solar_mass = 1.989e30  # kg
    length_unit = 1477  # m

    rho_center_vec *= solar_mass / length_unit**3  # density unit conversion to kg m^-3

    mass_stable = []
    mass_unstable = []
    drho = rho_center_vec[1] - rho_center_vec[0]  # central density differential
    ddrho = 1 / (2 * drho) * first_derivative_matrix(rho_center_vec.shape[0])  # derivative matrix w.r.t rho central
    mass_density_derivative = ddrho @ mass_data  # get mass derivatives w.r.t density for each data point
    for i in range(0, rho_center_vec.shape[0]):  # store mass values in different arrays depending on derivative sign
        if mass_density_derivative[i] >= 0:
            mass_stable.append(mass_data[i])
        else:
            mass_unstable.append(mass_data[i])

    # plot mass vs central density differently, depending on the dm/drho stability criterion
    pyplot.plot(rho_center_vec[:len(mass_stable)], mass_stable)
    pyplot.plot(rho_center_vec[len(mass_stable)::], mass_unstable, '--')
    pyplot.xlabel("Central density (geometric units)")
    pyplot.ylabel("Mass (solar masses)")
    pyplot.title("NS Mass as a Function of Central Density")
    pyplot.legend(['Stable mass range','Unstable mass range'])
    pyplot.show()


def einstein_d():
    k_vec = numpy.linspace(80, 120)  # potential K values in polytropic index equation, evenly spread from K = 100
    rho_center_vec = numpy.linspace(1e-3, 9e-3)  # central densities on the order of 10^-3 (geometric units)
    mass_max_data = numpy.zeros_like(k_vec)

    for i in range(0, k_vec.shape[0]):  # get maximum mass value for a given K
        mass_data = numpy.zeros_like(rho_center_vec)

        for j in range(0, rho_center_vec.shape[0]):  # ...by getting mass data for each rho central, like in part (c)
            dummy1, mass_data[j], dummy2 = evolve_tom_equations(rho_center_vec[j], k_vec[i])

        mass_max_data[i] = mass_data.max()  # record maximum for this K

    # plot maximum allowed mass as a function of K
    pyplot.plot(k_vec, mass_max_data)
    pyplot.xlabel("K, polytropic EoS coefficient")
    pyplot.ylabel("Maximum allowed NS mass (solar masses)")
    pyplot.title("Maximum NS Mass as a Function of Polytropic EoS Coefficient, K")
    pyplot.show()


def main():
    einstein_a_b_c()
    einstein_d()

