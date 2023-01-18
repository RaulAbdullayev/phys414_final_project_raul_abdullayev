import numpy
import csv
from matplotlib import pyplot
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp


# reads white_dwarf_data.csv file, converts log(g) surface gravity of a white dwarf to its radius in earth radii units,
# returns array of the radii and masses
def get_data():
    white_dwarf_data = []
    with open('white_dwarf_data.csv', newline='') as white_dwarf_file:  # assumes the file is in the working directory
        data_reader = csv.reader(white_dwarf_file)
        next(white_dwarf_file)  # skip the header

        for row in data_reader:
            data_raw = row
            surface_gravity = float(data_raw[1])  # log10(g) in cm
            mass = float(data_raw[2])  # in solar masses

            gravitational_const = 6.6743e-11  # m^3 kg^-1 s^-2
            mass_si = mass * 1.989e+30  # in kg
            acceleration_si = 10 ** surface_gravity * 1e-2  # in meters, no log

            earth_radius = 6731000  # in meters
            radius = numpy.sqrt(gravitational_const * mass_si / acceleration_si) / earth_radius  # in earth radii
            white_dwarf_data.append([radius, mass])

    return numpy.array(white_dwarf_data)


# returns the number of data points in the white dwarf data, whose radius is above the cutoff value
# only these data points will be used for the linear mass - radius fit
def data_point_counter(radius_data):
    cutoff_radius = 1.45
    lin_fit_radius_data = []

    i = 0
    while radius_data[i] > cutoff_radius:
        i = i + 1

    return i


# fits log of radius to log of mass of the white dwarf, returns the slope of the linear fit m, where m = (3 - n)/(1 - n)
# and n - estimate for the poly-tropic index
def mass_radius_lin_fit(radius_data, mass_data):
    lin_fit = numpy.polynomial.Polynomial.fit(numpy.log(radius_data), numpy.log(mass_data), deg=1)  # Polynomial object
    lin_fit_coeffs = [float(i) for i in lin_fit.convert().coef]  # get coefficients from the polynomial object lin_fit

    return lin_fit_coeffs[1]  # returns slope


# returns Taylor expansion solution of Lane - Emden equation. This will be used as the objective function by fsolve()
# to determine the scaled radius of low-mass WDs. Since radius in the Lane - Emden equation is scaled by central
# density of a star and low-mass WDs share an EoS, they can be considered to have a single scaled radius.
def lane_emden_taylor(x, n):
    return 1 - 1 / 6 * x ** 2 + n / 120 * x ** 4 - n * (8 * n - 5) / 15120 * x ** 6


# given parameters theta & 1st derivative of theta w.r.t xi, returns the 2nd derivative and the 1st derivative.
def lane_emden_ode_sys(x, system, n):
    dtheta_dx = system[0]
    theta = system[1]

    if x == 0:
        d2theta_dx2 = -1 * theta ** n  # at the origin, the other term vanishes
    else:
        d2theta_dx2 = -2 / x * dtheta_dx - theta ** n  # by expanding Lane - Emden & isolating for 2nd derivative

    return [d2theta_dx2, dtheta_dx]


def theta_root(x, system, n):
    return system[1]


theta_root.terminal = True


# evolves Lane - Emden equation until the surface is reached. Upper boundary on the radius is calculated from
# the results of Taylor series root-finding. In practice, solve_ivp() stops before the upper bound, when theta = 0, i.e,
# surface is reached.
def get_lane_emden_radius(n, radius_guess):
    lane_emden_radius_limit = 2 * radius_guess  # twice the guess from the Taylor series
    solution = solve_ivp(lane_emden_ode_sys, t_span=[0, lane_emden_radius_limit], y0=[0, 1], events=theta_root, args=[n])
    lane_emden_radius = solution.t[-1]
    dtheta_surface = solution.y[0, -1]

    return lane_emden_radius, dtheta_surface


def newton_b(radius_data, mass_data):
    # plot white dwarf masses as a function of radius (experimental data)
    marker_area = 5 * numpy.ones(radius_data.shape[0])  # parameter for sizes of the points on the scatter plot
    pyplot.scatter(radius_data, mass_data, s=marker_area)
    pyplot.title("White dwarf mass vs radius scatter plot")
    pyplot.xlabel("radius (in average earth radii)")
    pyplot.ylabel("mass (in solar masses)")
    pyplot.show()


def newton_c(radius_data, mass_data):
    lin_fit_count = data_point_counter(radius_data)  # get number of points to be used in the linear fit
    mass_radius_slope = mass_radius_lin_fit(radius_data[:lin_fit_count],
                                            mass_data[:lin_fit_count])  # get slope of linear mass - radius fit
    n = (mass_radius_slope - 3) / (mass_radius_slope - 1)  # polytropic index approx, via slope = (3 - n) / (1 - n)
    q_fit_param = int(numpy.rint(5 * n / (n + 1)))  # via n* = q / (5 - q)
    print("q = ", q_fit_param)

    n = q_fit_param / (5 - q_fit_param)  # update n* based on integer value of q
    mass_radius_slope = (3 - n) / (1 - n)  # update the slope based on the updated n*
    y_intercept = numpy.mean(numpy.log(mass_data[:lin_fit_count]) - mass_radius_slope
                             * numpy.log(radius_data[:lin_fit_count]))  # from least squares formula

    # get scaled radius of low-mass WD. for explanation, see comment to lane_emden_taylor() definition above
    radius_lane_emden = fsolve(lane_emden_taylor, x0=numpy.ones(1), args=numpy.array([n]))[0]
    print("Radius from Taylor series: ", radius_lane_emden)

    # refine lane - emden radius by evolving the lane - emden equation, but using the value obtained via Taylor series
    # as initial guess
    radius_lane_emden, dtheta_surface = get_lane_emden_radius(n, radius_lane_emden)
    print("Radius from Lane - Emden integration: ", radius_lane_emden)

    # dTheta / dXi in Lane - Emden equation, evaluated at the star surface (i.e., at the obtained scaled radius value)
    # The derivative expression was obtained by differentiating first 4 terms of Taylor expansion for the solution of
    # Lane - Emden equation
    derivative_lane_emden = -2 / 6 * radius_lane_emden + 4 * n / 120 * radius_lane_emden ** 3 \
                            - 6 * n * (8 * n - 5) / 15120 * radius_lane_emden ** 5

    # dimensionless polytropic-index-dependent constant that figures in the expression for y-intercept of the linear fit
    Nn = numpy.power(4 * numpy.pi, 1 / n) / (n + 1) * numpy.power(-1 * radius_lane_emden ** 2 * derivative_lane_emden,
                                                                  (1 - n) / n) * numpy.power(radius_lane_emden,
                                                                                             (n - 3) / n)

    gravitational_const = 6.6743e-11  # m^3 kg^-1 s^-2
    solar_mass = 1.989e+30  # in kg
    earth_radius = 6731000  # in meters
    gravitational_const *= 1 / earth_radius ** 3 * solar_mass  # unit conversion to (earth radii^3) (solar mass^-1) s^-2

    # using y intercept = n / (n-1) * ln(K / (G * Nn)) & isolating for K
    k_fit_param = gravitational_const * Nn * numpy.exp((n - 1) * y_intercept / n)
    print("K* = ", k_fit_param)

    pyplot.plot(numpy.log(radius_data[:lin_fit_count]),
                mass_radius_slope * numpy.log(radius_data[:lin_fit_count]) + y_intercept)  # plot linear fit

    marker_area = 5 * numpy.ones(lin_fit_count)  # parameter for sizes of the points on the scatter plot
    pyplot.scatter(numpy.log(radius_data[:lin_fit_count]),
                   numpy.log(mass_data[:lin_fit_count]), s=marker_area, c='red')  # plot data used in linear fit
    pyplot.xlabel("log radius (in average earth radii)")
    pyplot.ylabel("log mass (in solar masses)")
    pyplot.legend(['Linear Fit', 'Experimental Data'])
    pyplot.title("Low-mass WD mass - radius data and its linear fit")
    pyplot.show()

    rho_center_data = numpy.zeros_like(radius_data)  # array to store central densities of WDs

    # calculate central density for each star from the data set
    for i in range(radius_data.shape[0]):
        m = mass_data[i]
        rho_center = numpy.power(m * numpy.power(k_fit_param / gravitational_const * (n + 1) / (4 * numpy.pi), -3/2) /
                                 (4 * numpy.pi * -1 * radius_lane_emden**2 * dtheta_surface), 2 * n / (3 - n))
        rho_center_data[i] = rho_center

    marker_area = 5 * numpy.ones(mass_data.shape[0])  # parameter for sizes of the points on the scatter plot
    pyplot.scatter(mass_data, rho_center_data, s=marker_area)  # plot central density vs mass
    pyplot.xlabel("mass (in solar masses)")
    pyplot.ylabel("central density (in solar masses * earth radii^-3)")
    pyplot.title("WD central density as a function of mass")
    pyplot.show()


def main():
    # get white dwarf radius and mass data from the file
    white_dwarf_data = get_data()
    radius_data = white_dwarf_data[:, 0]
    mass_data = white_dwarf_data[:, 1]

    # sort radius and mass data in corresponding fashion w/ radius as the key. numpy.sort() breaks correspondence
    indices = radius_data.argsort()[::-1]
    radius_data = radius_data[indices]
    mass_data = mass_data[indices]

    newton_b(radius_data, mass_data)
    newton_c(radius_data, mass_data)

