#utilities 
import numpy as np

AU = 149597870700.  # Astronomical Unit in meters
Msjup = 1.047348644e3 # Msun to Mjup
Rsun = 696000. #  Sun radius in km
Rjup = 71492.  #  Jupiter radius in km
Rsjup = Rsun/Rjup # Rsun to Rjup
Gsi = 6.67428e-11  # Gravitational Constant in SI system [m^3/kg/s^2]
Msun = 1.9884e30 # Sun mass in kg
Msear = 332946.0487 # Msun to Mear
Mears = 1./Msear  #  Mear to Msun
d2s = 86400. 
RsunAU = (Rsun*1.e3)/AU #Sun radius in AU
RjupAU = (Rjup*1.e3)/AU #Jupiter radius in AU
rho_Sun = Msun / (4./3.*np.pi* (Rsun*1000.)**3 )

def compute_value_sigma(samples):
    if np.size(np.shape(samples)) == 1:
        sample_med = np.zeros(3)
        sample_tmp = np.percentile(samples, [15.865, 50, 84.135], axis=0)
        sample_med[0] = sample_tmp[1]
        sample_med[1] = sample_tmp[2] - sample_tmp[1]
        sample_med[2] = sample_tmp[0] - sample_tmp[1]

    elif np.size(np.shape(samples)) == 2:
        sample_med = np.asarray(
            list(map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), zip(*np.percentile(samples, [15.865, 50, 84.135], axis=0)))))

    else:
        print(' Error on computing the median and 1-sigma of the distributions! ')
        return None
    return sample_med

def return_significant_figures(perc0, perc1=None, perc2=None, are_percentiles=False):

        if perc1==None and perc2==None:
            if np.isnan(perc0):
                return 0

            abs_value = np.abs(perc0)
            if abs_value > 10.:
                return 6
            elif  abs_value > 1:
                return 6
            elif abs_value == 0:
                return 2
            else:
                return 6
                #x1 = np.log10(abs_value)
                #return int(np.ceil(abs(x1))+1)

        elif perc2==None:
            if np.isnan(perc0) or np.isnan(perc1):
                return 0
            value_err = np.abs(perc1)

            if value_err > 10. or value_err > 10.:
                return 0
            elif  value_err > 1. or value_err > 1.:
                return 1
            else:
                x0 = np.log10(value_err)
                return int(np.ceil(abs(x0))+1)
        else:
            if np.isnan(perc0) or np.isnan(perc1) or np.isnan(perc2):
                return 2, 2

            if are_percentiles:
                minus_err = np.abs(perc0 - perc1)
                plus_err = np.abs(perc2 - perc1)
            else:
                minus_err = np.abs(perc1)
                plus_err = np.abs(perc2)


            if minus_err > 10.:
                sig_minus = 0
            elif minus_err > 1.:
                sig_minus = 1
            else:
                try:
                    x0 = np.log10(minus_err)
                    sig_minus = int(np.ceil(abs(x0))+1)
                except OverflowError:
                    sig_minus = 6

            if plus_err > 10.:
                sig_plus = 0
            elif plus_err > 1.:
                sig_plus = 1
            else:
                try:
                    x0 = np.log10(plus_err)
                    sig_plus = int(np.ceil(abs(x0))+1)
                except OverflowError:
                    sig_plus = 6

            return sig_minus, sig_plus




from scipy.optimize import fsolve

def f_get_mass(mass_secondary, mass_primary, period, ecc, rv_semiamplitude):
    """ Compute the difference between the input radial velocity semi-amplitude
    of the primary star and the value corresponding to the provided orbital parameters.
    Supporting function to *kepler_get_planet_mass* subroutine.

    constants.Gsi: Gravitational constant in SI system [m^3 kg^-1 s^-2]
    constants.Msun: Sun mass in SI system [kg]

    :param mass_secondary: mass of the secondary/planet (in Solar mass units)
    :param mass_primary: mass of the primary star (in Solar mass units)
    :param period: orbital period of the secondary (in days)
    :param ecc: orbital eccentricity of the planet
    :param rv_semiamplitude: observed RV semi-amplitude of the primary (in m/s)
    :return: the difference between the observed and theoretical RV semi-amplitude of the primary (in m/s)
    """

    return rv_semiamplitude - ((2. * np.pi * Gsi * Msun / 86400.0) ** (1. / 3.)
            * (1. / np.sqrt(1. - ecc ** 2.))
            * period ** (-1. / 3.)
            * (mass_secondary * (mass_primary + mass_secondary) ** (-2. / 3.)))

def get_approximate_mass(period, rv_semiamplitude, ecc, mass_primary):
    """ Return the approximate mass of the planet in Solar mass units, under the assumption that
    the mass of the planet is negligible compared to the mass of the star.
    Supporting function to *kepler_get_planet_mass* subroutine.
    For a more precise calculation, decrease the *approximation_limit* value of the function
    *kepler_get_planet_mass* to enable the Newton-Raphson method. 

    :param period: orbital period of the planet (in days)
    :param rv_semiamplitude: observed RV semi-amplitude of the primary (in m/s)
    :param ecc: orbital eccentricity of star2
    :param mass_primary: mass of the primary star (in Solar mass units)
    :return: mass of the planet (in Solar mass units)
    """
    return rv_semiamplitude / ((2. * np.pi * Gsi * Msun / 86400.0) ** (1. / 3.)
         * (1. / np.sqrt(1. - ecc ** 2.))
         * period ** (-1. / 3.)
         * (mass_primary ** (-2. / 3.)))


def kepler_get_planet_mass(period, rv_semiamplitude, ecc, mass_star, approximation_limit=30.):
    """ Compute the mass of the planet in Solar mass units, given the orbital period (in days),
    the observed RV semi-amplitude (in m/s), the orbital eccentricity, and the mass of the primary star (in Solar mass units).
    If the approximate mass of the planet (computed under the assumption that the mass of the planet is negligible compared to the mass of the star)
    is larger than the specified approximation limit, the function will use the more accurate method to compute the planet's mass. 
    The default value for the approximation limit is 30 Earth masses, which is a good compromise between speed and accuracy.
    To force the more accurate method, decrease the *approximation_limit* value.
    To convert the planetary mass,
    ```python
    import pyorbit.subroutines.constants as constants
    mass_planet_earth = mass_planet_solar * constants.Msear
    mass_planet_jupiter = mass_planet_solar * constants.Msjup
    ```
    :param period: orbital period of the planet (in days)
    :param rv_semiamplitude: observed RV semi-amplitude of the primary (in m/s)
    :param ecc: orbital eccentricity of the planet
    :param mass_star: mass of the primary star (in Solar mass units)
    :param approximation_limit: threshold in Earth mass units to switch between the approximate and the more accurate method (default is 30 Earth masses)
    :return: mass of the planet (in Solar mass units)
    """

    n = np.size(rv_semiamplitude)
    if n == 1:
        M_approx = min(get_approximate_mass(period, rv_semiamplitude, ecc, mass_star), 2*Msear)
        return fsolve(f_get_mass, M_approx, args=(mass_star, period, ecc, rv_semiamplitude))

    M_approx = get_approximate_mass(period, rv_semiamplitude, ecc, mass_star)

    if np.average(M_approx) > approximation_limit/Msear:
        print('Computing exact mass of the planet (mean of approximate mass distribution larger than {0:3.1f} Me)'.format(approximation_limit))
        M_init = np.average(M_approx)
        for i in range(0, n):
            M_approx[i] = fsolve(f_get_mass, np.average(M_init), args=(mass_star[i], period[i], ecc[i], rv_semiamplitude[i]))[0]
    else:
        print('Computing planetary mass under the approximation M_planet << M_star (threshold at {0:3.1f} Me)'.format(approximation_limit))

    return M_approx
