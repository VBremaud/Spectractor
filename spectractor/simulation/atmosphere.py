import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.interpolate import interp1d, RegularGridInterpolator

from spectractor.config import set_logger
import spectractor.parameters as parameters

import spectractor.simulation.libradtran as libradtran
from spectractor.simulation.throughput import plot_transmission_simple


class Atmosphere:

    def __init__(self, airmass, pressure, temperature):
        """Class to evaluate an atmospheric transmission using Libradtran.

        Parameters
        ----------
        airmass: float
            Airmass of the source object.
        pressure: float
            Pressure of the atmosphere in hPa.
        temperature: float
            Temperature of the atmosphere in Celsius degrees.

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
        >>> print(a.airmass)
        1.2
        >>> print(a.pressure)
        800
        >>> print(a.temperature)
        5
        >>> print(a.transmission(500))
        1.0
        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.airmass = airmass
        self.pressure = pressure
        self.temperature = temperature
        self.pwv = None
        self.ozone = None
        self.aerosols = None
        self.transmission = lambda x: np.ones_like(x).astype(float)
        self.title = ""
        self.label = ""

    def set_title(self):
        """Make a title string for the simulation.

        """
        self.title = f'Atmospheric transmission with z={self.airmass:4.2f}, P={self.pressure:4.2f} hPa, ' \
                     rf'T={self.temperature:4.2f}$\degree$C'

    def set_label(self):
        """Make a label string for the simulation.

        """
        self.label = f'PWV={self.pwv:4.2f}mm, OZ={self.ozone:4.2f}DB, VAOD={self.aerosols:4.2f} '

    def simulate(self, ozone, pwv, aerosols):
        """Simulate the atmosphere transparency with Libradtran given atmospheric composition.

        Values outside the Libradtran simulation range are set to zero.

        Parameters
        ----------
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
        aerosols: float
            VAOD Vertical Aerosols Optical Depth

        Returns
        -------
        transmission: callable
            The transmission function with wavelengths in nm.

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
        >>> transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
        >>> assert transmission is not None
        >>> assert a.transmission(500) > 0
        >>> a.ozone
        400
        >>> a.pwv
        5
        >>> a.aerosols
        0.05
        >>> a.plot_transmission()

        ..plot:

            from spectractor.simulation.atmosphere import Atmosphere
            import matplotlib.pyplot as plt
            a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
            fig = plt.figure()
            plot_transmission_simple(plt.gca(), lambdas, transmission(lambdas), title=a.title, label=a.label)
            plt.show()
        """

        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        self.set_title()
        self.set_label()
        if parameters.VERBOSE:
            self.my_logger.info(f'\n\t{self.title}\n\t\t{self.label}')

        lib = libradtran.Libradtran()
        path = lib.simulate(self.airmass, pwv, ozone, aerosols, self.pressure)
        data = np.loadtxt(path)
        wl = data[:, 0]
        atm = data[:, 1]
        self.transmission = interp1d(wl, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission

    def plot_transmission(self):
        """Plot the atmospheric transmission computed with Libradtran.

        Examples
        --------
        >>> a = Atmosphere(airmass=1.2, pressure=800, temperature=5)
        >>> transmission = a.simulate(ozone=400, pwv=5, aerosols=0.05)
        >>> a.plot_transmission()

        """
        plot_transmission_simple(plt.gca(), parameters.LAMBDAS, self.transmission(parameters.LAMBDAS),
                                 title=self.title, label=self.label)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()


class AtmosphereGrid(Atmosphere):

    def __init__(self, image_filename="", filename="", airmass=1., pressure=800., temperature=10.,
                 pwv_grid=[0, 10, 10], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 10]):
        """Class to load and interpolate grids of atmospheric transmission computed with Libradtran.

        Parameters
        ----------
        image_filename: str, optional
            The original image fits file name from which the grid was computed or has to be computed (default: "").
        filename: str, optional
            The file name of the atmospheric grid if it exists (default: "").
        airmass: float
            Airmass of the source object (default: 1).
        pressure: float
            Pressure of the atmosphere in hPa (default: 800).
        temperature: float
            Temperature of the atmosphere in Celsius degrees (default: 10).
        pwv_grid: array_like
            array_like of 3 numbers for the PWV quantity: min, max, number of simulations (default: [0, 10, 10]).
        ozone_grid: array_like
            array_like of 3 numbers for the ozone quantity: min, max, number of simulations (default: [100, 700, 7]).
        aerosol_grid: array_like
            array_like of 3 numbers for the aerosol quantity: min, max, number of simulations (default: [0, 0.1, 10]).

        Examples
        --------
        >>> a = AtmosphereGrid(filename='./tests/data/reduc_20170530_134_atmsim.fits')
        >>> a.image_filename.split('/')[-1]
        'reduc_20170530_134_spectrum.fits'
        """
        Atmosphere.__init__(self, airmass, pressure, temperature)
        self.my_logger = set_logger(self.__class__.__name__)
        self.image_filename = image_filename
        self.filename = filename
        # ------------------------------------------------------------------------
        # Definition of data format for the atmospheric grid
        # -----------------------------------------------------------------------------

        #  row 0 : count number
        #  row 1 : aerosol value
        #  row 2 : pwv value
        #  row 3 : ozone value
        #  row 4 : data start
        self.index_atm_count = 0
        self.index_atm_aer = 1
        self.index_atm_pwv = 2
        self.index_atm_oz = 3
        self.index_atm_data = 4

        # specify parameters for the atmospheric grid
        self.atmgrid = None
        self.NB_ATM_HEADER = self.index_atm_data + 1
        self.NB_ATM_DATA = len(parameters.LAMBDAS) - 1
        self.NB_ATM_POINTS = 0
        self.AER_Points = np.array([])
        self.OZ_Points = np.array([])
        self.PWV_Points = np.array([])
        self.set_grid(pwv_grid=pwv_grid, ozone_grid=ozone_grid, aerosol_grid=aerosol_grid)

        # the interpolated grid
        self.lambdas = parameters.LAMBDAS
        self.model = None

        self.header = fits.Header()
        if filename != "":
            self.load_file(filename)

    def set_grid(self, pwv_grid=[0, 10, 10], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 10]):
        """Set the size of the simulation grid self.atmgrid before compute it.

        The first column of self.atmgrid will contain the wavelengths set by parameters.LAMBDAS,
        the other columns the future simulations.

        Parameters
        ----------
        pwv_grid: array_like
            array_like of 3 numbers for the PWV quantity: min, max, number of simulations (default: [0, 10, 10]).
        ozone_grid: array_like
            array_like of 3 numbers for the ozone quantity: min, max, number of simulations (default: [100, 700, 7]).
        aerosol_grid: array_like
            array_like of 3 numbers for the aerosol quantity: min, max, number of simulations (default: [0, 0.1, 10]).
        """
        # aerosols
        NB_AER_POINTS = int(aerosol_grid[2])
        AER_MIN = float(aerosol_grid[0])
        AER_MAX = float(aerosol_grid[1])

        # ozone
        NB_OZ_POINTS = int(ozone_grid[2])
        OZ_MIN = float(ozone_grid[0])
        OZ_MAX = float(ozone_grid[1])

        # pwv
        NB_PWV_POINTS = int(pwv_grid[2])
        PWV_MIN = float(pwv_grid[0])
        PWV_MAX = float(pwv_grid[1])

        # definition of the grid
        self.AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
        self.OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
        self.PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

        # total number of points
        self.NB_ATM_POINTS = NB_AER_POINTS * NB_OZ_POINTS * NB_PWV_POINTS

        # create the numpy array that will contains the atmospheric grid
        self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_ATM_HEADER + self.NB_ATM_DATA))
        self.atmgrid[0, self.index_atm_data:] = parameters.LAMBDAS
        self.lambdas = parameters.LAMBDAS

    def compute(self):
        """Compute atmospheric transmissions and fill self.atmgrid.

        The wavelengths used for the computation are the ones set by parameters.LAMBDAS.

        Returns
        -------
        atmospheric_grid: array_like
            The atmospheric grid self.atmgrid.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> atmospheric_grid
        array([[0.000000e+00, 0.000000e+00, 0.000000e+00, ..., 1.099400e+03,
                1.099600e+03, 1.099800e+03],
               [1.000000e+00, 0.000000e+00, 5.000000e+00, ..., 9.520733e-01,
                9.520733e-01, 9.520733e-01],
               [2.000000e+00, 1.000000e-01, 5.000000e+00, ..., 9.213718e-01,
                9.213718e-01, 9.213718e-01]])
        >>> assert np.all(np.isclose(a.atmgrid[0, a.index_atm_data:], parameters.LAMBDAS))
        >>> assert not np.any(np.isclose(a.atmgrid[1, a.index_atm_data:], np.zeros_like(parameters.LAMBDAS), rtol=1e-6))
        >>> assert a.atmgrid.shape == (3, a.index_atm_data+len(parameters.LAMBDAS))
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> a.plot_transmission()
        """
        # first determine the length
        if parameters.VERBOSE or parameters.DEBUG:
            self.my_logger.info(f'\n\tAtmosphere simulations for z={self.airmass:4.2f}, P={self.pressure:4.2f}hPa, '
                                rf'T={self.temperature:4.2f}$\degree$C, for data-file={self.image_filename} ')
        count = 0
        for aer in self.AER_Points:
            for pwv in self.PWV_Points:
                for oz in self.OZ_Points:
                    count += 1
                    # fills headers info in the numpy array
                    self.atmgrid[count, self.index_atm_count] = count
                    self.atmgrid[count, self.index_atm_aer] = aer
                    self.atmgrid[count, self.index_atm_pwv] = pwv
                    self.atmgrid[count, self.index_atm_oz] = oz
                    transmission = super(AtmosphereGrid, self).simulate(oz, pwv, aer)
                    transm = transmission(parameters.LAMBDAS)
                    self.atmgrid[count, self.index_atm_data:] = transm  # each of atmospheric spectrum
        return self.atmgrid

    def plot_transmission(self):
        """Plot the atmospheric transmission contained in the grid.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> a.plot_transmission()
        """
        plt.figure()
        counts = self.atmgrid[1:, self.index_atm_count]
        for count in counts:
            label = f'PWV={self.atmgrid[int(count), self.index_atm_pwv]} ' \
                    f'OZ={self.atmgrid[int(count), self.index_atm_oz]} ' \
                    f'VAOD={self.atmgrid[int(count), self.index_atm_aer]}'
            plot_transmission_simple(plt.gca(), self.lambdas, self.atmgrid[int(count), self.index_atm_data:],
                                     title="Atmospheric grid", label=label)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def plot_transmission_image(self):
        """Plot the atmospheric transmission contained in the grid using imshow.

        Examples
        --------
        >>> a = AtmosphereGrid(filename='tests/data/reduc_20170530_134_atmsim.fits')
        >>> a.plot_transmission_image()
        """
        plt.figure()
        img = plt.imshow(self.atmgrid[1:, self.index_atm_data:], origin='lower', cmap='jet', aspect="auto")
        plt.grid(True)
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title(" Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        if parameters.DISPLAY:
            plt.show()

    def save_file(self, filename=""):
        """Save the atmospheric grid in a fits file.

        Parameters
        ----------
        filename: str
            The output file name.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> assert os.path.isfile('tests/data/reduc_20170605_028_atmsim.fits')
        """
        hdr = fits.Header()

        if filename != "":
            self.filename = filename

        if self.filename == "":
            self.my_logger.error('\n\tNo file name is given...')
        else:
            hdr['ATMSIM'] = "libradtran"
            hdr['SIMVERS'] = "2.0.1"
            hdr['DATAFILE'] = self.image_filename
            hdr['SIMUFILE'] = os.path.basename(self.filename)

            hdr['AIRMASS'] = self.airmass
            hdr['PRESSURE'] = self.pressure
            hdr['TEMPERAT'] = self.temperature
            hdr['NBATMPTS'] = self.NB_ATM_POINTS

            hdr['NBAERPTS'] = self.AER_Points.size
            hdr['AERMIN'] = self.AER_Points.min()
            hdr['AERMAX'] = self.AER_Points.max()

            hdr['NBPWVPTS'] = self.PWV_Points.size
            hdr['PWVMIN'] = self.PWV_Points.min()
            hdr['PWVMAX'] = self.PWV_Points.max()

            hdr['NBOZPTS'] = self.OZ_Points.size
            hdr['OZMIN'] = self.OZ_Points.min()
            hdr['OZMAX'] = self.OZ_Points.max()

            hdr['AER_PTS'] = np.array_str(self.AER_Points)
            hdr['PWV_PTS'] = np.array_str(self.PWV_Points)
            hdr['OZ_PTS'] = np.array_str(self.OZ_Points)
            hdr['NBWLBIN'] = parameters.LAMBDAS.size
            hdr['WLMIN'] = parameters.LAMBDA_MIN
            hdr['WLMAX'] = parameters.LAMBDA_MAX

            hdr['IDX_CNT'] = self.index_atm_count
            hdr['IDX_AER'] = self.index_atm_aer
            hdr['IDX_PWV'] = self.index_atm_pwv
            hdr['IDX_OZ'] = self.index_atm_oz
            hdr['IDX_DATA'] = self.index_atm_data

            hdu = fits.PrimaryHDU(self.atmgrid, header=hdr)
            hdu.writeto(self.filename, overwrite=True)
            if parameters.VERBOSE:
                self.my_logger.info(f'\n\tAtmosphere.save atm-file={self.filename}')

    def load_file(self, filename):
        """Load the atmospheric grid from a fits file and interpolate across the points
        using RegularGridInterpolator. Automatically called from __init__.

        Parameters
        ----------
        filename: str
            The input file name.

        Examples
        --------
        >>> a = AtmosphereGrid(image_filename='tests/data/reduc_20170605_028.fits',
        ... pwv_grid=[5, 5, 1], ozone_grid=[400, 400, 1], aerosol_grid=[0.0, 0.1, 2])
        >>> atmospheric_grid = a.compute()
        >>> a.save_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> assert os.path.isfile('tests/data/reduc_20170605_028_atmsim.fits')
        >>> a.load_file(a.image_filename.replace('.fits', '_atmsim.fits'))
        >>> a.AER_Points
        array([0. , 0.1])
        >>> a.PWV_Points
        array([5.])
        >>> a.OZ_Points
        array([400.])
        """

        if filename != "":
            self.filename = filename

        if self.filename == "":
            self.my_logger.error('\n\tNo file name is given...')
        else:

            hdu = fits.open(self.filename)
            hdr = hdu[0].header
            self.header = hdr

            # hdr['ATMSIM'] = "libradtran"
            # hdr['SIMVERS'] = "2.0.1"
            self.image_filename = hdr['DATAFILE']
            # hdr['SIMUFILE']=os.path.basename(self.file_name)

            self.airmass = hdr['AIRMASS']
            self.pressure = hdr['PRESSURE']
            self.temperature = hdr['TEMPERAT']

            # hope those are the same parameters : TBD !!!!
            self.NB_ATM_POINTS = hdr['NBATMPTS']

            NB_AER_POINTS = hdr['NBAERPTS']
            AER_MIN = hdr['AERMIN']
            AER_MAX = hdr['AERMAX']

            NB_PWV_POINTS = hdr['NBPWVPTS']
            PWV_MIN = hdr['PWVMIN']
            PWV_MAX = hdr['PWVMAX']

            NB_OZ_POINTS = hdr['NBOZPTS']
            OZ_MIN = hdr['OZMIN']
            OZ_MAX = hdr['OZMAX']

            self.AER_Points = np.linspace(AER_MIN, AER_MAX, NB_AER_POINTS)
            self.OZ_Points = np.linspace(OZ_MIN, OZ_MAX, NB_OZ_POINTS)
            self.PWV_Points = np.linspace(PWV_MIN, PWV_MAX, NB_PWV_POINTS)

            NBWLBINS = hdr['NBWLBIN']
            # WLMIN = hdr['WLMIN']
            # WLMAX = hdr['WLMAX']

            self.index_atm_count = hdr['IDX_CNT']
            self.index_atm_aer = hdr['IDX_AER']
            self.index_atm_pwv = hdr['IDX_PWV']
            self.index_atm_oz = hdr['IDX_OZ']
            self.index_atm_data = hdr['IDX_DATA']

            self.atmgrid = np.zeros((self.NB_ATM_POINTS + 1, self.NB_ATM_HEADER + NBWLBINS - 1))

            self.atmgrid[:, :] = hdu[0].data[:, :]

            if parameters.VERBOSE or parameters.DEBUG:
                self.my_logger.info(f'\n\tAtmosphere.load_image atm-file={self.filename}')

            # interpolate the grid
            self.lambdas = self.atmgrid[0, self.index_atm_data:]
            self.model = RegularGridInterpolator((self.lambdas, self.OZ_Points, self.PWV_Points, self.AER_Points), (
                self.atmgrid[1:, self.index_atm_data:].reshape(NB_AER_POINTS, NB_PWV_POINTS,
                                                               NB_OZ_POINTS,
                                                               len(self.lambdas))).T, bounds_error=False, fill_value=0)

    def simulate(self, ozone, pwv, aerosols):
        """Interpolate from the atmospheric grid to get the atmospheric transmission.

        First ozone, second pwv, last aerosols, to respect order of loops when generating the grid

        Parameters
        ----------
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
        aerosols: float
            VAOD Vertical Aerosols Optical Depth

        Examples
        --------
        >>> a = AtmosphereGrid(filename='tests/data/reduc_20170530_134_atmsim.fits')
        >>> lambdas = np.arange(200, 1200)
        >>> fig = plt.figure()
        >>> for pwv in np.arange(5):
        ...     transmission = a.simulate(ozone=400, pwv=pwv, aerosols=0.05)
        ...     plot_transmission_simple(plt.gca(), lambdas, transmission(lambdas),
        ...     title=a.title, label=a.label)
        >>> if parameters.DISPLAY: plt.show()
        """
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        self.set_title()
        self.set_label()
        ones = np.ones_like(self.lambdas)
        points = np.array([self.lambdas, ozone * ones, pwv * ones, aerosols * ones]).T
        atm = self.model(points)
        self.transmission = interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission


from tables import IsDescription, Float32Col, open_file


class AtmospherePoint(IsDescription):

    airmass = Float32Col()
    pressure = Float32Col()
    temperature = Float32Col()
    pwv = Float32Col()
    ozone = Float32Col()
    aerosols = Float32Col()
    wavelength = Float32Col()
    transmission = Float32Col()


class FullAtmosphereGrid():

    def __init__(self, airmass_grid=[1.], pressure_grid=[800.], temperature_grid=[10.],
                 pwv_grid=[0, 10, 10], ozone_grid=[100, 700, 7], aerosol_grid=[0, 0.1, 10]):
        """Class to load and interpolate grids of atmospheric transmission computed with Libradtran.

        Parameters
        ----------
        airmass_grid: array_like
            Airmass of the source object (default: 1).
        pressure_grid: array_like
            Pressure of the atmosphere in hPa (default: 800).
        temperature_grid: array_like
            Temperature of the atmosphere in Celsius degrees (default: 10).
        pwv_grid: array_like
            array_like of 3 numbers for the PWV quantity: min, max, number of simulations (default: [0, 10, 10]).
        ozone_grid: array_like
            array_like of 3 numbers for the ozone quantity: min, max, number of simulations (default: [100, 700, 7]).
        aerosol_grid: array_like
            array_like of 3 numbers for the aerosol quantity: min, max, number of simulations (default: [0, 0.1, 10]).

        Examples
        --------
        >>> a = FullAtmosphereGrid()
        >>> a.airmass_grid
        'reduc_20170530_134_spectrum.fits'
        """
        self.my_logger = set_logger(self.__class__.__name__)

        # the interpolated grid
        self.atmgrid = None
        self.lambdas = parameters.LAMBDAS
        self.model = None
        self.airmass_grid = airmass_grid
        self.temperature_grid = temperature_grid
        self.pressure_grid = pressure_grid
        self.pwv_grid = pwv_grid
        self.ozone_grid = ozone_grid
        self.aerosols_grid = aerosol_grid

    def compute_h5(self):
        """Compute atmospheric transmissions and fill self.atmgrid.

        The wavelengths used for the computation are the ones set by parameters.LAMBDAS.

        Returns
        -------
        atmospheric_grid: array_like
            The atmospheric grid self.atmgrid.

        Examples
        --------
        >>> a = FullAtmosphereGrid(airmass_grid=[1], pressure_grid=[800, 850], temperature_grid=[10],
        ... pwv_grid=[5, 6], ozone_grid=[300, 400], aerosol_grid=[0.0, 0.1])
        >>> atmospheric_grid = a.compute()
        >>> atmospheric_grid
        >>> a.plot_transmission()
        """
        # first determine the length
        self.my_logger.warning(f'\n\tAtmosphere simulations for\n\t\tz={self.airmass_grid}'
                            f'\n\t\tP={self.pressure_grid}hPa\n\t\tT={self.temperature_grid}degC'
                            f'\n\t\tPWV={self.pwv_grid}mm\n\t\tOzone={self.ozone_grid}dobson'
                            f'\n\t\tAerosols={self.aerosols_grid}')
        count = 0
        h5file = open_file("full_atmospheric_grid.h5", mode="w",
                           title="Grid of atmosphere simulations from LibRadTran 2.0.2")
        group = h5file.create_group("/", 'atmospheres', 'Atmospheric simulations')
        table = h5file.create_table(group, 'transmissions', AtmospherePoint, "Atmospheric transmission [no units]")
        for z in self.airmass_grid:
            for pressure in self.pressure_grid:
                for temperature in self.temperature_grid:
                    for pwv in self.pwv_grid:
                        for oz in self.ozone_grid:
                            for aer in self.aerosols_grid:
                                self.my_logger.warning(f"\n\tz={z} P={pressure}hPa T={temperature} PWV={pwv}mm "
                                                       f"Ozone={oz}DB Aerosols={aer}")
                                a = Atmosphere(airmass=z, pressure=pressure, temperature=temperature)
                                transmission = a.simulate(ozone=oz, pwv=pwv, aerosols=aer)
                                transm = transmission(self.lambdas)
                                for i, lbda in enumerate(self.lambdas):
                                    atm = table.row
                                    atm['airmass'] = z
                                    atm['pressure'] = pressure
                                    atm['temperature'] = temperature
                                    atm['pwv'] = pwv
                                    atm['ozone'] = oz
                                    atm['aerosols'] = aer
                                    atm['wavelength'] = lbda
                                    atm['transmission'] = transm[i]
                                    atm.append()
                        table.flush()
        h5file.close()
        return table

    def compute(self):
        """Compute atmospheric transmissions and fill self.atmgrid.

        The wavelengths used for the computation are the ones set by parameters.LAMBDAS.

        Returns
        -------
        atmospheric_grid: array_like
            The atmospheric grid self.atmgrid.

        Examples
        --------
        >>> a = FullAtmosphereGrid(airmass_grid=[1], pressure_grid=[800, 850], temperature_grid=[10],
        ... pwv_grid=[5, 6], ozone_grid=[300, 400], aerosol_grid=[0.0, 0.1])
        >>> atmospheric_grid = a.compute()
        >>> atmospheric_grid
        >>> a.plot_transmission()
        """
        # first determine the length
        self.my_logger.warning(f'\n\tAtmosphere simulations for\n\t\tz={self.airmass_grid}'
                               f'\n\t\tP={self.pressure_grid}hPa\n\t\tT={self.temperature_grid}degC'
                               f'\n\t\tPWV={self.pwv_grid}mm\n\t\tOzone={self.ozone_grid}dobson'
                               f'\n\t\tAerosols={self.aerosols_grid}')
        count = 0
        from astropy.table import Table
        ncols = 8
        table = Table(names=("airmass", "pressure", "temperature", "pwv", "ozone", "aerosols", "lambdas","transmissions"), dtype=["f4"]*ncols)
        for z in self.airmass_grid:
            for pressure in self.pressure_grid:
                for temperature in self.temperature_grid:
                    for pwv in self.pwv_grid:
                        for oz in self.ozone_grid:
                            for aer in self.aerosols_grid:
                                self.my_logger.warning(f"\n\tz={z} P={pressure}hPa T={temperature} PWV={pwv}mm "
                                                       f"Ozone={oz}DB Aerosols={aer}")
                                a = Atmosphere(airmass=z, pressure=pressure, temperature=temperature)
                                transmission = a.simulate(ozone=oz, pwv=pwv, aerosols=aer)
                                transm = transmission(self.lambdas)
                                for i,lbda in enumerate(self.lambdas):
                                    table.add_row([z, pressure, temperature, pwv, oz, aer, lbda, transm[i]])
        table.write("full_atmospheric_grid.h5", path='atmospheres', append=True, compression=True)
        return table

    def load(self, file_name="./full_atmospheric_grid.h5"):
        """Load full atmospheric grid file.

        Examples
        --------
        >>> a = FullAtmosphereGrid()
        >>> a.load()

        """
        from astropy.table import Table
        self.my_logger.info(f'\n\tAtmosphere.load_image atm-file')
        table = Table.read("./full_atmospheric_grid.h5", format="hdf5", path="atmospheres")
        self.airmass_grid = np.unique(table['airmass'])
        self.pressure_grid = np.unique(table['pressure'])
        self.temperature_grid = np.unique(table['temperature'])
        self.pwv_grid = np.unique(table['pwv'])
        self.ozone_grid = np.unique(table['ozone'])
        self.aerosols_grid = np.unique(table['aerosols'])
        self.lambdas = np.unique(table['lambdas'])

        shape = (self.airmass_grid.size, self.pressure_grid.size, self.temperature_grid.size,
                 self.pwv_grid.size, self.ozone_grid.size, self.aerosols_grid.size, self.lambdas.size)
        self.atmgrid = np.array(table['transmissions']).reshape(shape)

        # interpolate the grid
        self.model = RegularGridInterpolator((self.airmass_grid, self.pressure_grid, self.temperature_grid,
                                              self.pwv_grid, self.ozone_grid, self.aerosols_grid, self.lambdas),
                                             self.atmgrid, bounds_error=False, fill_value=0)

    def plot_transmission(self):
        """Plot the atmospheric transmission contained in the grid.

        Examples
        --------
        >>> a = FullAtmosphereGrid()
        >>> a.load()
        >>> a.plot_transmission()
        """
        plt.figure()
        shape = np.copy(self.atmgrid.shape)
        shape = (np.prod(shape[:-1]).astype(int), shape[-1])
        data = self.atmgrid.reshape(shape)
        for count in range(int(shape[0])):
            label = f'{count}'
            plot_transmission_simple(plt.gca(), self.lambdas, data[int(count), :], title="Atmospheric grid",
                                     label=label)
        if parameters.DISPLAY:  # pragma: no cover
            plt.show()

    def plot_transmission_image(self):
        """Plot the atmospheric transmission contained in the grid using imshow.

        Examples
        --------
        >>> a = FullAtmosphereGrid()
        >>> a.load()
        >>> a.plot_transmission_image()
        """
        plt.figure()
        shape = np.copy(self.atmgrid.shape)
        shape = (np.prod(shape[:-1]), shape[-1])
        img = plt.imshow(self.atmgrid.reshape(shape), origin='lower', cmap='jet', aspect="auto")
        plt.grid(True)
        plt.xlabel(r"$\lambda$ [nm]")
        plt.ylabel("Simulation number")
        plt.title(" Atmospheric variations")
        cbar = plt.colorbar(img)
        cbar.set_label('Atmospheric transmission')
        if parameters.DISPLAY:
            plt.show()

    def simulate(self, airmass, pressure, temperature, ozone, pwv, aerosols):
        """Interpolate from the atmospheric grid to get the atmospheric transmission.

        Parameters
        ----------
        airmass: float
            Airmass of the source object.
        pressure: float
            Pressure of the atmosphere in hPa.
        temperature: float
            Temperature of the atmosphere in Celsius degrees.
        ozone: float
            Ozone quantity in Dobson
        pwv: float
            Precipitable Water Vapor quantity in mm
        aerosols: float
            VAOD Vertical Aerosols Optical Depth

        Examples
        --------
        >>> a = FullAtmosphereGrid()
        >>> a.load()
        >>> a.plot_transmission_image()
        >>> for pwv in np.arange(5):
        ...     transmission = a.simulate(airmass=1.2, pressure=815, temperature=11, ozone=400, pwv=pwv, aerosols=0.05)
        ...     plot_transmission_simple(plt.gca(), lambdas, transmission(lambdas),
        ...     title=a.title, label=a.label)
        >>> if parameters.DISPLAY: plt.show()
        """
        self.pwv = pwv
        self.ozone = ozone
        self.aerosols = aerosols
        self.set_title()
        self.set_label()
        ones = np.ones_like(self.lambdas)
        points = np.array([self.lambdas, ozone * ones, pwv * ones, aerosols * ones]).T
        atm = self.model(points)
        self.transmission = interp1d(self.lambdas, atm, kind='linear', bounds_error=False, fill_value=(0, 0))
        return self.transmission


if __name__ == "__main__":
    import doctest

    doctest.testmod()
