from astropy.coordinates import get_sun, get_body
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, solar_system_ephemeris
from astropy.time import Time
import datetime
import numpy as np
import polars as pl
from tqdm.autonotebook import tqdm
from great_tables import GT


class BadSun:
    summary_table_data = None
    fine = None
    coarse = None

    def __init__(
        self,
        lat,
        lon,
        azimuth,
        height=10,
        tz='US/Eastern',
        use_de430=True,
        tol=2,
        min_alt=0,
        max_alt=15,
    ):
        self.lat = lat
        self.use_de430 = use_de430
        self.lon = lon
        self.azimuth = azimuth
        self.loc = EarthLocation(lat=self.lat * u.deg, lon=self.lon * u.deg, height=height * u.m)
        self.fine = None
        self.tz = tz
        self.tol = tol
        self.low_az = self.azimuth - tol
        self.hi_az = self.azimuth + tol
        self.min_alt = min_alt
        self.max_alt = max_alt

    def get_sun(self, times):
        if self.use_de430:
            print('using de430')
            solar_system_ephemeris.set('de430')
            return get_body('sun', times)
        return get_sun(times)

    def coarse_run(self):
        start_time = Time(datetime.datetime.now())
        time_deltas = np.linspace(0, 365, 50000) * u.day
        times = start_time + time_deltas
        frames = AltAz(obstime=times, location=self.loc)
        alts = self.get_sun(times).transform_to(frames)
        low_az = self.azimuth - self.tol * 1.5
        hi_az = self.azimuth + self.tol * 1.5

        self.coarse = (
            alts.to_table()
            .to_pandas()
            .query('(@self.min_alt <= alt < @self.max_alt) and (@self.low_az < az < @self.hi_az)')
        )

    def high_res_run(self):
        if self.coarse is None:
            self.coarse_run()
        fine = []
        # self.coarse['day']
        for _day in tqdm(self.coarse['obstime']):
            day = Time(_day)
            time_deltas = np.linspace(-2, 2, 120) * u.hour
            times = day + time_deltas
            frames = AltAz(obstime=times, location=self.loc)
            alts = get_sun(times).transform_to(frames)
            fine.append(
                pl.from_pandas(
                    alts.to_table()
                    .to_pandas()
                    .query(
                        '(@self.min_alt <= alt < @self.max_alt) and (@self.low_az < az < @self.hi_az)'
                    )
                )
            )
        self.fine = pl.concat(fine)

    def make_summary_table(self):
        if self.fine is None:
            self.high_res_run()
        header_name = 'Bad Times'

        fmt2 = '%-I:%M%p'
        fine = self.fine
        fine = fine.with_columns(
            pl.col('obstime').dt.replace_time_zone('UTC').dt.convert_time_zone('US/Eastern')
        )
        self.summary_table_data = (
            fine.with_columns(Day=pl.col('obstime').dt.truncate('1d').dt.strftime('%Y %b %-d'))
            .group_by(['Day'])
            .agg(
                pl.col('obstime').min().dt.strftime(fmt2).alias('start_str'),
                pl.col('obstime').max().dt.strftime(fmt2).alias('end_str'),
                pl.col('alt').max().alias('Max Alt'),
                pl.col('alt').min().alias('Min Alt'),
                pl.col('obstime').min().alias('start'),
                pl.col('obstime').max().alias('end'),
            )
            .with_columns((pl.col('start_str') + ' -> ' + pl.col('end_str')).alias(header_name))
            .sort('start')
        )

    def summary_table(self, return_all=False):
        if self.summary_table_data is None:
            self.make_summary_table()
        header_name = 'Bad Times'
        cols = ['Day', header_name] if not return_all else self.summary_table_data.columns
        return self.summary_table_data.select(cols)

    def gt(self):
        """Make GT Table"""
        if self.summary_table_data is None:
            self.make_summary_table()
        data = self.summary_table_data.rename(
            {'start_str': 'Start', 'end_str': 'End', 'Min Alt': 'Min', 'Max Alt': 'Max'}
        ).select(['Day', 'Start', 'End', 'Min', 'Max'])
        degree_sign = '\N{DEGREE SIGN}'
        return (
            GT(data)
            .tab_header(
                title=f'Sun below {self.max_alt}{degree_sign}',
                subtitle=f'Direction between {self.azimuth-self.tol:.2f}{degree_sign} and {self.azimuth+self.tol:.2f}{degree_sign}',
            )
            .fmt_number(columns=['Min', 'Max'], decimals=2, use_seps=False)
            .tab_spanner('Times', columns=['Start', 'End'])
            .tab_spanner('Altitude', columns=['Min', 'Max'])
            .tab_source_note(
                source_note=f'Computed using astropy for Lat {self.lat:.3f}{degree_sign}, Lon: {self.lon:.3f}{degree_sign}'
            )
        )
