from astropy.coordinates import get_sun
import astropy.units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
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

    def __init__(self, lat, lon, azimuth, height=10, tz='US/Eastern'):
        self.lat = lat
        self.lon = lon
        self.azimuth = azimuth
        self.loc = EarthLocation(lat=self.lat * u.deg, lon=self.lon * u.deg, height=height * u.m)
        self.fine = None
        self.tz = tz

    def coarse_run(self):
        start_time = Time(datetime.datetime.now())
        time_deltas = np.linspace(0, 365, 50000) * u.day
        times = start_time + time_deltas
        frames = AltAz(obstime=times, location=self.loc)
        alts = get_sun(times).transform_to(frames)
        low_az = self.azimuth - 3  # noqa: F841
        hi_az = self.azimuth + 3  # noqa: F841
        self.coarse = (
            alts.to_table().to_pandas().query('(0 < alt < 15) and (@low_az < az < @hi_az)')
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
            low_az = self.azimuth - 2  # noqa: F841
            hi_az = self.azimuth + 2  # noqa: F841
            fine.append(
                pl.from_pandas(
                    alts.to_table()
                    .to_pandas()
                    .query('(-.01 <= alt < 15) and (@low_az < az < @hi_az)')
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
        return (
            GT(data)
            .fmt_number(columns=['Min', 'Max'], decimals=2, use_seps=False)
            .tab_spanner('Sun in Eyes', columns=['Start', 'End'])
            .tab_spanner('Altitude', columns=['Min', 'Max'])
        )
