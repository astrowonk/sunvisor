import datetime

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import (
    AltAz,
    EarthLocation,
    SkyCoord,
    get_body,
    get_sun,
    solar_system_ephemeris,
)
from astropy.time import Time
from great_tables import GT
from tqdm.autonotebook import tqdm


def flatten_columns(self):
    """Monkey patchable function onto pandas dataframes to flatten multiindex column names from tuples. Especially useful
    with plotly.
    pd.DataFrame.flatten_columns = flatten_columns
    """
    df = self.copy()
    df.columns = [
        '_'.join([str(x) for x in [y for y in item if y]]) if not isinstance(item, str) else item
        for item in df.columns
    ]
    return df


pd.DataFrame.flatten_columns = flatten_columns


class LowSun:
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
        min_alt=-0.5,
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
            solar_system_ephemeris.set('de430')
            return get_body('sun', times)
        return get_sun(times)

    def coarse_run(self, days=365):
        start_time = Time(datetime.datetime.now())
        time_deltas = np.linspace(0, 365, days * 24 * 6) * u.day
        times = start_time + time_deltas
        frames = AltAz(obstime=times, location=self.loc)
        alts = self.get_sun(times).transform_to(frames)
        low_az = self.azimuth - self.tol * 1.5  # noqa: F841
        hi_az = self.azimuth + self.tol * 1.5  # noqa: F841

        self.coarse = (
            alts.to_table()
            .to_pandas()
            .query('(@self.min_alt <= alt < @self.max_alt) and (@self.low_az < az < @self.hi_az)')
        )
        self.coarse['_day'] = self.coarse['obstime'].dt.floor('1d')

    def high_res_run(self):
        if self.coarse is None:
            self.coarse_run()
        fine = []
        # self.coarse['day']
        for _day in tqdm(self.coarse.drop_duplicates(subset=['_day'])['obstime']):
            day = Time(_day)
            time_deltas = np.linspace(-1, 1, 60) * u.hour
            times = day + time_deltas
            frames = AltAz(obstime=times, location=self.loc)
            alts = self.get_sun(times).transform_to(frames)
            dfpandas = alts.to_table().to_pandas()
            # print(dfpandas['obstime'].agg(['min', 'max']))
            df = dfpandas.query(
                '(@self.min_alt <= alt < @self.max_alt) and (@self.low_az < az < @self.hi_az)'
            )

            #  print(pldf['obstime'].min())
            # print(pldf.select(pl.col('ob')))
            fine.append(df)
        self.fine = pd.concat(fine)

    def make_summary_table(self):
        """Use polars to make the summary table."""
        if self.fine is None:
            self.high_res_run()
        header_name = 'Times'

        fmt2 = '%-I:%M%p'
        fine = self.fine
        fine = fine.assign(
            obstime=lambda x: x['obstime'].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
        )
        fine['Day'] = fine['obstime'].dt.floor('1d')
        self.summary_table_data = (
            fine.groupby(['Day'])[['obstime', 'alt']].agg(['min', 'max']).flatten_columns()
        ).reset_index()
        self.summary_table_data['start_str'] = self.summary_table_data['obstime_min'].dt.strftime(
            fmt2
        )
        self.summary_table_data['end_str'] = self.summary_table_data['obstime_max'].dt.strftime(
            fmt2
        )
        self.summary_table_data['Times'] = (
            self.summary_table_data['start_str'] + ' -> ' + self.summary_table_data['end_str']
        )
        self.summary_table_data = self.summary_table_data.rename(
            {
                'alt_min': 'Min Alt',
                'alt_max': 'Max Alt',
            },
            axis=1,
        )

    def summary_table(self, return_all=False):
        """Return the summary table"""
        if self.summary_table_data is None:
            self.make_summary_table()
        header_name = 'Times'
        cols = ['Day', header_name] if not return_all else self.summary_table_data.columns
        return self.summary_table_data[cols]

    def gt(self):
        """Make GT Table"""
        if self.summary_table_data is None:
            self.make_summary_table()
        data = self.summary_table_data.rename(
            columns={'start_str': 'Start', 'end_str': 'End', 'Min Alt': 'Min', 'Max Alt': 'Max'}
        )[['Day', 'Start', 'End', 'Min', 'Max']]
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
                source_note=f'Computed using astropy for Lat: {self.lat:.3f}{degree_sign}, Lon: {self.lon:.3f}{degree_sign}'
            )
        )
