

This module uses `astropy` to compute when the sun is low on the horizon in a specific direction. I wrote it because recently I drove down a local road directly into the setting sun, and I wanted to know when during the evening that would happen and for how much longer.

However, it could be used for fun uses like finding a sunset that lines up with a particular building, statue, or monument.

**Install**

Clone this repository and then pip should work. Requirements are `astropy`, `pandas`, `polars` and `great_tables`.

`pip install .`

**Usage**

Here is simple example to make a table for [Manhattanhenge](https://www.amnh.org/research/hayden-planetarium/manhattanhenge), when the sun lines up with the east west grid of the city.

```{python}
from sunvisor import LowSun

lat =   40.757841
lon = -73.985253
#az = 238.79
az = 299.13
bs = LowSun(lat,lon,az,use_de430=True,tol=1,min_alt=-.75)


# a polars dataframe 

bs.summary_table()

# a great_table table
bs.gt()

```

** Possible future plans

* Make `great_tables` an optional requirement.
* A command line interface with options for generating CSV.
* Find moonrise/moonset times in addition to the sun.
