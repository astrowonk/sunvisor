

This module uses `astropy` to compute when the sun is low on the horizon in a specific direction. I wrote it because the once the sun was very low and right in my eyes when driving down a local road, and wanted to know when during the evening that would happen.

However, it could be used for more fun things like finding a sunset that lines up with a particular monument.

USAGE:

Here is simple example to make a table for [Manhattanhenge](https://www.amnh.org/research/hayden-planetarium/manhattanhenge), when the sun lines up with the east west grid of the city.

```{python}
lat =   40.757841
lon = -73.985253
#az = 238.79
az = 299.13
bs = BadSun(lat,lon,az,use_de430=True,tol=1,min_alt=-.75)


# a polars dataframe 

bs.summary_table()

# a great_table table
bs.gt()

```

