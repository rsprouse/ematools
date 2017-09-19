# ematools
Python tools for working with EMA data.

# Install

    pip install git+git://github.com/rsprouse/ematools

# Usage

```python
from ematools import EmaUCSFDataLoader as DataLoader

datadir = '/path/to/subject_data'
loader = DataLoader(datadir)

speaker_id = '7'
dataname = 'MOCHA_TIMIT'
utterance = '013'
channel = 0
rate, audio = loader.get_audio(speakerid, dataname, utterance, channel)
df = loader.get_speaker_utt(speakerid, dataname, utterance)
```

If you don't want all of the data columns, you can exclude particular sensor
prefixes:

```python
excludelist=['REF', 'UNK', 'EMPTY']
df = loader.get_speaker_utt(
    speakerid,
    dataname,
    utterance,
    drop_prefixes=excludelist
)
```

The DataFrame returned by `get_speaker_utt()` is expected to contain a time
column named `sec` (if the `.ndi` file contains a column named `time` it
will be renamed `sec`).

Additional expected columns are named by the pattern `<sensor>_<meas>`, where
`<sensor>` is the name of an EMA sensor, and `<meas>` is a sensor measurement,
one of `x`, `y`, `z`, `q0`, `qx`, `qy`, `qz`. There may also be other
columns of the form `<sensor>_<name>`, where `<name>` can be any arbitrary
name.

# Adding derivative columns

You might be interested in adding new columns to your DataFrame that are
calculated from existing columns. To start you can find the names of the
columns that contain coordinate measures with `get_coordcols()` or
quaternion values with `get_quatcols()`:

```python
# returns e.g. ['UL_x', 'UL_y', 'UL_z']
coordcols = loader.get_coordcols(df)

# returns e.g. ['UL_q0', 'UL_qx', 'UL_qy', 'UL_qz']
quatcols = loader.get_quatcols(df)
```

With the list of coordinate columns in hand you can perform a calculation
on each and join the results to your DataFrame using the original column
names plus a suffix:

```python
# E.g. calculate first differentials and add as columns
# ['UL_x_vel', 'UL_y_vel', 'UL_z_vel']
df = df.join(df[coordcols].diff(), rsuffix='_vel')
```
 
