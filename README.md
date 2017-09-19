# ematools
Python tools for working with EMA data.

## Install

    pip install git+git://github.com/rsprouse/ematools

## Usage

```python
from ematools.data_loader import EmaUcsfDataLoader as DataLoader
import wavio

datadir = '/path/to/subject_data'
loader = DataLoader(datadir)

# Utterance-specific values.
speaker_id = '7'
dataname = 'MOCHA_TIMIT'
sentnum = '013'
audiochan = 0

# Get absolute path to an utterance without extension.
fpath = loader.get_utterance_path(speakerid, dataname, sentnum)

w = wavio.read('{}.wav'.format(fpath))
aurate = w.rate
audio = w.data[:, audiochan])
 
df = loader.get_utterance_df('{}.ndi'.format(fpath))
```

If you don't want all of the data columns, you can exclude particular sensor
measurements with `drop_prefixes`:

```python
df = loader.get_utterance_df(
    '{}.ndi'.format(fpath),
    drop_prefixes=['REF', 'UNK', 'EMPTY']
)
```

The DataFrame returned by `get_utterance_df()` is expected to contain a time
column named `time` (if the `.ndi` file contains a column named `time` it
will be renamed `sec`).

Additional expected columns are named by the pattern `<sensor>_<meas>`, where
`<sensor>` is the name of an EMA sensor, and `<meas>` is a sensor measurement,
one of `x`, `y`, `z`, `q0`, `qx`, `qy`, `qz`. There may also be other
columns of the form `<sensor>_<name>`, where `<name>` can be any arbitrary
name.

## Adding derivative columns

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

# Same as above, with some smoothing
df = df.join(df[coordcols].rolling(3, center=True).diff(), rsuffix='_vel')
```
 
## Subject metadata

Per-subject metadata is stored in the file `SN<subj_num>_metadata.yaml` in
the subject folder. The yaml file instantiates a Python dict. The `paltrace`
key contains a list of dicts containing information regarding palate trace
recordings. The first dict in the list is regarded as the best palate
trace recording and ideally is rotation-corrected data that is constrained
to the time range of the palate trace.

An example yaml file is shown below. It describes two palate trace files.
The .tsv file in the list is an original recording in which the
palate trace occurs between 8.3 and 17.6 seconds. The sensor that traced
the palate is labelled 'PL', and the mapping of data columns to real-world
dimensions is 'zxy'. The .ndi file contains corrected data limited to the
palate trace times. The .ndi file is listed first since it is the preferred
one.

```python
paltrace:
- filename: SN4_Palate_times.ndi
  sensor: PL
  rotated: true
  start_trace: null
  end_trace: null
  xyz: xyz
- filename: SN4_Palate.tsv
  sensor: PL
  rotated: false
  start_trace: 8.3
  end_trace: 17.6
  xyz: zxy
```
