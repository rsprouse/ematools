import sys
from abc import ABC, abstractproperty, abstractmethod
import numpy as np
import pandas as pd
import rowan

def rotation_ref_creator(tsvname, tsvcolmap, *args, **kwargs):
    '''Return RotationRef object of correct type.'''
    try:
        sensors = list(tsvcolmap.keys())  # Input is a dict
    except AttributeError:
        sensors = tsvcolmap    # Already a list.
    params = list(kwargs.keys())
    refname = kwargs['nasion'] if 'nasion' in params else 'REF'
    rmaname = kwargs['right_mastoid'] if 'right_mastoid' in params else 'RMA'
    lmaname = kwargs['left_mastoid'] if 'left_mastoid' in params else 'LMA'
    if refname in sensors and rmaname in sensors and lmaname in sensors:
        rotref = WaxBiteplate3Point(tsvname, tsvcolmap, *args, **kwargs)
    elif not (refname in sensors or rmaname in sensors or lmaname in sensors):
        rotref = WaxBiteplateReferenced(tsvname, tsvcolmap, *args, **kwargs)
    else:
        msg = 'Could not return correct RotationRef based on input sensors. ' \
            'Sensors should include all of \'REF\', \'RMA\', \'LMA\' or none.\n'
        raise TypeError(msg)
    return rotref

def coords2df(coords, sec, sensors):
    '''Convert an ndarray of coordinates to a dataframe.

    Parameters
    ----------
    coords : 3D ndarray
        The coordinate data with axes time, sensors, xyz

    sec : list-like
        The labels for the first dimension of `coords`
        (time in seconds). The length must match `coords.shape[0]`.

    sensors : list-like
        The labels for the second dimension of `coords`
        (sensors). The length must match `coords.shape[1]`.

    Returns
    -------
    df : DataFrame
        The coordinate data as a DataFrame, one row per time
        frame and columns of <sensor>_<xyz>. The first column
        is named 'sec' and contains the labels from input
        parameter`sec`. The remaining columns are groups of
       '<sensor>_x', '<sensor>_y', '<sensor>_z', e.g.
       'REF_x', 'REF_y', 'REF_z', 'OS_x', 'OS_y', 'OS_z'.
    '''
    d = coords.reshape(len(sec), len(sensors) * 3)
    names = []
    for s in sensors:
        names.extend([s + '_x', s + '_y', s + '_z'])
    df = pd.DataFrame(d, columns=names)
    df.insert(0, 'sec', sec)
    return df

class NDIData(object):
    '''
    A representation of NDI Wave tsv data.

    Instantiate the NDIData object with the path to a .tsv file and mapping of
    sensor names to channel index in the .tsv file. The .tsv file is expected to
    have one (time) column at the left, followed by repetitions of sets
    of columns consisting of a 'State' column (and possibly others, e.g.
    channel name and frame identifier) followed by columns
    Q0, Qx, Qy, Qz, Tx, Ty, Tz in exactly that order. These repetitions are
    expected to occur regularly with no exception, other than the possibility
    of empty columns identified in the header with ' ', which are ignored. The
    Q0...Tz columns may have suffixes appended to them, either in the .tsv file
    or as a side effect of `read_csv`.

    The sensor map is a dictionary or list. If a dict, it is of the form
    {'sensor_name': idx, ...}, with the index 0 indicating the first repetition
    of Q0...Tz columns and index 1 for the second repetition. If a list, it is
    a list of sensors in the order they appear in the .tsv file.

    The 'State' values are processed to replace Q0...Tz values with
    `Nan` for any sensor-frame that is not 'OK'. Replacement occurs
    automatically when the `NDIData` object is instantiated.

    Access to the .tsv data is available as a dataframe via the `df` attribute:

    ```
    tsv = NDIData(tsvpath, tsvcolmap)
    tsv.df
    ```

    Convenience methods are provided to access the Q0...Tz columns for all
    sensors or a set of sensors as a 3d numpy ndarray, with dimensions 0)
    frame index (time); 1) sensors; 2) Q0...Tz. Use `qtvals` to return
    columns [`Q0`, `Qx`, `Qy`, `Qz`, `Tx`, `Ty`, `Tz`] for a given set of
    sensors. Use `qvals` to return only [`Q0`, `Qx`, `Qy`, `Qz`], and use
    `coords` to return only [`Tx`, `Ty`, `Tz`] for the given sensors.

    ```
    # Return 3d ndarray (axes: time, sensor, Q0...Tz)
    tsv.qtvals()           # Return Q0, Qx, Qy, Qz, Tx, Ty, Tz for all sensors
    tsv.qvals(['nasion', 'leftMastoid']) # Return Q0, Qx, Qy, Qz for two sensors
    tsv.coords(['nasion', 'leftMastoid']) # Return Tx, Ty, Tz for two sensors
    '''

    def _inspect_first_line(self, tsvname):
        with open(tsvname, 'r') as fh:
            hd = np.array(fh.readline().split('\t'))
            if hd[0] == 'Wav Time':  # Has a header.
                skiprows = 1
            else:
                skiprows = 0  # No header line.
        return (hd, skiprows)

    def __init__(self, tsvname, tsvcolmap=None, load_all=False):
        ''''''
        subcolumns = ["id","frame","state","q0","qx","qy","qz","x","y","z"]
        cols_per_sensor = len(subcolumns)  # Number of columns for each sensor
        cols_at_left = 1      # Number of columns at left, before sensor columns
        self._rate = None
        firstline, skiprows = self._inspect_first_line(tsvname)
        num_empty = len(np.argwhere(firstline == ' '))
        # TODO: tsvcolmap processing could probably be cleaned up
        if tsvcolmap is None or load_all is True:
            num_sensorcols = len(firstline) - cols_at_left - num_empty
            assert(num_sensorcols % cols_per_sensor == 0)
            num_sensors = np.int(num_sensorcols / cols_per_sensor)
            self.tsvcolmap = {
                'S{:d}'.format(n): n for n in np.arange(num_sensors)
            }
        if isinstance(tsvcolmap, list):
            tsvcolmap = {
                s: idx for idx, s in enumerate(tsvcolmap) if s is not None
            }
        if tsvcolmap is not None and load_all is False:
            self.tsvcolmap = tsvcolmap
        elif tsvcolmap is not None and load_all is True:  # Merge user-provided columns into all sensor columns
            tsvdf = pd.DataFrame(
                list(self.tsvcolmap.items()), columns=['sensor', 'idx']) \
                .set_index('idx')
            userdf = pd.DataFrame(
                list(tsvcolmap.items()), columns=['sensor', 'idx']) \
                .set_index('idx')
            tsvdf.update(userdf)
            self.tsvcolmap = dict(zip(tsvdf.sensor, tsvdf.index))

        self.sensors = sorted(self.tsvcolmap, key=lambda x: self.tsvcolmap[x])
        usecols = list(np.arange(cols_at_left))
        better_head = ['sec']
        for s in self.sensors:
            start = (self.tsvcolmap[s] * cols_per_sensor) + cols_at_left
            usecols.extend(np.arange(start, start + cols_per_sensor))
            better_head.extend(['{}_{}'.format(s, c) for c in subcolumns])
        usecols = np.array(usecols)

        # Find empty columns of only a single space.
        for idx in np.argwhere(firstline == ' '):
            usecols[usecols >= np.squeeze(idx)] += 1
        self.df = pd.read_csv(
            tsvname,
            sep='\t',
            usecols=usecols,
            header=None,          # The last three parameters
            skiprows=skiprows,    # are used to override
            names=better_head     # the existing file header.
        )
        self._set_bad_to_nan()
        self.df[self.state_cols] = self.df[self.state_cols].astype('category')

    @property
    def rate(self):
        '''The data sample rate.'''
        if self._rate is None:
            self._rate = 1.0 / np.mean(np.diff(self.df.sec))
        return self._rate

    @property
    def num_sensors(self):
        '''The number of sensors.'''
        return len(self.sensors)

    @property
    def state_cols(self):
        '''The <sensor>_state column names.'''
        return [c for c in self.df.columns if c.endswith('_state')]

    def check_for_5d(self):
        '''Check data for whether data values indicate 5d sensor.'''
        # All qz columns are expected to be 0.0 (or NaN) if 5d
        qzcols = self.df.columns.str.endswith('_qz')
        return np.nansum(self.df.loc[:,qzcols].values) == 0.0

    def _set_bad_to_nan(self):
        '''Set values of Q/T columns to NaN if corresponding 'State' column
        is not 'OK'.'''
        bad = self.df.loc[:,self.df.columns.str.match('.+_[Ss]tate$')] != 'OK'
        qtvals = self.qtvals()[0]
        qtvals[bad] = np.nan  # bad should broadcast along Q0Txyz dimension
        self.replace_columns(qtvals, qt='QT')

    def replace_columns(self, vals, qt='T', sensors=None):
        '''Replace column data in self.df with new values from an ndarray.

        Parameters
        ----------

        vals: 2D or 3D ndarray
            New values for replacement. If 3D, `vals` has axes (time, sensors, Q0...Tz)
            and will be reshaped to 2D to match the dataframe column arrangement; the
            latter two dimensions are collapsed. An input 2D array must have the same
            axes arrangement as the reshaped 3D array.

        qt: str ('QT', 'Q', or 'T')
            The kinds of new column data in `vals`. Use 'QT' if replacing all 'Q' and
            'T' values. Use 'Q' if replacing only 'Q' values and 'T' if only 'T' values.
            If `vals` is 3D, then the shape of second dimension must match the `qt`
            value: {'QT': 7, 'Q': 4, 'T': 3}.

        sensors: list of str
            List of sensor values in `vals`. For 3D `vals` the third axis must be the
            same length as `sensors`.

        Returns
        -------
        Returns True on success. Raises an error if assignment does not succeed.

        '''
        try:
            assert(len(vals) == len(self.df))
        except AssertionError:
            msg = "Can't replace columns. New values are not equal in length to old.\n"
            raise ValueError(msg)
        if len(vals.shape) == 3:
            vals = vals.reshape(len(vals), -1)
        if sensors is None:
            sensors = self.sensors
        try:
            self.df.loc[:, self.qtcols(qt=qt, sensors=sensors)] = vals
        except:
            msg = "Could not replace columns with new data.\n"
            raise ValueError(msg)
        return True

    def sensor_mean_coords(self, sensor):
        '''Return the mean x, y, z for sensor, excluding NaN.'''
        return np.nanmean(np.squeeze(self.coords(sensor)[0]), axis=0)

    def _qt_getter(self, qt, sensors=None, start=0.0, end=np.Inf):
        '''Get Q and/or T values. To be called by qtvals(), qvals(), coords().'''
        qtlen = {'QT': 7, 'Q': 4, 'T': 3}
        if sensors is None:
            sensors = self.sensors
        elif isinstance(sensors, str):
            sensors = [sensors]
        d = self.df.loc[
            (self.df.sec >= start) & (self.df.sec <= end),
            self.qtcols(qt=qt, sensors=sensors)
        ]
        # Return tuple of:
        # 0. The data values.
        # 1. Time in seconds to match the first dimension of the values.
        # 2. The list of sensor labels for the second dimension of the values.
        return (
            d.values.reshape(len(d), len(sensors), qtlen[qt]),
            self.df.sec[(self.df.sec >= start) & (self.df.sec <= end)],
            sensors
        )

    def qtcols(self, qt, sensors=None):
        '''Get names of Q and/or T columns for given sensors.'''
        suffixes = {
            'Q': ['q0', 'qx', 'qy', 'qz'],
            'T': ['x', 'y', 'z'],
            'QT': ['q0', 'qx', 'qy', 'qz', 'x', 'y', 'z']
        }
        if sensors is None:
            sensors = self.sensors
        cols = ['{}_{}'.format(s, sfx) for s in sensors for sfx in suffixes[qt]]
        return cols

    def qtvals(self, sensors=None, start=0.0, end=np.Inf):
        '''Return the Q0, Qx, Qy, Qz, Tx, Ty, Tz values for given sensors
        as a 3d ndarray. If sensors is None, return all sensors.

        The dimensions are:
            frame index (sec)
            sensors
            0xyz,xyz coordinates
        '''
        return self._qt_getter('QT', sensors, start, end)

    def qvals(self, sensors=None, start=0.0, end=np.Inf):
        '''Return the Q0, Qx, Qy, Qz values for given sensors as a 3d ndarray.
        If sensors is None, return all sensors.

        The dimensions are:
            frame index (sec)
            sensors
            0xyz values
        '''
        return self._qt_getter('Q', sensors, start, end)

    def coords(self, sensors=None, start=0.0, end=np.Inf):
        '''Return the x, y, z values for given sensors as a 3d ndarray.
        If sensors is None, return all sensors.

        The dimensions are:
            frame index (sec)
            sensors
            xyz coordinates
        '''
        return self._qt_getter('T', sensors, start, end)

    def time_range_as_int_index(self, start=None, end=None):
        '''Return the 0-based integer indexes of the rows in self.df where the time column
        is in the range specified by `start` and `end`.

        Parameters
        ----------

        start: numeric
            The start time for the returned range. Integer indexes for times greater than or
            equal to `start` are returned. If `start` is not specified, 0.0 is the default.

        end: numeric
            The end time for the returned range. Integer indexes for times less than or
            equal to `end` are returned. If `end` is not specified, np.Inf is the default.
        '''
        if start is None:
            start = 0.0
        if end is None:
            end = np.Inf
        return np.flatnonzero(
            (self.df['sec'] >= start) & (self.df['sec'] <= end)
        )

    def time_range(self, start=None, end=None):
        '''Return the time values for the range specified by `start` and `end`.'''
        tidx = self.time_range_as_int_index(start=start, end=end)
        return self.df['sec'].iloc[tidx].values

class RotationRef(ABC):
    '''A mixin for reference recordings used to rotate to ideal head position.'''

    @abstractproperty
    def has_global_ref(self):
        '''Return True if reference sensor is a 6 Degrees Of Freedom global
        reference for other sensor data.
        '''
        pass

    @abstractmethod
    def transform(self, coords):
        '''Transform (rotate and translate) a 3d set of sensor points into the
        reference space.'''
        pass

# TODO: parameter and attribute names are not necessarily sensible and meaningful.
class WaxBiteplate3Point(NDIData, RotationRef):
    '''A class for wax biteplate recordings using the 5DOF nasion sensor.'''
    _origin = None
    _nasion = None
    _right = None
    _left = None
    _head_loc = None

    def __init__(self, tsvname, tsvcolmap, origin_correction=(0.0,0.0,0.0),
            nasion='REF', right_mastoid='RMA', left_mastoid='LMA',
            origin='OS', molar='MS', *args, **kwargs):
        super(WaxBiteplate3Point, self).__init__(tsvname, tsvcolmap, **kwargs)
        self._origin_correction = origin_correction
        self._nasion = nasion
        self._right = right_mastoid
        self._left = left_mastoid
        self._origin = origin
        self._molar = molar

    @property
    def has_global_ref(self):
        return False

    @property
    def fixed_sensors(self):
        return [self._nasion, self._right, self._left]

    @property
    def fixed_ref(self):
        '''
        Return the idealized xyz coordinates of the fixed head sensors
        transformed into the reference space.
        '''
        if self._head_loc is None:
            # 1) start by translating the space so OS is at the origin
            sensors = [self._nasion, self._right, self._left, self._molar]
            ref_t, rma_t, lma_t, ms_t = np.squeeze(
                np.vsplit(
                    self.translated_sensors(sensors),
                    len(sensors)
                )
            )

            # 2) now find the rotation matrix to the occlusal coordinate system
            z = np.cross(ref_t, ms_t)  # z is perpendicular to ms and ref vectors
            z = z / np.linalg.norm(z)

            y = np.cross(ms_t, z)        # y is perpendicular to z and ms
            y = y / np.linalg.norm(y)

            x = np.cross(y, z)
            x = x / np.linalg.norm(x)

            m = np.array([x, y, z])    # rotation matrix directly

            # 3) now rotate the mastoid points - using the rotation matrix
            ref_t = np.dot(ref_t, m.T)
            rma_t = np.dot(rma_t, m.T)
            lma_t = np.dot(lma_t, m.T)

            self._head_loc = np.vstack([ref_t, rma_t, lma_t]) + self.origin_correction
        return self._head_loc

    def translated_sensors(self, sensors):
        '''Return sensor data translated so that the origin sensor is at [0.0, 0.0, 0.0].'''
        return np.nanmean(self.coords(sensors)[0], axis=0) + self.translation

    @property
    def translation(self):
        return -(self.sensor_mean_coords(self._origin))

    def transform(self, coords, sensor_labels):
        sensor_order = {s: idx for idx, s in enumerate(sensor_labels)}
        ref_cols = [
            sensor_order[self._nasion],
            sensor_order[self._right],
            sensor_order[self._left]
        ]
        if len(coords.shape) == 2:
            coords = np.expand_dims(coords, axis=0)
        for n in np.arange(coords.shape[0]):
            try:
                q, t = rowan.mapping.davenport(
                    coords[n, ref_cols, :],
                    self.fixed_ref
                )
                coords[n,:,:] = rowan.rotate(q, coords[n,:,:]) + t
            except Exception as e:
                coords[n,:,:] = np.nan
        return coords

class WaxBiteplateReferenced(NDIData, RotationRef):
    '''A class for wax biteplate recordings using that use the 6DOF nasion
    sensor as a global reference.
    '''
    def __init__(self, tsvname, tsvcolmap, origin_correction=(0.0,0.0,0.0),
            origin='OS', molar='MS', *args, **kwargs):
        super(WaxBiteplateReferenced, self).__init__(
            tsvname, tsvcolmap, **kwargs
        )
        self._origin_correction = origin_correction
        self._origin = origin
        self._molar = molar
        self._quaternion = None

    @property
    def has_global_ref(self):
        return True

    @property
    def ref_quaternion(self):
        '''Return quaternion for rotating data into occlusal plane.'''
        if self._quaternion is None:
            # 1) start by translating the space so origin sensor is at the origin
            REF = self.ref_translation
            MS = self.sensor_mean_coords(self._molar) + self.ref_translation

            # 2) now find the rotation matrix to the occlusal coordinate system
            z = np.cross(REF, MS)  # z is perpendicular to ms and ref vectors
            z = z / np.linalg.norm(z)

            y = np.cross(MS, z)        # y is perpendicular to z and ms
            y = y / np.linalg.norm(y)

            x = np.cross(y, z)
            x = x / np.linalg.norm(x)

            m = np.array([x, y, z])    # rotation matrix directly
            self._quaternion = rowan.from_matrix(m)
        return self._quaternion

    @property
    def ref_translation(self):
        return -(self.sensor_mean_coords(self._origin)) + self.origin_correction

    def transform(self, coords):
        if len(coords.shape) == 2:
            coords = np.expand_dims(coords, axis=0)
        return rowan.rotate(self.ref_quaternion, coords) + self.ref_translation

