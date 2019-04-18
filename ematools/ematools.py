import numpy as np
import rowan

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
    
    By default the 'State' values are processed to replace Q0...Tz values with
    `Nan` for any sensor-frame that is not 'OK'. Replacement occurs
    automatically when the `NDIData` object is instantiated unless
    `replace_bad` is set to `False`.
    
    Access to the .tsv data is available as a dataframe via the `df` attribute:
    
    ```
    tsv = NDIData(tsvpath, colmap)
    tsv.df
    ```
    
    Convenience methods are provided to access the Q0...Tz columns for all
    sensors or a set of sensors as a 3d numpy ndarray, with dimensions 0)
    frame index (time); 1) sensors; 2) Q0...Tz. Use `qtvals` to return
    columns [`Q0`, `Qx`, `Qy`, `Qz`, `Tx`, `Ty`, `Tz`] for a given set of
    sensors. Use `qvals` to return only [`Q0`, `Qx`, `Qy`, `Qz`], and use
    `tvals` to return only [`Tx`, `Ty`, `Tz`] for the given sensors.
    
    ```
    # Return 3d ndarray (axes: time, sensor, Q0...Tz)
    tsv.qtvals()           # Return Q0, Qx, Qy, Qz, Tx, Ty, Tz for all sensors
    tsv.qvals(['nasion', 'leftMastoid']) # Return Q0, Qx, Qy, Qz for two sensors
    tsv.tvals(['nasion', 'leftMastoid']) # Return Tx, Ty, Tz for two sensors
    '''

    def __init__(self, tsvname, colmap, replace_bad=True, time_col='Wav Time'):
        ''''''
        self._sensors = None
        self.colmap = {s: idx for idx, s in enumerate(colmap) if s is not None}
        cols_per_sensor = 10  # Number of columns for each sensor
        cols_at_left = 1      # Number of columns at left, before sensor columns
        usecols = list(np.arange(cols_at_left))
        for s in self.sensors:
            start = (self.colmap[s] * cols_per_sensor) + cols_at_left
            usecols.extend(np.arange(start, start + cols_per_sensor))
        usecols = np.array(usecols)
        # Find empty columns, which have a whitespace-only header.
        with open(tsvname, 'r') as fh:
            hd = np.array(fh.readline().split('\t'))
            for idx in np.argwhere(hd == ' '):
                usecols[usecols >= np.squeeze(idx)] += 1
        self.df = pd.read_csv(
            tsvname,
            sep='\t',
            usecols=usecols
        )
        self.time_col = time_col
        if replace_bad is True:
            self._set_bad_to_nan()
        # TODO: relabel columns with sensor names?

    @property
    def sensors(self):
        '''Ordered list of sensors in the dataframe.'''
        if self._sensors is None:
            items = list(self.colmap.items())
            items.sort(key=lambda x: x[1])
            self._sensors = [i[0] for i in items]
        return self._sensors

    def _set_bad_to_nan(self):
        '''Set values of Q/T columns to NaN if corresponding 'State' column
        is not 'OK'.'''
        bad = self.df.loc[:,self.df.columns.str.match('[Ss]tate')] != 'OK'
        qtvals = self.qtvals()
        qtvals[bad] = np.nan  # bad should broadcast along Q0Txyz dimension
        self.replace_columns(qtvals, qt='QT')
    
    def qtindexes(self, qt, sensors=None):
        '''
        Return a list of column integer indexes in self.df for the
        Q/T columns for given sensors. The Q/T columns for a given sensor
        are assumed to be in the order Q0, Qx, Qy, Qz, Tx, Ty, Tz with no
        other columns intervening.
            
        This should be robust whether the Q/T values have additional
        suffixes or not, e.g. Q0, Q0.1, Q0.15, etc.
            
        The value of sensors is a list of the elements found in self.sensors.
        If sensors is None, include all sensors.
        '''

        # Get indexes of all Q0/Tx columns.
        qtmap = {'Q': 'Q0', 'T': 'Tx', 'QT': 'Q0'}
        q0s = (
            self.df.columns.str.match('^{:}'.format(qtmap[qt]))
        ).nonzero()[0]

        # Test our assumption that Q/T0 channels are equally-spaced.
        try:
            assert(np.diff(q0s).std() == 0)
        except AssertionError:
            msg = 'Q/T channels are not equally-spaced!\n'
            raise RuntimeError(msg)

        # Calculate the indexes of the selected Q0/Tx columns.
        step = int(np.diff(q0s)[0])
        # First get sensor indexes, e.g. in range 0-15.
        if sensors is None:  # Use all sensors.
            snums = np.arange(len(q0s))
        else:
            snums = np.array(
                [idx for idx, s in enumerate(self.sensors) if s in sensors]
            )
        # Multiply by step and add column offset from left.
        snums = (snums * step) + q0s[0]
        # Add the x,y,z column indexes and flatten the list.
        if qt == 'Q':
            qnums = [(n, n+1, n+2, n+3) for n in snums]
        elif qt == 'T':
            qnums = [(n, n+1, n+2) for n in snums]
        elif qt == 'QT':
            qnums = [(n, n+1, n+2, n+3, n+4, n+5, n+6) for n in snums]
        flatnums = [item for sublist in qnums for item in sublist]
        return flatnums

    def replace_columns(self, vals, qt, sensors=None):
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
        try:
            self.df.iloc[:, self.qtindexes(qt=qt, sensors=sensors)] = vals
        except:
            msg = "Could not replace columns with new data.\n"
            raise ValueError(msg)
        return True

    def sensor_mean_tvals(self, sensor):
        '''Return the mean x, y, z for sensor, excluding NaN.'''
        return np.nanmean(np.squeeze(self.tvals(sensor)), axis=0)

    def _qt_getter(self, qt, sensors=None, start=None, end=None):
        '''Get Q and/or T values. To be called by qtvals(), qvals(), tvals().'''
        qtlen = {'QT': 7, 'Q': 4, 'T': 3}
        tidx = self.time_range_as_int_index(start, end)
        if isinstance(sensors, str):
            sensors = [sensors]
        d = self.df.iloc[tidx, self.qtindexes(qt, sensors)]
        return d.values.reshape(len(tidx), -1, qtlen[qt])
        
    def qtvals(self, sensors=None, start=None, end=None):
        '''Return the Q0, Qx, Qy, Qz, Tx, Ty, Tz values for given sensors
        as a 3d ndarray. If sensors is None, return all sensors.
        
        The dimensions are:
            frame index (time)
            sensors
            0xyz,xyz coordinates
        '''
        return self._qt_getter('QT', sensors, start, end)

    def qvals(self, sensors=None, start=None, end=None):
        '''Return the Q0, Qx, Qy, Qz values for given sensors as a 3d ndarray.
        If sensors is None, return all sensors.
        
        The dimensions are:
            frame index (time)
            sensors
            0xyz values
        '''
        return self._qt_getter('Q', sensors, start, end)

    def tvals(self, sensors=None, start=None, end=None, fixed_ref=None,
        fixed_sensors=None):
        '''Return the Tx, Ty, Tz values for given sensors as a 3d ndarray.
        If sensors is None, return all sensors.
        
        The dimensions are:
            frame index (time)
            sensors
            xyz coordinates
        '''
        tvals = self._qt_getter('T', sensors, start, end)
        if fixed_ref is not None:
            hdvals = self._qt_getter('T', fixed_sensors, start, end)
            for n in np.arange(tvals.shape[0]):
                try:
                    q, t = rowan.mapping.davenport(hdvals[n,:3,:], fixed_ref)
                    tvals[n,:,:] = rowan.rotate(q, tvals[n,:,:]) + t
                except Exception as e:
                    pass
        return tvals

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
            (self.df[self.time_col] >= start) & (self.df[self.time_col] <= end)
        )

    def time_range(self, start=None, end=None):
        '''Return the time values for the range specified by `start` and `end`.'''
        tidx = self.time_range_as_int_index(start=start, end=end)
        return self.df[self.time_col].iloc[tidx].values

class BiteplateRec(ABC):
    '''A mixin for biteplate reference recordings.'''
    _origin = None
    _center = None
    _right = None
    _left = None
    _head_loc = None

    @abstractproperty
    def translation(self):
        '''
        Get x, y, z of translation that moves a defined point or sensor to
        the origin.
        '''
        pass

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
            z = np.cross(ms_t, ref_t)  # z is perpendicular to ms and ref vectors
            z = z / np.linalg.norm(z)

            y = np.cross(z, ms_t)        # y is perpendicular to z and ms
            y = y / np.linalg.norm(y)

            x = np.cross(z, y)
            x = x / np.linalg.norm(x)

            m = np.array([x, y, z])    # rotation matrix directly

            # 3) now rotate the mastoid points - using the rotation matrix
            rma_t = np.dot(rma_t, m.T) 
            lma_t = np.dot(lma_t, m.T)

            self._head_loc = np.vstack([rma_t, lma_t, ref_t])
        return self._head_loc

    def translated_sensors(self, sensors):
        '''Return sensor data translated so that the origin sensor is at [0.0, 0.0, 0.0].'''
        return np.nanmean(self.tvals(sensors), axis=0) + self.translation

# TODO: parameter and attribute names are not necessarily sensible and meaningful.
class WaxBiteplateRec(NDIData, BiteplateRec):
    '''A class for wax biteplate recordings.'''
    def __init__(self, tsvname, colmap, nasion='REF', right_mastoid='RMA', left_mastoid='LMA', origin='OS', molar='MS', *args, **kwargs):
        super(WaxBiteplateRec, self).__init__(tsvname, colmap, **kwargs)
        self._nasion = nasion
        self._right = right_mastoid
        self._left = left_mastoid
        self._origin = origin
        self._molar = molar

    @property
    def translation(self):
        return -(self.sensor_mean_tvals(self._origin))

