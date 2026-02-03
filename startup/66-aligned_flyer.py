print(ttime.ctime() + ' >>>> ' + __file__)

"""
Aligned Flyer Module

This module provides an alternative to the standard FlyerWithMotors that performs
data interpolation at collection time instead of relying on post-processing with
the xas package.

All detector data is aligned to the APB trigger timestamps during collect(),
eliminating the need for separate interpolation steps.

APB (Analog Pizza Box)
----------------------
The APB is a custom electronics module developed at NSLS-II (Brookhaven National
Laboratory) for beamline data acquisition. The name comes from its flat, rectangular
form factor.

The APB is an analog-to-digital converter (ADC) system that:

1. Reads ion chamber signals:
   - i0:  Incident beam intensity (before sample)
   - it:  Transmitted beam intensity (after sample)
   - ir:  Reference foil signal (for energy calibration)
   - iff: Fluorescence signal

2. Timestamps data with high precision (~8ns resolution)

3. Generates trigger pulses that other detectors (Xspress3, Pilatus) use to
   synchronize their acquisitions

4. Streams data continuously during fly scans at high rates

In this module, the APB trigger timestamps serve as the "master clock" - the common
time grid that all other detector data gets interpolated to. This is why trigger
data is critical for alignment, and collection fails if triggers cannot be read.

Usage:
    # Use aligned_fly_scan_plan instead of fly_scan_plan
    RE(aligned_fly_scan_plan(name='test', trajectory_filename='Cu_K.txt', ...))

    # Or use aligned_flyer_hhm directly
    RE(bp.fly([aligned_flyer_hhm], md={...}))
"""

import time as ttime
import numpy as np
import h5py
from scipy.interpolate import interp1d
from collections import OrderedDict

import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


# Constants for encoder to energy conversion (Si(111) crystal)
HC_ANGSTROM_EV = 12398.42  # hc in eV·Å
D_SPACING_SI111 = 3.1356   # d-spacing for Si(111) in Å


def encoder2energy(encoder_counts, pulses_per_deg, offset=0):
    """
    Convert encoder counts to energy in eV using Bragg's law.

    Parameters
    ----------
    encoder_counts : float or np.array
        Encoder counts from the monochromator
    pulses_per_deg : float
        Encoder resolution (pulses per degree)
    offset : float
        Angle offset in degrees

    Returns
    -------
    energy : float or np.array
        Energy in eV
    """
    theta_deg = (encoder_counts / pulses_per_deg) - float(offset)
    theta_rad = np.deg2rad(theta_deg)
    energy = -HC_ANGSTROM_EV / (2 * D_SPACING_SI111 * np.sin(theta_rad))
    return energy


def read_apb_binary_file(file_path):
    """
    Read APB (Analog Pizza Box) binary file.

    Parameters
    ----------
    file_path : str
        Path to the .bin file

    Returns
    -------
    dict with keys:
        'timestamps': np.array of Unix timestamps
        'i0', 'it', 'ir', 'iff', 'aux1'-'aux4': np.array of ADC values
    """
    raw_data = np.fromfile(file_path, dtype=np.int32)
    num_columns = 10
    raw_data = raw_data.reshape((-1, num_columns))

    # Timestamp is in last two columns: seconds + nanoseconds
    timestamps = raw_data[:, -2] + raw_data[:, -1] * 8.0051232e-9

    channel_names = ['i0', 'it', 'ir', 'iff', 'aux1', 'aux2', 'aux3', 'aux4']
    result = {'timestamps': timestamps}
    for i, name in enumerate(channel_names):
        result[name] = raw_data[:, i].astype(np.float64)

    return result


def read_encoder_text_file(file_path):
    """
    Read encoder text file from PizzaBox.

    Parameters
    ----------
    file_path : str
        Path to the encoder .txt file

    Returns
    -------
    dict with keys:
        'timestamps': np.array of Unix timestamps
        'encoder_counts': np.array of encoder counts
    """
    data = np.genfromtxt(file_path)
    if data.ndim == 1:
        data = data.reshape((1, -1))

    timestamps = data[:, 0] + data[:, 1] * 1e-9
    encoder_counts = data[:, 2]

    return {
        'timestamps': timestamps,
        'encoder_counts': encoder_counts
    }


def read_trigger_binary_file(file_path, use_midpoint=True):
    """
    Read APB trigger binary file.

    Parameters
    ----------
    file_path : str
        Path to the trigger .bin file
    use_midpoint : bool
        If True, return midpoint between rise and fall edges.
        If False, return fall edge timestamps.

    Returns
    -------
    np.array of trigger timestamps
    """
    raw_data = np.fromfile(file_path, dtype=np.int32)
    raw_data = raw_data.reshape((-1, 3))

    timestamps = raw_data[:, 1] + raw_data[:, 2] * 8.0051232e-9
    transitions = raw_data[:, 0]

    if use_midpoint:
        # Find pairs of rise (1) and fall (0) edges
        rise_mask = transitions == 1
        fall_mask = transitions == 0

        rise_times = timestamps[rise_mask]
        fall_times = timestamps[fall_mask]

        # Use midpoint as trigger time
        n_triggers = min(len(rise_times), len(fall_times))
        if n_triggers > 0:
            trigger_times = (rise_times[:n_triggers] + fall_times[:n_triggers]) / 2
        else:
            trigger_times = fall_times
    else:
        # Use fall edge times directly
        trigger_times = timestamps[transitions == 0]

    return trigger_times


def read_xspress3_rois(h5_file_path, num_channels=4):
    """
    Read Xspress3 ROI data from HDF5 file.

    Parameters
    ----------
    h5_file_path : str
        Path to the Xspress3 HDF5 file
    num_channels : int
        Number of detector channels

    Returns
    -------
    dict mapping 'xs_ch{nn}_roi{mm}' to np.array of counts
    """
    result = {}
    try:
        with h5py.File(h5_file_path, 'r') as f:
            attrs_path = '/entry/instrument/detector/NDAttributes'
            if attrs_path not in f:
                return result

            all_keys = list(f[attrs_path].keys())

            for ch in range(1, num_channels + 1):
                base = f'CHAN{ch}ROI'
                roi_keys = [k for k in all_keys if k.startswith(base) and not k.endswith('LM')]

                for key in roi_keys:
                    roi_num = int(key.replace(base, ''))
                    data = f[f'{attrs_path}/{key}'][()].squeeze()
                    result[f'xs_ch{ch:02d}_roi{roi_num:02d}'] = data

    except Exception as e:
        print(f'Warning: Could not read Xspress3 file {h5_file_path}: {e}')

    return result


def read_pilatus_rois(h5_file_path, num_rois=4):
    """
    Read Pilatus ROI data from HDF5 file.

    Parameters
    ----------
    h5_file_path : str
        Path to the Pilatus HDF5 file
    num_rois : int
        Number of ROIs

    Returns
    -------
    dict mapping 'pil100k_roi{n}' to np.array of counts
    """
    result = {}
    try:
        with h5py.File(h5_file_path, 'r') as f:
            attrs_path = '/entry/instrument/NDAttributes'
            if attrs_path not in f:
                return result

            for i in range(num_rois):
                key = f'_ROI{i+1}Total'
                full_path = f'{attrs_path}/{key}'
                if full_path in f:
                    data = f[full_path][()].squeeze()
                    result[f'pil100k_roi{i+1}'] = data

    except Exception as e:
        print(f'Warning: Could not read Pilatus file {h5_file_path}: {e}')

    return result


def interpolate_to_timestamps(source_timestamps, source_values, target_timestamps,
                               method='linear', pre_bin=True, pre_bin_factor=5):
    """
    Interpolate source data to target timestamps.

    Parameters
    ----------
    source_timestamps : np.array
        Original timestamps
    source_values : np.array
        Original values
    target_timestamps : np.array
        Target timestamps to interpolate to
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')
    pre_bin : bool
        If True, pre-bin dense data before interpolation
    pre_bin_factor : int
        Pre-bin if source has more than this factor times target points

    Returns
    -------
    np.array of interpolated values
    """
    # Filter to valid time range
    min_t = target_timestamps.min()
    max_t = target_timestamps.max()
    mask = (source_timestamps >= min_t) & (source_timestamps <= max_t)

    src_t = source_timestamps[mask]
    src_v = source_values[mask]

    if len(src_t) < 2:
        # Not enough points, return zeros
        return np.zeros_like(target_timestamps)

    # Pre-bin if source is much denser than target
    if pre_bin and len(src_t) > pre_bin_factor * len(target_timestamps):
        n_bins = len(target_timestamps)
        bin_size = len(src_t) // n_bins

        binned_t = []
        binned_v = []
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(src_t)
            binned_t.append(np.mean(src_t[start:end]))
            binned_v.append(np.mean(src_v[start:end]))

        src_t = np.array(binned_t)
        src_v = np.array(binned_v)

    # Perform interpolation
    if method == 'nearest':
        interpolator = interp1d(src_t, src_v, kind='nearest',
                                bounds_error=False, fill_value='extrapolate')
    elif method == 'cubic':
        interpolator = interp1d(src_t, src_v, kind='cubic',
                                bounds_error=False, fill_value='extrapolate')
    else:  # linear
        interpolator = interp1d(src_t, src_v, kind='linear',
                                bounds_error=False, fill_value='extrapolate')

    return interpolator(target_timestamps)


class AlignedFlyerWithMotors(FlyerWithMotors):
    """
    A flyer that returns interpolated/aligned data instead of filestore references.

    This eliminates the need for post-processing interpolation in the xas package.
    All detector data is aligned to the APB trigger timestamps during collect().

    The aligned flyer:
    1. Collects raw data files during the fly scan (same as FlyerWithMotors)
    2. In collect(), reads all raw files
    3. Uses APB trigger timestamps as the common time grid
    4. Interpolates APB stream and encoder data to trigger times
    5. Yields aligned events with actual data values

    Parameters
    ----------
    default_dets : list
        Default detectors (typically [apb_stream, hhm_encoder])
    motors : list or Motor
        Motors to fly
    shutter : Device
        Shutter device
    pulses_per_deg : float
        Encoder resolution for energy conversion (default: 360000)
    angle_offset : float
        Monochromator angle offset in degrees (default: 0)
    interpolation_method : str
        Interpolation method: 'linear', 'nearest', 'cubic' (default: 'linear')
    yield_raw_events : bool
        If True, also yield raw filestore events (default: False)
    """

    def __init__(self, default_dets, motors, shutter, *args,
                 pulses_per_deg=360000, angle_offset=0,
                 interpolation_method='linear',
                 yield_raw_events=False, **kwargs):
        super().__init__(default_dets, motors, shutter, *args, **kwargs)

        self.pulses_per_deg = pulses_per_deg
        self.angle_offset = angle_offset
        self.interpolation_method = interpolation_method
        self.yield_raw_events = yield_raw_events

        self._file_paths = {}
        self._aligned_data = None
        self._collection_status = {}  # Track success/failure of each data source

    def set_angle_offset(self, offset):
        """Set the monochromator angle offset."""
        self.angle_offset = offset

    def _identify_detector_type(self, det):
        """
        Identify detector type using a priority-based approach.

        Returns one of: 'apb_stream', 'encoder', 'xspress3', 'pilatus', 'trigger', or None
        """
        # Priority 1: Explicit type declaration (most reliable)
        if hasattr(det, 'aligned_flyer_type'):
            return det.aligned_flyer_type

        # Priority 2: Check for unique attribute signatures
        det_name = getattr(det, 'name', '')

        # Encoder: has _full_path and pulses_per_deg attributes
        if hasattr(det, '_full_path') and hasattr(det, 'pulses_per_deg'):
            return 'encoder'

        # APB stream: has filename attribute and produces .bin files
        if hasattr(det, 'filename') and hasattr(det, 'stream'):
            return 'apb_stream'

        # Xspress3: has hdf5 component with specific structure
        if hasattr(det, 'hdf5') and hasattr(det, 'channel1'):
            return 'xspress3'

        # Pilatus: has hdf5 component and is an area detector
        if hasattr(det, 'hdf5') and hasattr(det, 'cam'):
            return 'pilatus'

        # Priority 3: Fall back to name matching (least reliable)
        det_name_lower = det_name.lower()
        if 'apb_stream' in det_name_lower or det_name_lower == 'apb_stream':
            return 'apb_stream'
        if 'encoder' in det_name_lower or 'enc' in det_name_lower:
            return 'encoder'
        if 'xs' in det_name_lower and hasattr(det, 'hdf5'):
            return 'xspress3'
        if 'pil' in det_name_lower and hasattr(det, 'hdf5'):
            return 'pilatus'

        return None

    def _gather_file_paths(self):
        """Gather file paths from all detectors after complete()."""
        self._file_paths = {}

        all_dets = self.dets + self.default_dets

        for det in all_dets:
            det_type = self._identify_detector_type(det)

            if det_type == 'apb_stream':
                if hasattr(det, 'filename') and det.filename is not None:
                    self._file_paths['apb'] = f"{det.filename}.bin"

            elif det_type == 'encoder':
                if hasattr(det, '_full_path'):
                    self._file_paths['encoder'] = det._full_path

            elif det_type == 'xspress3':
                if hasattr(det, 'hdf5') and hasattr(det.hdf5, 'full_file_name'):
                    self._file_paths['xspress3'] = det.hdf5.full_file_name.get()

            elif det_type == 'pilatus':
                if hasattr(det, 'hdf5') and hasattr(det.hdf5, 'full_file_name'):
                    self._file_paths['pilatus'] = det.hdf5.full_file_name.get()

            # Check for external trigger device (can be on any detector)
            if hasattr(det, 'ext_trigger_device'):
                trig = det.ext_trigger_device
                if hasattr(trig, 'fn'):
                    self._file_paths['trigger'] = trig.fn

    def _read_and_align_data(self):
        """
        Read all raw files and align to trigger timestamps.

        Returns
        -------
        aligned : OrderedDict
            Aligned data with 'timestamps' and data channels
        """
        self._gather_file_paths()
        self._collection_status = {
            'trigger': {'status': 'skipped', 'message': 'No trigger file found'},
            'apb': {'status': 'skipped', 'message': 'No APB file found'},
            'encoder': {'status': 'skipped', 'message': 'No encoder file found'},
            'xspress3': {'status': 'skipped', 'message': 'No Xspress3 file found'},
            'pilatus': {'status': 'skipped', 'message': 'No Pilatus file found'},
        }

        aligned = OrderedDict()
        aligned['timestamps'] = np.array([])

        # Read trigger timestamps (common time grid) - CRITICAL
        trigger_times = None
        if 'trigger' in self._file_paths:
            try:
                trigger_times = read_trigger_binary_file(self._file_paths['trigger'])
                aligned['timestamps'] = trigger_times
                self._collection_status['trigger'] = {
                    'status': 'success',
                    'message': f'Read {len(trigger_times)} trigger timestamps',
                    'n_points': len(trigger_times)
                }
                print_to_gui(f'Read {len(trigger_times)} trigger timestamps', add_timestamp=True)
            except Exception as e:
                self._collection_status['trigger'] = {
                    'status': 'error',
                    'message': str(e),
                    'file': self._file_paths.get('trigger')
                }
                print_to_gui(f'ERROR: Could not read trigger file: {e}', add_timestamp=True)

        if trigger_times is None or len(trigger_times) == 0:
            print_to_gui('ERROR: No trigger timestamps found, cannot align data', add_timestamp=True)
            return aligned

        # Read and interpolate APB data
        if 'apb' in self._file_paths:
            try:
                apb_data = read_apb_binary_file(self._file_paths['apb'])
                for channel in ['i0', 'it', 'ir', 'iff']:
                    aligned[channel] = interpolate_to_timestamps(
                        apb_data['timestamps'],
                        apb_data[channel],
                        trigger_times,
                        method=self.interpolation_method
                    )
                self._collection_status['apb'] = {
                    'status': 'success',
                    'message': f'Interpolated {len(apb_data["timestamps"])} -> {len(trigger_times)} points',
                    'channels': ['i0', 'it', 'ir', 'iff'],
                    'n_source_points': len(apb_data['timestamps']),
                    'n_output_points': len(trigger_times)
                }
                print_to_gui(f'Interpolated APB data ({len(apb_data["timestamps"])} -> {len(trigger_times)} points)',
                           add_timestamp=True)
            except Exception as e:
                self._collection_status['apb'] = {
                    'status': 'error',
                    'message': str(e),
                    'file': self._file_paths.get('apb')
                }
                print_to_gui(f'ERROR: Could not read/interpolate APB data: {e}', add_timestamp=True)

        # Read and interpolate encoder data
        if 'encoder' in self._file_paths:
            try:
                enc_data = read_encoder_text_file(self._file_paths['encoder'])
                encoder_interp = interpolate_to_timestamps(
                    enc_data['timestamps'],
                    enc_data['encoder_counts'],
                    trigger_times,
                    method=self.interpolation_method
                )
                aligned['encoder'] = encoder_interp
                aligned['energy'] = encoder2energy(
                    encoder_interp,
                    self.pulses_per_deg,
                    self.angle_offset
                )
                self._collection_status['encoder'] = {
                    'status': 'success',
                    'message': f'Interpolated encoder and converted to energy',
                    'channels': ['encoder', 'energy'],
                    'n_source_points': len(enc_data['timestamps']),
                    'n_output_points': len(trigger_times)
                }
                print_to_gui(f'Interpolated encoder data and converted to energy', add_timestamp=True)
            except Exception as e:
                self._collection_status['encoder'] = {
                    'status': 'error',
                    'message': str(e),
                    'file': self._file_paths.get('encoder')
                }
                print_to_gui(f'ERROR: Could not read/interpolate encoder data: {e}', add_timestamp=True)

        # Read Xspress3 ROI data (already aligned to triggers)
        if 'xspress3' in self._file_paths:
            try:
                xs_data = read_xspress3_rois(self._file_paths['xspress3'])
                channels_read = []
                for key, values in xs_data.items():
                    aligned[key] = values[:len(trigger_times)]
                    channels_read.append(key)
                self._collection_status['xspress3'] = {
                    'status': 'success',
                    'message': f'Read {len(xs_data)} ROI channels',
                    'channels': channels_read,
                    'n_points': len(trigger_times)
                }
                print_to_gui(f'Read Xspress3 ROI data ({len(xs_data)} channels)', add_timestamp=True)
            except Exception as e:
                self._collection_status['xspress3'] = {
                    'status': 'error',
                    'message': str(e),
                    'file': self._file_paths.get('xspress3')
                }
                print_to_gui(f'ERROR: Could not read Xspress3 data: {e}', add_timestamp=True)

        # Read Pilatus ROI data (already aligned to triggers)
        if 'pilatus' in self._file_paths:
            try:
                pil_data = read_pilatus_rois(self._file_paths['pilatus'])
                channels_read = []
                for key, values in pil_data.items():
                    aligned[key] = values[:len(trigger_times)]
                    channels_read.append(key)
                self._collection_status['pilatus'] = {
                    'status': 'success',
                    'message': f'Read {len(pil_data)} ROIs',
                    'channels': channels_read,
                    'n_points': len(trigger_times)
                }
                print_to_gui(f'Read Pilatus ROI data ({len(pil_data)} ROIs)', add_timestamp=True)
            except Exception as e:
                self._collection_status['pilatus'] = {
                    'status': 'error',
                    'message': str(e),
                    'file': self._file_paths.get('pilatus')
                }
                print_to_gui(f'ERROR: Could not read Pilatus data: {e}', add_timestamp=True)

        self._aligned_data = aligned
        return aligned

    def _log_collection_summary(self):
        """Log a summary of collection status for all data sources."""
        successes = []
        errors = []
        skipped = []

        for source, status in self._collection_status.items():
            if status['status'] == 'success':
                successes.append(source)
            elif status['status'] == 'error':
                errors.append(f"{source}: {status['message']}")
            else:
                skipped.append(source)

        # Summary line
        summary_parts = []
        if successes:
            summary_parts.append(f"OK: {', '.join(successes)}")
        if errors:
            summary_parts.append(f"FAILED: {len(errors)}")
        if skipped:
            summary_parts.append(f"skipped: {len(skipped)}")

        print_to_gui(f"Collection summary - {' | '.join(summary_parts)}",
                    add_timestamp=True, tag='AlignedFlyer')

        # Log errors in detail
        if errors:
            for err in errors:
                print_to_gui(f"  ERROR: {err}", add_timestamp=True, tag='AlignedFlyer')

    def get_collection_status(self):
        """
        Return the collection status dict for programmatic access.

        Returns
        -------
        dict
            Keys are data sources ('trigger', 'apb', 'encoder', 'xspress3', 'pilatus').
            Values are dicts with 'status' ('success', 'error', 'skipped'),
            'message', and additional info depending on status.
        """
        return self._collection_status.copy()

    def collect(self):
        """
        Read raw files and yield aligned events.

        Instead of yielding filestore references, this reads the raw data files,
        interpolates everything to the trigger timestamps, and yields actual data.

        Collection status for each data source is tracked and can be retrieved
        via get_collection_status() after collection completes.
        """
        print_to_gui('AlignedFlyer collect starting...', add_timestamp=True, tag='AlignedFlyer')

        # Optionally yield raw events first (for backwards compatibility)
        if self.yield_raw_events:
            print_to_gui('Yielding raw filestore events...', add_timestamp=True, tag='AlignedFlyer')
            yield from super().collect()

        # Read and align all data
        aligned = self._read_and_align_data()

        # Log collection summary
        self._log_collection_summary()

        if len(aligned.get('timestamps', [])) == 0:
            print_to_gui('ERROR: No aligned data to yield - check collection status',
                        add_timestamp=True, tag='AlignedFlyer')
            return

        timestamps = aligned['timestamps']
        n_points = len(timestamps)

        # Build list of data keys (excluding timestamps)
        data_keys = [k for k in aligned.keys() if k != 'timestamps']

        print_to_gui(f'Yielding {n_points} aligned events with {len(data_keys)} channels...',
                    add_timestamp=True, tag='AlignedFlyer')

        # Yield one event per trigger timestamp
        for i in range(n_points):
            ts = timestamps[i]

            data = {}
            for key in data_keys:
                values = aligned[key]
                if i < len(values):
                    data[key] = float(values[i])

            yield {
                'data': data,
                'timestamps': {k: ts for k in data.keys()},
                'time': ts,
            }

        print_to_gui(f'AlignedFlyer collect complete', add_timestamp=True, tag='AlignedFlyer')

    def describe_collect(self):
        """Describe the aligned data streams."""
        # Start with parent's description if yielding raw events
        if self.yield_raw_events:
            return_dict = super().describe_collect()
        else:
            return_dict = {}

        # Add aligned stream description
        aligned_desc = {
            'aligned': {
                'i0': {'source': 'APB', 'dtype': 'number', 'shape': []},
                'it': {'source': 'APB', 'dtype': 'number', 'shape': []},
                'ir': {'source': 'APB', 'dtype': 'number', 'shape': []},
                'iff': {'source': 'APB', 'dtype': 'number', 'shape': []},
                'energy': {'source': 'encoder', 'dtype': 'number', 'shape': []},
                'encoder': {'source': 'encoder', 'dtype': 'number', 'shape': []},
            }
        }

        # Add XS channels if we expect them
        for ch in range(1, 5):
            for roi in range(1, 5):
                aligned_desc['aligned'][f'xs_ch{ch:02d}_roi{roi:02d}'] = {
                    'source': 'xspress3', 'dtype': 'number', 'shape': []
                }

        # Add Pilatus ROIs
        for roi in range(1, 5):
            aligned_desc['aligned'][f'pil100k_roi{roi}'] = {
                'source': 'pilatus', 'dtype': 'number', 'shape': []
            }

        return {**return_dict, **aligned_desc}


# Create aligned flyer instance
aligned_flyer_hhm = AlignedFlyerWithMotors(
    [apb_stream, hhm_encoder],
    hhm,
    shutter,
    pulses_per_deg=hhm_encoder.pulses_per_deg,
    yield_raw_events=True,  # Capture both aligned data and raw filestore references
    name='aligned_flyer_hhm'
)


def get_aligned_fly_scan_md(name, comment, trajectory_filename, detectors_dict,
                            element, e0, edge, metadata):
    """Generate metadata for aligned fly scan."""
    md_general = get_hhm_scan_md(name, comment, trajectory_filename, detectors_dict,
                                  element, e0, edge, metadata, fn_ext='.raw')
    md_scan = {
        'experiment': 'aligned_fly_scan',
        'data_type': 'aligned',  # Indicates data is pre-aligned
    }
    return {**md_general, **md_scan}


def aligned_fly_scan_plan(name=None, comment=None, trajectory_filename=None,
                          mono_angle_offset=None, detectors=[],
                          element='', e0=0, edge='', metadata={}):
    """
    Fly scan plan that returns aligned/interpolated data.

    This plan is similar to fly_scan_plan but uses AlignedFlyerWithMotors
    to return pre-interpolated data instead of filestore references.

    Parameters
    ----------
    name : str
        Scan name
    comment : str
        Scan comment
    trajectory_filename : str
        Path to trajectory file
    mono_angle_offset : float
        Monochromator angle offset
    detectors : list
        List of detector names to include
    element : str
        Element being scanned
    e0 : float
        Edge energy in eV
    edge : str
        Edge name (e.g., 'K', 'L3')
    metadata : dict
        Additional metadata
    """
    if mono_angle_offset is not None:
        hhm.set_new_angle_offset(mono_angle_offset)
        aligned_flyer_hhm.set_angle_offset(mono_angle_offset)

    trajectory_stack.set_trajectory(trajectory_filename, offset=mono_angle_offset)

    aux_detectors = get_detector_device_list(detectors, flying=True)
    aligned_flyer_hhm.set_aux_dets(aux_detectors)

    detectors_dict = {k: {'device': v} for k, v in zip(detectors, aux_detectors)}
    md = get_aligned_fly_scan_md(name, comment, trajectory_filename, detectors_dict,
                                  element, e0, edge, metadata)

    @bpp.stage_decorator([aligned_flyer_hhm])
    def _fly(md):
        yield from bp.fly([aligned_flyer_hhm], md=md)

    yield from _fly(md)


def aligned_general_epics_motor_fly_scan(detectors, motor_dict, trajectory_dict, md,
                                          pulses_per_deg=None, angle_offset=0,
                                          interpolation_method='linear',
                                          yield_raw_events=True):
    """
    General fly scan for arbitrary EPICS motors with aligned/interpolated data.

    This is the aligned equivalent of general_epics_motor_fly_scan from
    65-fly_scan_plans.py. It uses AlignedFlyerWithMotors to return pre-interpolated
    data instead of filestore references.

    Parameters
    ----------
    detectors : list
        List of detector devices to use during the scan
    motor_dict : dict
        Dictionary mapping motor keys to motor devices, e.g.,
        {'motor1': some_motor, 'motor2': another_motor}
    trajectory_dict : dict
        Dictionary mapping motor keys to trajectory arrays, e.g.,
        {'motor1': trajectory_array_1, 'motor2': trajectory_array_2}
    md : dict
        Metadata dictionary for the scan
    pulses_per_deg : float, optional
        Encoder resolution for energy conversion. If None, energy conversion
        is skipped (appropriate for non-monochromator motors).
    angle_offset : float, optional
        Angle offset for energy conversion (default: 0)
    interpolation_method : str, optional
        Interpolation method: 'linear', 'nearest', 'cubic' (default: 'linear')
    yield_raw_events : bool, optional
        If True, also yield raw filestore events (default: True)

    Notes
    -----
    Unlike aligned_fly_scan_plan which is specific to the HHM monochromator,
    this function works with arbitrary EPICS motors. If pulses_per_deg is not
    provided, the aligned data will not include energy conversion (encoder
    counts will still be interpolated if available).

    After the scan completes, motors are returned to their initial positions.

    Examples
    --------
    >>> motor_dict = {'roll': cr_main_roll}
    >>> trajectory_dict = {'roll': np.linspace(0, 10, 100)}
    >>> md = {'purpose': 'alignment scan'}
    >>> RE(aligned_general_epics_motor_fly_scan(
    ...     [apb_stream], motor_dict, trajectory_dict, md))
    """
    flyable_motors = []
    motor_position_monitors = []
    motor_stream_names = []
    motor_init_pos = {}

    for motor_key, motor in motor_dict.items():
        flyable_motor = FlyableEpicsMotor(motor, name=f'flyable_{motor.name}')
        flyable_motor.set_trajectory(trajectory_dict[motor_key])
        flyable_motors.append(flyable_motor)
        motor_position_monitors.append(motor.user_readback)
        motor_stream_names.append(f'{motor.name}_monitor')
        motor_init_pos[motor_key] = motor.position

    md['motor_stream_names'] = motor_stream_names
    md['data_type'] = 'aligned'
    md['experiment'] = 'aligned_general_motor_fly_scan'

    # Create aligned flyer for general motors
    # Use default pulses_per_deg if not specified (energy conversion may not apply)
    flyer_pulses_per_deg = pulses_per_deg if pulses_per_deg is not None else 360000

    aligned_general_flyer = AlignedFlyerWithMotors(
        detectors,
        flyable_motors,
        shutter,
        pulses_per_deg=flyer_pulses_per_deg,
        angle_offset=angle_offset,
        interpolation_method=interpolation_method,
        yield_raw_events=yield_raw_events,
        name='aligned_general_motor_flyer'
    )

    @bpp.stage_decorator([aligned_general_flyer])
    def _fly(md):
        fly_plan = bp.fly([aligned_general_flyer], md=md)
        yield from monitor_during_wrapper(fly_plan, motor_position_monitors)

    yield from _fly(md)

    # Return motors to initial positions
    for motor_key, motor in motor_dict.items():
        motor.move(motor_init_pos[motor_key], wait=True)
