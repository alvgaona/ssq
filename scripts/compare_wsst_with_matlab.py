#!/usr/bin/env python3
"""
Compare our WSST implementation against MATLAB reference data.

Usage:
    1. Run generate_wsst_reference.m in MATLAB
    2. Copy wsst_reference.mat to scripts/ directory
    3. Run: pixi run validate-wsst
"""

import sys
from pathlib import Path

import numpy as np
import ssq


class MatlabStruct:
    """Simple container to mimic MATLAB struct access."""
    pass


def load_matlab_reference(filepath: str):
    """Load MATLAB reference data from .mat file (supports v7.3 HDF5 format)."""
    try:
        import scipy.io
        mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        return mat["reference"]
    except NotImplementedError:
        import h5py

        def h5_to_struct(h5_group, h5_file):
            """Recursively convert HDF5 group to struct-like object."""
            result = MatlabStruct()
            for key in h5_group.keys():
                item = h5_group[key]
                if isinstance(item, h5py.Dataset):
                    data = item[()]
                    if item.dtype == np.uint16 or (hasattr(item.dtype, 'char') and item.dtype.char == 'H'):
                        try:
                            data = ''.join(chr(c) for c in data.flatten())
                        except (TypeError, ValueError):
                            pass
                    elif data.dtype.names and 'real' in data.dtype.names and 'imag' in data.dtype.names:
                        data = data['real'] + 1j * data['imag']
                    elif isinstance(data, np.ndarray) and data.ndim == 2:
                        data = data.T
                    elif isinstance(data, np.ndarray) and data.ndim == 1:
                        data = data.flatten()
                    setattr(result, key, data)
                elif isinstance(item, h5py.Group):
                    setattr(result, key, h5_to_struct(item, h5_file))
                elif isinstance(item, h5py.Reference):
                    deref = h5_file[item]
                    if isinstance(deref, h5py.Dataset):
                        data = deref[()]
                        if deref.dtype == np.uint16:
                            try:
                                data = ''.join(chr(c) for c in data.flatten())
                            except (TypeError, ValueError):
                                pass
                        setattr(result, key, data)
                    else:
                        setattr(result, key, h5_to_struct(deref, h5_file))
            return result

        with h5py.File(filepath, 'r') as f:
            return h5_to_struct(f['reference'], f)


def ensure_1d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.flatten()
    return arr


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def compare_outputs(
    name: str,
    signal: np.ndarray,
    matlab_spectrum: np.ndarray,
    matlab_freqs: np.ndarray,
    matlab_times: np.ndarray,
    fs: float,
    num_voices: int,
) -> dict:
    """Compare our WSST output against MATLAB reference."""
    signal = ensure_1d(signal)
    matlab_freqs = ensure_1d(matlab_freqs)
    matlab_times = ensure_1d(matlab_times)
    matlab_spectrum = ensure_2d(matlab_spectrum)

    # Ensure spectrum is (freq x time) orientation
    # MATLAB saves as (freq x time), but HDF5 transpose may flip it
    # Use frequency array length to determine correct orientation
    if matlab_spectrum.shape[0] != len(matlab_freqs) and matlab_spectrum.shape[1] == len(matlab_freqs):
        matlab_spectrum = matlab_spectrum.T

    # Recompute times if it was loaded incorrectly (sometimes HDF5 gives scalar)
    if len(matlab_times) != matlab_spectrum.shape[1]:
        n_times = matlab_spectrum.shape[1]
        matlab_times = np.arange(n_times) / fs

    # Compute our WSST
    our_spectrum, our_freqs, our_times = ssq.wsst(
        signal.astype(np.float64), fs, 'amor', num_voices
    )

    results = {
        "name": name,
        "passed": True,
        "errors": [],
        "warnings": [],
    }

    # Compare dimensions (may differ due to scale computation)
    results["our_shape"] = our_spectrum.shape
    results["matlab_shape"] = matlab_spectrum.shape

    # Compare frequency range
    results["our_freq_range"] = (our_freqs.min(), our_freqs.max())
    results["matlab_freq_range"] = (matlab_freqs.min(), matlab_freqs.max())

    # Compare time axis
    if len(our_times) == len(matlab_times):
        time_diff = np.max(np.abs(our_times - matlab_times))
        results["time_max_diff"] = time_diff
    else:
        results["warnings"].append(f"Time axis length mismatch: ours={len(our_times)}, MATLAB={len(matlab_times)}")

    # Find peak frequencies at middle time point
    t_mid_ours = our_spectrum.shape[1] // 2
    t_mid_matlab = matlab_spectrum.shape[1] // 2

    our_mag = np.abs(our_spectrum[:, t_mid_ours])
    matlab_mag = np.abs(matlab_spectrum[:, t_mid_matlab])

    our_peak_freq = our_freqs[np.argmax(our_mag)]
    matlab_peak_freq = matlab_freqs[np.argmax(matlab_mag)]

    results["our_peak_freq"] = our_peak_freq
    results["matlab_peak_freq"] = matlab_peak_freq

    # Check if peaks are close (within 10% or 5 Hz)
    freq_tolerance = max(5.0, 0.1 * matlab_peak_freq)
    if abs(our_peak_freq - matlab_peak_freq) > freq_tolerance:
        results["errors"].append(
            f"Peak frequency mismatch: ours={our_peak_freq:.1f} Hz, MATLAB={matlab_peak_freq:.1f} Hz"
        )
        results["passed"] = False

    return results


def main():
    script_dir = Path(__file__).parent
    ref_file = script_dir / "wsst_reference.mat"

    if not ref_file.exists():
        print(f"Error: Reference file not found: {ref_file}")
        print("\nTo generate reference data:")
        print("  1. Run generate_wsst_reference.m in MATLAB")
        print("  2. Copy wsst_reference.mat to scripts/ directory")
        return 1

    print("=" * 70)
    print("WSST Implementation Comparison: Python (ours) vs MATLAB")
    print("=" * 70)
    print()

    print(f"Loading reference data from: {ref_file}")
    ref = load_matlab_reference(str(ref_file))

    fs = float(np.asarray(ref.fs).flat[0])
    num_voices = int(np.asarray(ref.num_voices).flat[0])
    print(f"Parameters: fs={fs} Hz, num_voices={num_voices}, wavelet=amor")
    print()

    test_cases = ["test1", "test2", "test3", "test4"]
    all_passed = True
    all_results = []

    for test_name in test_cases:
        if not hasattr(ref, test_name):
            print(f"Skipping {test_name}: not found in reference data")
            continue

        test = getattr(ref, test_name)
        print(f"Test: {test.name}")
        print("-" * 50)

        results = compare_outputs(
            name=test.name,
            signal=test.signal,
            matlab_spectrum=test.spectrum,
            matlab_freqs=test.frequencies,
            matlab_times=test.times,
            fs=fs,
            num_voices=num_voices,
        )

        all_results.append(results)

        if results["passed"]:
            print(f"  Status: PASSED")
        else:
            print(f"  Status: FAILED")
            all_passed = False

        print(f"  Our shape: {results['our_shape']}, MATLAB shape: {results['matlab_shape']}")
        print(f"  Our freq range: {results['our_freq_range'][0]:.1f}-{results['our_freq_range'][1]:.1f} Hz")
        print(f"  MATLAB freq range: {results['matlab_freq_range'][0]:.1f}-{results['matlab_freq_range'][1]:.1f} Hz")
        print(f"  Peak frequency: ours={results['our_peak_freq']:.1f} Hz, MATLAB={results['matlab_peak_freq']:.1f} Hz")

        if "time_max_diff" in results:
            print(f"  Time axis max diff: {results['time_max_diff']:.6f} s")

        for error in results["errors"]:
            print(f"  ERROR: {error}")

        for warning in results["warnings"]:
            print(f"  WARNING: {warning}")

        print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed_count = sum(1 for r in all_results if r["passed"])
    total_count = len(all_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if all_passed:
        print("\nAll tests PASSED!")
        return 0
    else:
        print("\nSome tests FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
