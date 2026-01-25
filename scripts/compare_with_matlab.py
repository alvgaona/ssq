#!/usr/bin/env python3
"""
Compare our FSST implementation against MATLAB reference data.

Usage:
    1. Run generate_fsst_reference.m in MATLAB
    2. Copy fsst_reference.mat to scripts/ directory
    3. Run: pixi run python scripts/compare_with_matlab.py

Requirements:
    - scipy or h5py (for loading .mat files)
    - fsst module (our implementation)
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
        # Try scipy first (for older .mat formats)
        import scipy.io
        mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=True)
        return mat["reference"]
    except NotImplementedError:
        # Use h5py for v7.3 HDF5 format
        import h5py

        def h5_to_struct(h5_group, h5_file):
            """Recursively convert HDF5 group to struct-like object."""
            result = MatlabStruct()
            for key in h5_group.keys():
                item = h5_group[key]
                if isinstance(item, h5py.Dataset):
                    data = item[()]
                    # Handle MATLAB strings (stored as uint16 arrays)
                    if item.dtype == np.uint16 or (hasattr(item.dtype, 'char') and item.dtype.char == 'H'):
                        try:
                            data = ''.join(chr(c) for c in data.flatten())
                        except (TypeError, ValueError):
                            pass
                    # Handle complex data stored as compound type
                    elif data.dtype.names and 'real' in data.dtype.names and 'imag' in data.dtype.names:
                        data = data['real'] + 1j * data['imag']
                    # Transpose to match MATLAB column-major order
                    elif isinstance(data, np.ndarray) and data.ndim == 2:
                        data = data.T
                    elif isinstance(data, np.ndarray) and data.ndim == 1:
                        data = data.flatten()
                    setattr(result, key, data)
                elif isinstance(item, h5py.Group):
                    setattr(result, key, h5_to_struct(item, h5_file))
                elif isinstance(item, h5py.Reference):
                    # Dereference and convert
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
    """Ensure array is 1D by flattening if needed."""
    arr = np.asarray(arr)
    if arr.ndim > 1:
        arr = arr.flatten()
    return arr


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Ensure array is 2D."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def compare_outputs(
    name: str,
    signal: np.ndarray,
    matlab_spectrum: np.ndarray,
    matlab_freqs: np.ndarray,
    matlab_times: np.ndarray,
    fs: float,
    window: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> dict:
    """Compare our FSST output against MATLAB reference."""
    # Ensure proper shapes
    signal = ensure_1d(signal)
    window = ensure_1d(window)
    matlab_freqs = ensure_1d(matlab_freqs)
    matlab_times = ensure_1d(matlab_times)
    matlab_spectrum = ensure_2d(matlab_spectrum)

    # MATLAB stores as (time x freq), we store as (freq x time)
    # Transpose MATLAB spectrum to match our convention
    if matlab_spectrum.shape[0] == len(matlab_times) and matlab_spectrum.shape[1] == len(matlab_freqs):
        matlab_spectrum = matlab_spectrum.T

    # Compute our FSST
    our_spectrum, our_freqs, our_times = ssq.fsst(signal.astype(np.float64), fs, window.astype(np.float64))

    results = {
        "name": name,
        "passed": True,
        "errors": [],
        "warnings": [],
    }

    # Compare dimensions
    if our_spectrum.shape != matlab_spectrum.shape:
        results["errors"].append(
            f"Spectrum shape mismatch: ours={our_spectrum.shape}, MATLAB={matlab_spectrum.shape}"
        )
        results["passed"] = False
    else:
        results["shape_match"] = True

    # Compare frequency axis
    if len(our_freqs) == len(matlab_freqs):
        freq_diff = np.max(np.abs(our_freqs - matlab_freqs))
        if freq_diff > atol:
            results["warnings"].append(f"Frequency axis max diff: {freq_diff:.6f} Hz")
        results["freq_max_diff"] = freq_diff
    else:
        results["errors"].append(f"Frequency axis length mismatch: ours={len(our_freqs)}, MATLAB={len(matlab_freqs)}")
        results["passed"] = False

    # Compare time axis
    if len(our_times) == len(matlab_times):
        time_diff = np.max(np.abs(our_times - matlab_times))
        if time_diff > atol:
            results["warnings"].append(f"Time axis max diff: {time_diff:.6f} s")
        results["time_max_diff"] = time_diff
    else:
        results["errors"].append(f"Time axis length mismatch: ours={len(our_times)}, MATLAB={len(matlab_times)}")
        results["passed"] = False

    # Compare spectrum values (if dimensions match)
    if our_spectrum.shape == matlab_spectrum.shape:
        # Compare magnitudes
        our_mag = np.abs(our_spectrum)
        matlab_mag = np.abs(matlab_spectrum)

        # Normalize for comparison
        our_mag_norm = our_mag / (np.max(our_mag) + 1e-10)
        matlab_mag_norm = matlab_mag / (np.max(matlab_mag) + 1e-10)

        mag_diff = np.abs(our_mag_norm - matlab_mag_norm)
        results["mag_max_diff"] = np.max(mag_diff)
        results["mag_mean_diff"] = np.mean(mag_diff)
        results["mag_median_diff"] = np.median(mag_diff)

        # Check if peak frequencies match at various time points
        test_times = [0.25, 0.5] if len(our_times) > 100 else [len(our_times) // 2]
        peak_matches = 0
        peak_tests = 0

        for t_frac in test_times:
            if isinstance(t_frac, float):
                t_idx = int(t_frac * fs)
                if t_idx >= our_spectrum.shape[1]:
                    continue
            else:
                t_idx = t_frac

            our_peak_idx = np.argmax(our_mag[:, t_idx])
            matlab_peak_idx = np.argmax(matlab_mag[:, t_idx])

            our_peak_freq = our_freqs[our_peak_idx]
            matlab_peak_freq = matlab_freqs[matlab_peak_idx]

            peak_tests += 1
            if abs(our_peak_freq - matlab_peak_freq) <= fs / len(window):  # Within 1 bin
                peak_matches += 1
            else:
                results["warnings"].append(
                    f"Peak mismatch at t_idx={t_idx}: ours={our_peak_freq:.1f} Hz, MATLAB={matlab_peak_freq:.1f} Hz"
                )

        results["peak_match_rate"] = peak_matches / peak_tests if peak_tests > 0 else 0

        # Overall correlation
        correlation = np.corrcoef(our_mag.flatten(), matlab_mag.flatten())[0, 1]
        results["magnitude_correlation"] = correlation

        if correlation < 0.9:
            results["errors"].append(f"Low magnitude correlation: {correlation:.4f}")
            results["passed"] = False

    return results


def main():
    # Find reference file
    script_dir = Path(__file__).parent
    ref_file = script_dir / "fsst_reference.mat"

    if not ref_file.exists():
        print(f"Error: Reference file not found: {ref_file}")
        print("\nTo generate reference data:")
        print("  1. Run generate_fsst_reference.m in MATLAB")
        print("  2. Copy fsst_reference.mat to scripts/ directory")
        return 1

    print("=" * 70)
    print("FSST Implementation Comparison: Python (ours) vs MATLAB")
    print("=" * 70)
    print()

    # Load reference data
    print(f"Loading reference data from: {ref_file}")
    ref = load_matlab_reference(str(ref_file))

    # Extract parameters
    window = ensure_1d(ref.window)
    fs = float(np.asarray(ref.fs).flat[0])
    kaiser_beta = float(np.asarray(ref.kaiser_beta).flat[0])
    print(f"Parameters: fs={fs} Hz, window_len={len(window)}, kaiser_beta={kaiser_beta}")
    print()

    # Compare each test case
    test_cases = ["test1", "test2", "test3", "test4", "test5", "test6"]
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
            window=window,
        )

        all_results.append(results)

        if results["passed"]:
            print(f"  Status: PASSED")
        else:
            print(f"  Status: FAILED")
            all_passed = False

        if "shape_match" in results:
            print(f"  Shape match: Yes")

        if "freq_max_diff" in results:
            print(f"  Frequency axis max diff: {results['freq_max_diff']:.6f} Hz")

        if "time_max_diff" in results:
            print(f"  Time axis max diff: {results['time_max_diff']:.6f} s")

        if "magnitude_correlation" in results:
            print(f"  Magnitude correlation: {results['magnitude_correlation']:.6f}")

        if "mag_max_diff" in results:
            print(f"  Magnitude diff (normalized): max={results['mag_max_diff']:.4f}, mean={results['mag_mean_diff']:.4f}")

        if "peak_match_rate" in results:
            print(f"  Peak frequency match rate: {results['peak_match_rate']*100:.0f}%")

        for error in results["errors"]:
            print(f"  ERROR: {error}")

        for warning in results["warnings"]:
            print(f"  WARNING: {warning}")

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed_count = sum(1 for r in all_results if r["passed"])
    total_count = len(all_results)

    print(f"Tests passed: {passed_count}/{total_count}")

    if all_passed:
        print("\nAll tests PASSED - Implementation matches MATLAB reference!")
        avg_correlation = np.mean([r.get("magnitude_correlation", 0) for r in all_results])
        print(f"Average magnitude correlation: {avg_correlation:.4f}")
        return 0
    else:
        print("\nSome tests FAILED - See details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
