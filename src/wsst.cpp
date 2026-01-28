#include "ssq/wsst.hpp"

#include "ssq/constants.hpp"

#include <algorithm>
#include <cmath>

namespace ssq {

namespace {

// Find nearest frequency bin using binary search (O(log n) instead of O(n))
// Assumes frequencies are sorted in ascending order
Eigen::Index find_nearest_bin(const Eigen::VectorXd& frequencies, double target) {
    const Eigen::Index n = frequencies.size();
    if (n == 0) return 0;
    if (target <= frequencies(0)) return 0;
    if (target >= frequencies(n - 1)) return n - 1;

    // Binary search for first element >= target
    const double* data = frequencies.data();
    const double* it = std::lower_bound(data, data + n, target);
    Eigen::Index idx = static_cast<Eigen::Index>(it - data);

    // Check if previous element is closer
    if (idx > 0 && (target - frequencies(idx - 1)) < (frequencies(idx) - target)) {
        return idx - 1;
    }
    return idx;
}

// Compute frequency spacing for integration (central differences)
Eigen::VectorXd compute_df(const Eigen::VectorXd& frequencies) {
    const Eigen::Index n = frequencies.size();
    Eigen::VectorXd df(n);
    df(0) = frequencies(1) - frequencies(0);
    for (Eigen::Index k = 1; k < n - 1; ++k) {
        df(k) = (frequencies(k + 1) - frequencies(k - 1)) / 2.0;
    }
    df(n - 1) = frequencies(n - 1) - frequencies(n - 2);
    return df;
}

// Internal implementation for inverse WSST with bin range
Eigen::VectorXd iwsst_impl(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& frequencies,
                           const Eigen::VectorXd& df, Eigen::Index k_min, Eigen::Index k_max) {
    const Eigen::Index num_times = spectrum.cols();

    // Weighted sum with scale-frequency correction: ∑_k S(k, t) * df_k / sqrt(f_k)
    Eigen::VectorXd result(num_times);
    for (Eigen::Index t = 0; t < num_times; ++t) {
        std::complex<double> sum = 0.0;
        for (Eigen::Index k = k_min; k <= k_max; ++k) {
            double weight = df(k) / std::sqrt(frequencies(k));
            sum += spectrum(k, t) * weight;
        }
        result(t) = sum.real();
    }

    // Normalization constant for Morlet wavelet
    // For omega0=6, the reconstruction normalization is omega0 + sqrt(pi)/4
    constexpr double omega0 = 6.0;
    double norm = omega0 + std::sqrt(PI) / 4.0;
    return result * norm;
}
}  // namespace

Eigen::MatrixXd compute_wsst_phase_transform(const CwtResult& cwt, double sample_rate, double threshold) {
    // Compute instantaneous frequency from CWT
    //
    // For the wavelet synchrosqueezing transform, the instantaneous frequency is:
    // omega(a,b) = Im(W'_psi(a,b) / W_psi(a,b)) / (2*pi)
    //
    // where W_psi is the CWT and W'_psi is the CWT with derivative wavelet
    // Since W'_psi is computed using physical angular frequency (rad/s) in the derivative,
    // the result is already in Hz after dividing by 2*pi
    (void)sample_rate;  // Not needed - derivative already uses physical frequency

    const Eigen::Index num_scales = cwt.cwt.rows();
    const Eigen::Index num_times = cwt.cwt.cols();

    // Compute magnitudes (vectorized)
    Eigen::MatrixXd mag = cwt.cwt.cwiseAbs();
    Eigen::MatrixXd mag_sq = mag.array().square();

    // Compute ratio = W_d * conj(W) / |W|^2 (vectorized)
    // Im(W_d * conj(W)) = Im(W_d) * Re(W) - Re(W_d) * Im(W)
    Eigen::MatrixXd imag_ratio =
        (cwt.cwt_d.imag().cwiseProduct(cwt.cwt.real()) - cwt.cwt_d.real().cwiseProduct(cwt.cwt.imag()))
            .cwiseQuotient(mag_sq.cwiseMax(threshold * threshold));

    // Initialize omega with scale frequencies (broadcast)
    Eigen::MatrixXd omega = cwt.frequencies.replicate(1, num_times);

    // Compute instantaneous frequency and clamp to non-negative
    Eigen::MatrixXd inst_freq = (imag_ratio / TWO_PI).cwiseMax(0.0);

    // Apply threshold mask: use inst_freq where mag > threshold, else keep omega
    omega = (mag.array() > threshold).select(inst_freq, omega);

    return omega;
}

Eigen::MatrixXcd wsst_synchrosqueeze(const Eigen::MatrixXcd& cwt, const Eigen::MatrixXd& omega,
                                     const Eigen::VectorXd& target_frequencies, double threshold) {
    // Synchrosqueeze CWT coefficients to frequency bins
    //
    // For each CWT coefficient at (scale, time):
    //   1. Get the estimated instantaneous frequency omega(scale, time)
    //   2. Find the closest frequency bin in target_frequencies
    //   3. Accumulate the CWT coefficient at that frequency bin

    const Eigen::Index num_scales = cwt.rows();
    const Eigen::Index num_times = cwt.cols();
    const Eigen::Index num_freqs = target_frequencies.size();

    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(num_freqs, num_times);

    // Column-first iteration for cache efficiency on column-major matrices
    for (Eigen::Index t = 0; t < num_times; ++t) {
        for (Eigen::Index s = 0; s < num_scales; ++s) {
            std::complex<double> val = cwt(s, t);
            double mag = std::abs(val);

            if (mag > threshold) {
                double target_freq = omega(s, t);
                Eigen::Index target_bin = find_nearest_bin(target_frequencies, target_freq);
                result(target_bin, t) += val;
            }
        }
    }

    return result;
}

WsstResult wsst(const Eigen::VectorXd& signal, double sample_rate, WaveletType wavelet, int num_voices,
                double threshold) {
    // Step 1: Compute CWT with wavelet and its derivative
    Cwt cwt_computer(wavelet, num_voices);
    CwtResult cwt_result = cwt_computer.compute(signal, sample_rate);

    // Step 2: Compute instantaneous frequency estimates
    Eigen::MatrixXd omega = compute_wsst_phase_transform(cwt_result, sample_rate, threshold);

    // Step 3: Create target frequency grid for synchrosqueezing
    // Use the CWT frequencies as the target (they're already computed from scales)
    // But we want them in ascending order (low to high frequency)
    Eigen::VectorXd target_frequencies = cwt_result.frequencies.reverse();

    // Step 4: Synchrosqueeze
    Eigen::MatrixXcd squeezed = wsst_synchrosqueeze(cwt_result.cwt, omega, target_frequencies, threshold);

    // Return result
    WsstResult result;
    result.spectrum = std::move(squeezed);
    result.frequencies = std::move(target_frequencies);
    result.times = std::move(cwt_result.times);

    return result;
}

Eigen::VectorXd iwsst(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& frequencies) {
    const Eigen::Index num_freqs = spectrum.rows();
    if (num_freqs < 2) {
        return spectrum.colwise().sum().real().transpose();
    }

    Eigen::VectorXd df = compute_df(frequencies);
    return iwsst_impl(spectrum, frequencies, df, 0, num_freqs - 1);
}

Eigen::VectorXd iwsst(const WsstResult& result) {
    return iwsst(result.spectrum, result.frequencies);
}

Eigen::VectorXd iwsst(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& frequencies,
                      const std::pair<double, double>& freqrange) {
    const Eigen::Index num_freqs = spectrum.rows();
    const Eigen::Index num_times = spectrum.cols();

    if (num_freqs < 2) {
        // Edge case: single frequency bin
        Eigen::VectorXd result(num_times);
        bool in_range = frequencies(0) >= freqrange.first && frequencies(0) <= freqrange.second;
        for (Eigen::Index t = 0; t < num_times; ++t) {
            result(t) = in_range ? spectrum(0, t).real() : 0.0;
        }
        return result;
    }

    // Find bin indices for frequency range using binary search
    const double* freq_data = frequencies.data();
    auto it_min = std::lower_bound(freq_data, freq_data + num_freqs, freqrange.first);
    auto it_max = std::upper_bound(freq_data, freq_data + num_freqs, freqrange.second);
    Eigen::Index k_min = static_cast<Eigen::Index>(it_min - freq_data);
    Eigen::Index k_max = static_cast<Eigen::Index>(it_max - freq_data);
    if (k_max > 0) --k_max;

    // Handle empty range
    if (k_min > k_max) {
        return Eigen::VectorXd::Zero(num_times);
    }

    Eigen::VectorXd df = compute_df(frequencies);
    return iwsst_impl(spectrum, frequencies, df, k_min, k_max);
}

}  // namespace ssq
