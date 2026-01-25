#include "ssq/wsst.hpp"

#include <cmath>

namespace ssq {

namespace {
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;
}  // namespace

Eigen::MatrixXd compute_wsst_phase_transform(const CwtResult& cwt, double sample_rate, double threshold) {
    // Compute instantaneous frequency from CWT
    //
    // For the wavelet synchrosqueezing transform, the instantaneous frequency is:
    // omega(a,b) = -Im(W'_psi(a,b) / W_psi(a,b)) / (2*pi)
    //
    // where W_psi is the CWT and W'_psi is the CWT with derivative wavelet
    // Since W'_psi is computed using physical angular frequency (rad/s) in the derivative,
    // the result is already in Hz after dividing by 2*pi
    (void)sample_rate;  // Not needed - derivative already uses physical frequency

    const Eigen::Index num_scales = cwt.cwt.rows();
    const Eigen::Index num_times = cwt.cwt.cols();

    Eigen::MatrixXd omega(num_scales, num_times);

    for (Eigen::Index s = 0; s < num_scales; ++s) {
        double scale_freq = cwt.frequencies(s);

        for (Eigen::Index t = 0; t < num_times; ++t) {
            std::complex<double> w = cwt.cwt(s, t);
            std::complex<double> w_d = cwt.cwt_d(s, t);

            double mag = std::abs(w);

            if (mag > threshold) {
                // omega = Im(W'_psi / W_psi) / (2*pi)
                // W'_psi already contains the physical frequency factor from the derivative
                // For analytic signal: W' = i*omega*W, so Im(W'/W) = omega
                std::complex<double> ratio = w_d * std::conj(w) / (mag * mag);
                double inst_freq = std::imag(ratio) / TWO_PI;

                // Clamp to positive frequencies
                inst_freq = std::max(0.0, inst_freq);

                omega(s, t) = inst_freq;
            } else {
                // When magnitude is too small, use the scale's center frequency
                omega(s, t) = scale_freq;
            }
        }
    }

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

    for (Eigen::Index s = 0; s < num_scales; ++s) {
        for (Eigen::Index t = 0; t < num_times; ++t) {
            std::complex<double> val = cwt(s, t);
            double mag = std::abs(val);

            if (mag > threshold) {
                double target_freq = omega(s, t);

                // Find nearest frequency bin by searching (frequencies may not be uniformly spaced)
                Eigen::Index target_bin = 0;
                double min_diff = std::abs(target_frequencies(0) - target_freq);
                for (Eigen::Index k = 1; k < num_freqs; ++k) {
                    double diff = std::abs(target_frequencies(k) - target_freq);
                    if (diff < min_diff) {
                        min_diff = diff;
                        target_bin = k;
                    }
                }

                // Accumulate at target bin
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

}  // namespace ssq
