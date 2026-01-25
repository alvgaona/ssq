#include "ssq/fsst.hpp"

#include <cmath>

namespace ssq {

namespace {
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;
}  // namespace

Eigen::MatrixXd compute_phase_transform(const StftResult& stft, double sample_rate, double threshold) {
    // Compute instantaneous frequency estimate:
    // omega(t, k) = k * df - Im(V_dg(t,k) / V_g(t,k)) / (2*pi)
    //
    // The instantaneous frequency is derived from the phase derivative.
    // Using V_dg (STFT with window derivative) and V_g (STFT with window),
    // we can estimate the instantaneous frequency without explicit phase unwrapping.

    const Eigen::Index num_freqs = stft.stft.rows();
    const Eigen::Index num_times = stft.stft.cols();

    Eigen::MatrixXd omega(num_freqs, num_times);

    for (Eigen::Index k = 0; k < num_freqs; ++k) {
        double bin_freq = stft.frequencies(k);

        for (Eigen::Index t = 0; t < num_times; ++t) {
            std::complex<double> vg = stft.stft(k, t);
            std::complex<double> vdg = stft.stft_dg(k, t);

            double mag = std::abs(vg);

            if (mag > threshold) {
                // omega = eta - Im(V_dg / V_g) / (2*pi)
                // V_dg / V_g = (V_dg * conj(V_g)) / |V_g|^2
                std::complex<double> ratio = vdg * std::conj(vg) / (mag * mag);
                double inst_freq = bin_freq - std::imag(ratio) / TWO_PI;

                // Clamp to valid frequency range
                inst_freq = std::max(0.0, std::min(inst_freq, sample_rate / 2.0));

                omega(k, t) = inst_freq;
            } else {
                // When magnitude is too small, use bin frequency
                omega(k, t) = bin_freq;
            }
        }
    }

    return omega;
}

Eigen::MatrixXcd synchrosqueeze(const Eigen::MatrixXcd& stft, const Eigen::MatrixXd& omega,
                                const Eigen::VectorXd& frequencies, double threshold) {
    // Reassign STFT energy to nearest frequency bin based on instantaneous frequency
    //
    // For each STFT bin (k, t):
    //   1. Get the estimated instantaneous frequency omega(k, t)
    //   2. Find the closest frequency bin k' such that f[k'] is nearest to omega(k, t)
    //   3. Add the STFT value to the output at position (k', t)

    const Eigen::Index num_freqs = stft.rows();
    const Eigen::Index num_times = stft.cols();

    // Initialize output with zeros
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(num_freqs, num_times);

    // Compute frequency bin spacing
    double df = (num_freqs > 1) ? (frequencies(num_freqs - 1) - frequencies(0)) / static_cast<double>(num_freqs - 1)
                                : 1.0;

    for (Eigen::Index k = 0; k < num_freqs; ++k) {
        for (Eigen::Index t = 0; t < num_times; ++t) {
            std::complex<double> val = stft(k, t);
            double mag = std::abs(val);

            if (mag > threshold) {
                // Find the target frequency bin
                double target_freq = omega(k, t);

                // Linear interpolation to find nearest bin
                // k' = (omega - f[0]) / df
                double k_prime = (target_freq - frequencies(0)) / df;

                // Round to nearest bin and clamp
                Eigen::Index target_bin = static_cast<Eigen::Index>(std::round(k_prime));
                target_bin = std::max(Eigen::Index{0}, std::min(target_bin, num_freqs - 1));

                // Accumulate energy at target bin
                result(target_bin, t) += val;
            }
        }
    }

    return result;
}

FsstResult fsst(const Eigen::VectorXd& signal, double sample_rate, const Eigen::VectorXd& window, double threshold) {
    // Step 1: Compute STFT with window and its derivative
    Stft stft_computer(window);
    StftResult stft_result = stft_computer.compute(signal, sample_rate);

    // Step 2: Compute instantaneous frequency estimates (phase transform)
    Eigen::MatrixXd omega = compute_phase_transform(stft_result, sample_rate, threshold);

    // Step 3: Synchrosqueeze - reassign energy to estimated frequency bins
    Eigen::MatrixXcd squeezed = synchrosqueeze(stft_result.stft, omega, stft_result.frequencies, threshold);

    // Return result
    FsstResult result;
    result.spectrum = std::move(squeezed);
    result.frequencies = std::move(stft_result.frequencies);
    result.times = std::move(stft_result.times);

    return result;
}

}  // namespace ssq
