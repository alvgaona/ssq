#include "ssq/fsst.hpp"

#include "ssq/fftw_wrapper.hpp"

#include <algorithm>
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
    const double nyquist = sample_rate / 2.0;

    // Compute magnitudes (vectorized)
    Eigen::MatrixXd mag = stft.stft.cwiseAbs();
    Eigen::MatrixXd mag_sq = mag.array().square();

    // Compute ratio = V_dg * conj(V_g) / |V_g|^2 (vectorized)
    // Im(V_dg * conj(V_g)) = Im(V_dg) * Re(V_g) - Re(V_dg) * Im(V_g)
    Eigen::MatrixXd imag_ratio =
        (stft.stft_dg.imag().cwiseProduct(stft.stft.real()) - stft.stft_dg.real().cwiseProduct(stft.stft.imag()))
            .cwiseQuotient(mag_sq.cwiseMax(threshold * threshold));

    // Initialize omega with bin frequencies (broadcast)
    Eigen::MatrixXd omega = stft.frequencies.replicate(1, num_times);

    // Compute instantaneous frequency where magnitude is above threshold
    // inst_freq = bin_freq - Im(ratio) / (2*pi)
    Eigen::MatrixXd inst_freq = omega - imag_ratio / TWO_PI;

    // Apply threshold mask and clamp (column-first for cache efficiency)
    for (Eigen::Index t = 0; t < num_times; ++t) {
        for (Eigen::Index k = 0; k < num_freqs; ++k) {
            if (mag(k, t) > threshold) {
                omega(k, t) = std::clamp(inst_freq(k, t), 0.0, nyquist);
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

    // Column-first iteration for cache efficiency on column-major matrices
    for (Eigen::Index t = 0; t < num_times; ++t) {
        for (Eigen::Index k = 0; k < num_freqs; ++k) {
            std::complex<double> val = stft(k, t);
            double mag = std::abs(val);

            if (mag > threshold) {
                double target_freq = omega(k, t);
                double k_prime = (target_freq - frequencies(0)) / df;
                Eigen::Index target_bin = static_cast<Eigen::Index>(std::round(k_prime));
                target_bin = std::max(Eigen::Index{0}, std::min(target_bin, num_freqs - 1));
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

Eigen::VectorXd ifsst(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& window) {
    // Inverse FSST: reconstruct signal using overlap-add synthesis
    //
    // For each time point:
    //   1. Inverse FFT the spectrum column to get a time-domain frame
    //   2. Window the frame
    //   3. Overlap-add into the output

    const Eigen::Index num_freqs = spectrum.rows();
    const Eigen::Index num_times = spectrum.cols();
    const Eigen::Index nfft = (num_freqs - 1) * 2;
    const Eigen::Index half_win = nfft / 2;
    const Eigen::Index win_len = window.size();

    Eigen::VectorXd reconstructed = Eigen::VectorXd::Zero(num_times);
    Eigen::VectorXd window_sum = Eigen::VectorXd::Zero(num_times);

    // Allocate FFTW arrays for inverse FFT
    FftwArray<fftw_complex> freq_in(static_cast<size_t>(num_freqs));
    FftwArray<double> time_out(static_cast<size_t>(nfft));

    for (Eigen::Index t = 0; t < num_times; ++t) {
        // Copy spectrum column to FFTW array
        for (Eigen::Index k = 0; k < num_freqs; ++k) {
            freq_in[static_cast<size_t>(k)][0] = spectrum(k, t).real();
            freq_in[static_cast<size_t>(k)][1] = spectrum(k, t).imag();
        }

        // Inverse FFT (complex-to-real)
        FftwManager::instance().execute_c2r(static_cast<int>(nfft), freq_in.get(), time_out.get());

        // Overlap-add with window
        for (Eigen::Index i = 0; i < win_len && i < nfft; ++i) {
            Eigen::Index out_idx = t - half_win + i;
            if (out_idx >= 0 && out_idx < num_times) {
                double sample = time_out[static_cast<size_t>(i)] / static_cast<double>(nfft);
                reconstructed(out_idx) += sample * window(i);
                window_sum(out_idx) += window(i) * window(i);
            }
        }
    }

    // Normalize by window sum
    for (Eigen::Index i = 0; i < num_times; ++i) {
        if (window_sum(i) > 1e-10) {
            reconstructed(i) /= window_sum(i);
        }
    }

    return reconstructed;
}

Eigen::VectorXd ifsst(const FsstResult& result, const Eigen::VectorXd& window) {
    return ifsst(result.spectrum, window);
}

}  // namespace ssq
