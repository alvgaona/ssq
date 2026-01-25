#ifndef SSQ_WSST_HPP
#define SSQ_WSST_HPP

#include "ssq/cwt.hpp"
#include "ssq/types.hpp"

namespace ssq {

// WSST result (same structure as FsstResult for API consistency)
struct WsstResult {
    Eigen::MatrixXcd spectrum;  // Synchrosqueezed spectrum (freq_bins x time_steps)
    Eigen::VectorXd frequencies;
    Eigen::VectorXd times;
};

// Main WSST computation function (MATLAB-compatible API)
// signal: Input signal (1D array)
// sample_rate: Sampling frequency in Hz
// wavelet: Wavelet type ('amor' = Morlet, default)
// num_voices: Voices per octave (default 32)
// threshold: Numerical stability threshold (default 1e-6)
WsstResult wsst(const Eigen::VectorXd& signal, double sample_rate, WaveletType wavelet = WaveletType::Morlet,
                int num_voices = 32, double threshold = 1e-6);

// Compute instantaneous frequency from CWT (phase transform)
// omega = -Im(W'_psi / W_psi) * sample_rate / (2*pi)
Eigen::MatrixXd compute_wsst_phase_transform(const CwtResult& cwt, double sample_rate, double threshold);

// Synchrosqueeze CWT coefficients to frequency bins
Eigen::MatrixXcd wsst_synchrosqueeze(const Eigen::MatrixXcd& cwt, const Eigen::MatrixXd& omega,
                                     const Eigen::VectorXd& target_frequencies, double threshold);

}  // namespace ssq

#endif  // SSQ_WSST_HPP
