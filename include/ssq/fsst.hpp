#ifndef SSQ_SSQ_HPP
#define SSQ_SSQ_HPP

#include "ssq/stft.hpp"
#include "ssq/types.hpp"

namespace ssq {

// Main FSST computation function
// signal: Input signal (1D array)
// sample_rate: Sampling frequency in Hz
// window: Analysis window (e.g., Kaiser window)
// threshold: Numerical stability threshold (default 1e-6)
FsstResult fsst(const Eigen::VectorXd& signal, double sample_rate, const Eigen::VectorXd& window,
                double threshold = 1e-6);

// Compute instantaneous frequency estimate (phase transform)
// omega = eta - (1 / 2*pi*j) * (V_dg / V_g)
// where eta is the bin frequency
// V_g is the STFT with window g
// V_dg is the STFT with window derivative dg
Eigen::MatrixXd compute_phase_transform(const StftResult& stft, double sample_rate, double threshold);

// Perform synchrosqueezing: reassign STFT energy to estimated frequency bins
// stft: Original STFT (V_g)
// omega: Instantaneous frequency estimates
// frequencies: Frequency axis
// threshold: Magnitude threshold below which no reassignment occurs
Eigen::MatrixXcd synchrosqueeze(const Eigen::MatrixXcd& stft, const Eigen::MatrixXd& omega,
                                const Eigen::VectorXd& frequencies, double threshold);

// Inverse FSST: reconstruct signal from synchrosqueezed spectrum
// Synchrosqueezing preserves column sums, so summing along frequency reconstructs the signal
// spectrum: Synchrosqueezed spectrum (freq_bins x time_steps)
// window: Analysis window used in forward transform
// Returns: Reconstructed signal
Eigen::VectorXd ifsst(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& window);
Eigen::VectorXd ifsst(const FsstResult& result, const Eigen::VectorXd& window);

// Inverse FSST with frequency range filtering (MATLAB-compatible)
// spectrum: Synchrosqueezed spectrum (freq_bins x time_steps)
// window: Analysis window used in forward transform
// frequencies: Frequency axis from forward transform
// freqrange: Frequency range [fmin, fmax] in Hz - only reconstruct this range
// Returns: Reconstructed signal containing only the specified frequency range
Eigen::VectorXd ifsst(const Eigen::MatrixXcd& spectrum, const Eigen::VectorXd& window,
                      const Eigen::VectorXd& frequencies, const std::pair<double, double>& freqrange);

}  // namespace ssq

#endif  // SSQ_SSQ_HPP
