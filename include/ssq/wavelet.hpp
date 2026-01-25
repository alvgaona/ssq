#ifndef SSQ_WAVELET_HPP
#define SSQ_WAVELET_HPP

#include <Eigen/Dense>
#include <complex>

namespace ssq {

// Wavelet types supported (matching MATLAB naming)
enum class WaveletType {
    Morlet,  // 'amor' - Analytic Morlet (default)
    Bump     // 'bump' - Bump wavelet (future)
};

// Wavelet parameters
struct WaveletParams {
    double center_frequency;  // f_c: center frequency in cycles/unit
    double bandwidth;         // For Morlet: related to sigma
};

// Get default parameters for a wavelet type
WaveletParams get_wavelet_params(WaveletType type);

// Compute the analytic Morlet wavelet in frequency domain
// psi_hat(omega) = pi^(-1/4) * exp(-(omega - omega_0)^2 / 2) for omega > 0
// Returns wavelet evaluated at angular frequencies omega = 2*pi*f
// scale: wavelet scale parameter
// n: number of frequency points
// dt: sampling period (1/sample_rate)
Eigen::VectorXcd morlet_wavelet_freq(Eigen::Index n, double scale, double dt, double omega0 = 6.0);

// Compute the derivative of the Morlet wavelet in frequency domain
// Used for instantaneous frequency estimation
// d/dt psi(t) <-> i*omega * psi_hat(omega)
Eigen::VectorXcd morlet_wavelet_freq_derivative(Eigen::Index n, double scale, double dt, double omega0 = 6.0);

// Compute scales for CWT (logarithmically spaced)
// Returns scales from high frequency (small scale) to low frequency (large scale)
// num_voices: voices per octave (MATLAB default: 32)
// signal_length: length of input signal
// sample_rate: sampling frequency
Eigen::VectorXd compute_cwt_scales(Eigen::Index signal_length, double sample_rate, int num_voices = 32,
                                   double omega0 = 6.0);

// Convert scales to frequencies for a given wavelet
// f = omega0 / (2*pi*scale) * sample_rate
Eigen::VectorXd scales_to_frequencies(const Eigen::VectorXd& scales, double sample_rate, double omega0 = 6.0);

}  // namespace ssq

#endif  // SSQ_WAVELET_HPP
