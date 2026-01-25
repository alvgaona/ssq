#include "ssq/wavelet.hpp"

#include <cmath>

namespace ssq {

namespace {
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;
}  // namespace

WaveletParams get_wavelet_params(WaveletType type) {
    WaveletParams params;

    switch (type) {
        case WaveletType::Morlet:
            // Analytic Morlet (MATLAB 'amor')
            // omega0 = 6 corresponds to center frequency f_c ≈ 0.9549 cycles/unit
            params.center_frequency = 6.0 / TWO_PI;  // ~0.9549
            params.bandwidth = 1.0;
            break;

        case WaveletType::Bump:
            // Bump wavelet - to be implemented
            params.center_frequency = 1.0;
            params.bandwidth = 1.0;
            break;
    }

    return params;
}

Eigen::VectorXcd morlet_wavelet_freq(Eigen::Index n, double scale, double dt, double omega0) {
    // Analytic Morlet wavelet in frequency domain:
    // psi_hat(s*omega) = sqrt(s) * pi^(-1/4) * exp(-(s*omega - omega0)^2 / 2)
    //
    // For CWT: we need sqrt(scale) * conj(psi_hat(scale * omega))
    // The conjugate is because CWT uses <f, psi_a,b> = integral f(t) * conj(psi((t-b)/a)) dt
    //
    // omega0 is in normalized angular frequency units (typical value ~6)
    // omega is the normalized angular frequency: omega = 2*pi*k/n

    // Normalization factor includes dt for proper energy scaling
    const double norm = std::sqrt(scale * dt) * std::pow(PI, -0.25);

    // Normalized angular frequency spacing (0 to 2*pi over n bins)
    const double domega = TWO_PI / static_cast<double>(n);

    // Build frequency vector: positive frequencies [0, n/2], then negative [(n/2+1)-n, -1]
    Eigen::VectorXd omega(n);
    for (Eigen::Index k = 0; k <= n / 2; ++k) {
        omega(k) = k * domega;
    }
    for (Eigen::Index k = n / 2 + 1; k < n; ++k) {
        omega(k) = (k - n) * domega;
    }

    // Scaled angular frequency (vectorized)
    Eigen::VectorXd xi = scale * omega.array();

    // Compute wavelet: norm * exp(-0.5 * (xi - omega0)^2) for positive frequencies only
    Eigen::ArrayXd exponent = -0.5 * (xi.array() - omega0).square();
    Eigen::ArrayXd magnitude = norm * exponent.exp();

    // Zero out non-positive frequencies (analytic wavelet)
    magnitude = (omega.array() > 0.0).select(magnitude, 0.0);

    return magnitude.cast<std::complex<double>>();
}

Eigen::VectorXcd morlet_wavelet_freq_derivative(Eigen::Index n, double scale, double dt, double omega0) {
    // Derivative of wavelet in time domain corresponds to multiplication by i*omega in frequency domain
    // d/dt psi(t) <-> i*omega * psi_hat(omega)
    //
    // For the phase transform, we need the CWT with the derivative wavelet
    // The omega here should be in physical units (rad/s) for the derivative to give correct
    // time-domain derivative: d/dt * exp(i*omega*t) = i*omega * exp(i*omega*t)

    Eigen::VectorXcd psi = morlet_wavelet_freq(n, scale, dt, omega0);

    // Physical angular frequency spacing (rad/s)
    // omega_physical = omega_normalized / dt = 2*pi*k/(n*dt) = 2*pi*f
    const double domega_physical = TWO_PI / (n * dt);

    // Build physical frequency vector
    Eigen::VectorXd omega_physical(n);
    for (Eigen::Index k = 0; k <= n / 2; ++k) {
        omega_physical(k) = k * domega_physical;
    }
    for (Eigen::Index k = n / 2 + 1; k < n; ++k) {
        omega_physical(k) = (k - n) * domega_physical;
    }

    // Multiply by -i*omega (vectorized)
    // psi = psi * (-i * omega_physical)
    // (-i * omega) has real=0, imag=-omega
    // (a + bi) * (-ci) = bc + (-ac)i = bc - aci
    Eigen::ArrayXd psi_real = psi.real();
    Eigen::ArrayXd psi_imag = psi.imag();
    Eigen::ArrayXd neg_omega = -omega_physical.array();

    Eigen::VectorXcd result(n);
    result.real() = psi_imag * neg_omega;
    result.imag() = psi_real * neg_omega;

    return result;
}

Eigen::VectorXd compute_cwt_scales(Eigen::Index signal_length, double sample_rate, int num_voices, double omega0) {
    // Compute scales for CWT, matching MATLAB's approach
    //
    // For Morlet wavelet: f = omega0 * sample_rate / (2*pi*scale)
    // So: scale = omega0 * sample_rate / (2*pi*f)
    //
    // Scale range is determined by:
    // - Minimum scale: corresponds to Nyquist frequency (f_max = sample_rate/2)
    // - Maximum scale: limited by signal length (minimum resolvable frequency)

    // Minimum scale: corresponds to highest frequency we can represent
    // f_max = sample_rate / 2 (Nyquist)
    // scale_min = omega0 * sample_rate / (2*pi * sample_rate/2) = omega0 / pi
    double scale_min = omega0 / PI;

    // Maximum scale: corresponds to lowest frequency we can resolve
    // We want at least ~2 cycles of the wavelet to fit in the signal
    // f_min = 2 * sample_rate / signal_length
    // scale_max = omega0 * sample_rate / (2*pi * f_min) = omega0 * signal_length / (4*pi)
    double scale_max = omega0 * static_cast<double>(signal_length) / (4.0 * PI);

    // Ensure scale_max > scale_min
    if (scale_max <= scale_min) {
        scale_max = scale_min * 2.0;
    }

    // Number of octaves
    double num_octaves = std::log2(scale_max / scale_min);
    Eigen::Index num_scales = static_cast<Eigen::Index>(std::ceil(num_octaves * num_voices));

    if (num_scales < 2) {
        num_scales = 2;
    }

    // Generate logarithmically spaced scales (from small to large = high to low frequency)
    // Using vectorized operations
    double log_scale_min = std::log(scale_min);
    double log_scale_max = std::log(scale_max);

    Eigen::VectorXd t = Eigen::VectorXd::LinSpaced(num_scales, 0.0, 1.0);
    Eigen::VectorXd log_scales = log_scale_min + t.array() * (log_scale_max - log_scale_min);

    return log_scales.array().exp();
}

Eigen::VectorXd scales_to_frequencies(const Eigen::VectorXd& scales, double sample_rate, double omega0) {
    // Convert scales to frequencies (vectorized)
    // For Morlet wavelet: f = omega0 / (2*pi*scale) * sample_rate
    //
    // This gives the pseudo-frequency corresponding to each scale
    const double factor = omega0 * sample_rate / TWO_PI;
    return factor / scales.array();
}

}  // namespace ssq
