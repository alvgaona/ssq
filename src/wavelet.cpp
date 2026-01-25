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

    Eigen::VectorXcd psi(n);

    // Normalization factor includes dt for proper energy scaling
    const double norm = std::sqrt(scale * dt) * std::pow(PI, -0.25);

    // Normalized angular frequency spacing (0 to 2*pi over n bins)
    double domega = TWO_PI / static_cast<double>(n);

    for (Eigen::Index k = 0; k < n; ++k) {
        // Compute normalized angular frequency for this bin
        // For FFT: positive frequencies are k=0..n/2, negative are k=n/2+1..n-1
        double omega;
        if (k <= n / 2) {
            omega = k * domega;
        } else {
            omega = (k - n) * domega;
        }

        // Scaled angular frequency
        double xi = scale * omega;

        // Morlet wavelet (analytic - only positive frequencies)
        if (omega > 0) {
            double exponent = -0.5 * (xi - omega0) * (xi - omega0);
            psi(k) = norm * std::exp(exponent);
        } else {
            psi(k) = 0.0;
        }
    }

    return psi;
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
    double domega_physical = TWO_PI / (n * dt);

    for (Eigen::Index k = 0; k < n; ++k) {
        double omega_physical;
        if (k <= n / 2) {
            omega_physical = k * domega_physical;
        } else {
            omega_physical = (k - n) * domega_physical;
        }

        // Multiply by -i*omega (physical angular frequency)
        // The negative sign accounts for the conjugate wavelet used in CWT
        psi(k) *= std::complex<double>(0.0, -omega_physical);
    }

    return psi;
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
    Eigen::VectorXd scales(num_scales);
    double log_scale_min = std::log(scale_min);
    double log_scale_max = std::log(scale_max);

    for (Eigen::Index i = 0; i < num_scales; ++i) {
        double t = (num_scales > 1) ? static_cast<double>(i) / static_cast<double>(num_scales - 1) : 0.0;
        scales(i) = std::exp(log_scale_min + t * (log_scale_max - log_scale_min));
    }

    return scales;
}

Eigen::VectorXd scales_to_frequencies(const Eigen::VectorXd& scales, double sample_rate, double omega0) {
    // Convert scales to frequencies
    // For Morlet wavelet: f = omega0 / (2*pi*scale) * sample_rate
    //
    // This gives the pseudo-frequency corresponding to each scale

    Eigen::VectorXd frequencies(scales.size());

    for (Eigen::Index i = 0; i < scales.size(); ++i) {
        frequencies(i) = omega0 * sample_rate / (TWO_PI * scales(i));
    }

    return frequencies;
}

}  // namespace ssq
