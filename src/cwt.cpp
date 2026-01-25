#include "ssq/cwt.hpp"

#include <cmath>
#include <cstring>

namespace ssq {

Cwt::Cwt(WaveletType wavelet, int num_voices) : wavelet_type_(wavelet), num_voices_(num_voices), omega0_(6.0) {
    // omega0 = 6 is MATLAB's default for analytic Morlet
}

CwtResult Cwt::compute(const Eigen::VectorXd& signal, double sample_rate) const {
    // Compute default scales
    Eigen::VectorXd scales = compute_cwt_scales(signal.size(), sample_rate, num_voices_, omega0_);
    return compute(signal, sample_rate, scales);
}

CwtResult Cwt::compute(const Eigen::VectorXd& signal, double sample_rate, const Eigen::VectorXd& scales) const {
    // CWT via FFT convolution:
    // W(a,b) = IFFT( FFT(x) * conj(FFT(psi_a)) )
    //
    // where psi_a(t) = (1/sqrt(a)) * psi(t/a) is the scaled wavelet

    const Eigen::Index n = signal.size();
    const Eigen::Index num_scales = scales.size();
    const double dt = 1.0 / sample_rate;

    CwtResult result;
    result.cwt = Eigen::MatrixXcd::Zero(num_scales, n);
    result.cwt_d = Eigen::MatrixXcd::Zero(num_scales, n);
    result.scales = scales;
    result.frequencies = scales_to_frequencies(scales, sample_rate, omega0_);
    result.times = Eigen::VectorXd::LinSpaced(n, 0.0, static_cast<double>(n - 1) / sample_rate);

    // Allocate FFTW arrays for signal FFT
    FftwArray<double> signal_in(static_cast<size_t>(n));
    FftwArray<fftw_complex> signal_fft(static_cast<size_t>(n / 2 + 1));

    // Copy signal to FFTW array
    for (Eigen::Index i = 0; i < n; ++i) {
        signal_in[static_cast<size_t>(i)] = signal(i);
    }

    // Compute signal FFT (real-to-complex)
    FftwPlan forward_plan =
        FftwManager::instance().create_r2c_plan(static_cast<int>(n), signal_in.get(), signal_fft.get());
    FftwManager::instance().execute(forward_plan);

    // Convert to full complex spectrum for easier manipulation
    Eigen::VectorXcd signal_spectrum(n);

    // Copy positive frequencies from FFTW output
    for (Eigen::Index k = 0; k <= n / 2; ++k) {
        signal_spectrum(k) =
            std::complex<double>(signal_fft[static_cast<size_t>(k)][0], signal_fft[static_cast<size_t>(k)][1]);
    }

    // Negative frequencies via conjugate symmetry (vectorized)
    // signal_spectrum[n/2+1 : n-1] = conj(signal_spectrum[n/2-1 : 1])
    Eigen::Index neg_start = n / 2 + 1;
    Eigen::Index neg_count = n - neg_start;
    signal_spectrum.segment(neg_start, neg_count) = signal_spectrum.segment(1, neg_count).reverse().conjugate();

    // Allocate arrays for inverse FFT
    FftwArray<fftw_complex> product(static_cast<size_t>(n));
    FftwArray<fftw_complex> cwt_out(static_cast<size_t>(n));

    // Create inverse FFT plan (complex-to-complex)
    FftwPlan inverse_plan =
        FftwManager::instance().create_dft_plan(static_cast<int>(n), product.get(), cwt_out.get(), FFTW_BACKWARD);

    // Temporary vectors for FFTW interop
    Eigen::VectorXcd product_vec(n);
    Eigen::VectorXcd cwt_out_vec(n);
    const double norm = 1.0 / static_cast<double>(n);

    // Process each scale
    for (Eigen::Index s = 0; s < num_scales; ++s) {
        double scale = scales(s);

        // Get wavelet in frequency domain
        Eigen::VectorXcd psi = morlet_wavelet_freq(n, scale, dt, omega0_);
        Eigen::VectorXcd psi_d = morlet_wavelet_freq_derivative(n, scale, dt, omega0_);

        // Compute CWT for this scale: IFFT(signal_fft * conj(psi_fft))
        // Vectorized element-wise multiplication
        product_vec = signal_spectrum.cwiseProduct(psi.conjugate());

        // Copy to FFTW array
        std::memcpy(product.get(), product_vec.data(), static_cast<size_t>(n) * sizeof(fftw_complex));

        FftwManager::instance().execute(inverse_plan);

        // Copy from FFTW and normalize (vectorized)
        std::memcpy(cwt_out_vec.data(), cwt_out.get(), static_cast<size_t>(n) * sizeof(fftw_complex));
        result.cwt.row(s) = cwt_out_vec.transpose() * norm;

        // Compute CWT with derivative wavelet (vectorized)
        product_vec = signal_spectrum.cwiseProduct(psi_d.conjugate());
        std::memcpy(product.get(), product_vec.data(), static_cast<size_t>(n) * sizeof(fftw_complex));

        FftwManager::instance().execute(inverse_plan);

        std::memcpy(cwt_out_vec.data(), cwt_out.get(), static_cast<size_t>(n) * sizeof(fftw_complex));
        result.cwt_d.row(s) = cwt_out_vec.transpose() * norm;
    }

    return result;
}

}  // namespace ssq
