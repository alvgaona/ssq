#include "ssq/stft.hpp"

#include <cstring>

namespace ssq {

Stft::Stft(const Eigen::VectorXd& window, Eigen::Index nfft)
    : window_(window),
      window_derivative_(compute_window_derivative(window)),
      nfft_(nfft == 0 ? window.size() : nfft) {}

Eigen::VectorXd Stft::compute_window_derivative(const Eigen::VectorXd& window) {
    // Central difference: dg[i] = (window[i+1] - window[i-1]) / 2
    // Boundary: use 0 for out-of-bounds values
    const Eigen::Index n = window.size();
    Eigen::VectorXd dg(n);

    // Interior points (vectorized)
    if (n > 2) {
        dg.segment(1, n - 2) = (window.segment(2, n - 2) - window.segment(0, n - 2)) / 2.0;
    }

    // Boundary conditions
    dg(0) = window(1) / 2.0;           // (window[1] - 0) / 2
    dg(n - 1) = -window(n - 2) / 2.0;  // (0 - window[n-2]) / 2

    return dg;
}

void Stft::compute_fft_column(const Eigen::VectorXd& signal, const Eigen::VectorXd& win, Eigen::Index center,
                              std::complex<double>* output) const {
    const Eigen::Index win_len = win.size();
    const Eigen::Index half_win = win_len / 2;
    const Eigen::Index sig_len = signal.size();

    FftwArray<double> input_buffer(static_cast<size_t>(nfft_));
    FftwArray<fftw_complex> output_buffer(static_cast<size_t>(nfft_ / 2 + 1));

    std::memset(input_buffer.get(), 0, static_cast<size_t>(nfft_) * sizeof(double));

    for (Eigen::Index i = 0; i < win_len; ++i) {
        Eigen::Index sig_pos = center - half_win + i;
        double sample = (sig_pos >= 0 && sig_pos < sig_len) ? signal(sig_pos) : 0.0;
        input_buffer[static_cast<size_t>(i)] = sample * win(i);
    }

    FftwPlan plan = FftwManager::instance().create_r2c_plan(static_cast<int>(nfft_), input_buffer.get(),
                                                            output_buffer.get());
    FftwManager::instance().execute(plan);

    const Eigen::Index num_freqs = nfft_ / 2 + 1;
    for (Eigen::Index k = 0; k < num_freqs; ++k) {
        output[k] = std::complex<double>(output_buffer[static_cast<size_t>(k)][0],
                                         output_buffer[static_cast<size_t>(k)][1]);
    }
}

StftResult Stft::compute(const Eigen::VectorXd& signal, double sample_rate) const {
    const Eigen::Index sig_len = signal.size();
    const Eigen::Index num_freqs = nfft_ / 2 + 1;
    const Eigen::Index num_times = sig_len;

    StftResult result;
    result.stft = Eigen::MatrixXcd::Zero(num_freqs, num_times);
    result.stft_dg = Eigen::MatrixXcd::Zero(num_freqs, num_times);
    result.frequencies = Eigen::VectorXd::LinSpaced(num_freqs, 0.0, sample_rate / 2.0);
    result.times = Eigen::VectorXd::LinSpaced(num_times, 0.0, static_cast<double>(sig_len - 1) / sample_rate);

    Eigen::VectorXcd fft_g(num_freqs);
    Eigen::VectorXcd fft_dg(num_freqs);

    for (Eigen::Index t = 0; t < num_times; ++t) {
        compute_fft_column(signal, window_, t, fft_g.data());
        compute_fft_column(signal, window_derivative_, t, fft_dg.data());

        // Vectorized column copy
        result.stft.col(t) = fft_g;
        result.stft_dg.col(t) = fft_dg;
    }

    return result;
}

}  // namespace ssq
