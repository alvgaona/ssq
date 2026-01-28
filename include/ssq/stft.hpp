#ifndef SSQ_STFT_HPP
#define SSQ_STFT_HPP

#include "ssq/types.hpp"

namespace ssq {

class Stft {
public:
    explicit Stft(const Eigen::VectorXd& window, Eigen::Index nfft = 0);

    StftResult compute(const Eigen::VectorXd& signal, double sample_rate) const;

    const Eigen::VectorXd& window_derivative() const {
        return window_derivative_;
    }

    const Eigen::VectorXd& window() const {
        return window_;
    }

    Eigen::Index nfft() const {
        return nfft_;
    }

private:
    static Eigen::VectorXd compute_window_derivative(const Eigen::VectorXd& window);

    void compute_fft_column(const Eigen::VectorXd& signal, const Eigen::VectorXd& win, Eigen::Index center,
                            std::complex<double>* output) const;

    Eigen::VectorXd window_;
    Eigen::VectorXd window_derivative_;
    Eigen::Index nfft_;
};

}  // namespace ssq

#endif  // SSQ_STFT_HPP
