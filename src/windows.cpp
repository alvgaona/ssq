#include "ssq/windows.hpp"

namespace ssq {
namespace windows {

namespace {
const double PI = 3.14159265358979323846;
const double TWO_PI = 2.0 * PI;

// Modified Bessel function I0 (series expansion)
double bessel_i0(double x) {
    double sum = 1.0;
    double term = 1.0;
    double x_half = x / 2.0;
    for (int k = 1; k < 50; ++k) {
        term *= (x_half / k) * (x_half / k);
        sum += term;
        if (term < 1e-15 * sum) break;
    }
    return sum;
}
}  // namespace

Eigen::VectorXd kaiser(Eigen::Index length, double beta) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    const double N = static_cast<double>(length - 1);
    const double i0_beta = bessel_i0(beta);

    // Vectorized ratio computation
    Eigen::VectorXd n = Eigen::VectorXd::LinSpaced(length, 0.0, N);
    Eigen::ArrayXd ratio = (2.0 * n.array() / N) - 1.0;
    Eigen::ArrayXd arg = beta * (1.0 - ratio.square()).sqrt();

    // bessel_i0 must be called element-wise (no vectorized version)
    Eigen::VectorXd w(length);
    for (Eigen::Index i = 0; i < length; ++i) {
        w(i) = bessel_i0(arg(i)) / i0_beta;
    }

    return w;
}

Eigen::VectorXd hamming(Eigen::Index length) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    const double N = static_cast<double>(length - 1);
    Eigen::VectorXd n = Eigen::VectorXd::LinSpaced(length, 0.0, N);
    return 0.54 - 0.46 * (TWO_PI * n / N).array().cos();
}

Eigen::VectorXd hann(Eigen::Index length) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    const double N = static_cast<double>(length - 1);
    Eigen::VectorXd n = Eigen::VectorXd::LinSpaced(length, 0.0, N);
    return 0.5 * (1.0 - (TWO_PI * n / N).array().cos());
}

Eigen::VectorXd gaussian(Eigen::Index length, double sigma) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    // Default sigma: length/6 (window decays to ~0.01 at edges)
    if (sigma <= 0.0) {
        sigma = static_cast<double>(length) / 6.0;
    }

    const double center = static_cast<double>(length - 1) / 2.0;
    Eigen::VectorXd n = Eigen::VectorXd::LinSpaced(length, 0.0, static_cast<double>(length - 1));
    Eigen::ArrayXd x = (n.array() - center) / sigma;
    return (-0.5 * x.square()).exp();
}

}  // namespace windows
}  // namespace ssq
