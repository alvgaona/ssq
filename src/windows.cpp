#include "ssq/windows.hpp"

#include <cmath>

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

    Eigen::VectorXd w(length);
    const double N = static_cast<double>(length - 1);
    const double i0_beta = bessel_i0(beta);

    for (Eigen::Index n = 0; n < length; ++n) {
        double ratio = (2.0 * n / N) - 1.0;
        double arg = beta * std::sqrt(1.0 - ratio * ratio);
        w(n) = bessel_i0(arg) / i0_beta;
    }

    return w;
}

Eigen::VectorXd hamming(Eigen::Index length) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    Eigen::VectorXd w(length);
    const double N = static_cast<double>(length - 1);

    for (Eigen::Index n = 0; n < length; ++n) {
        w(n) = 0.54 - 0.46 * std::cos(TWO_PI * n / N);
    }

    return w;
}

Eigen::VectorXd hann(Eigen::Index length) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    Eigen::VectorXd w(length);
    const double N = static_cast<double>(length - 1);

    for (Eigen::Index n = 0; n < length; ++n) {
        w(n) = 0.5 * (1.0 - std::cos(TWO_PI * n / N));
    }

    return w;
}

Eigen::VectorXd gaussian(Eigen::Index length, double sigma) {
    if (length <= 0) return Eigen::VectorXd();
    if (length == 1) return Eigen::VectorXd::Ones(1);

    // Default sigma: length/6 (window decays to ~0.01 at edges)
    if (sigma <= 0.0) {
        sigma = static_cast<double>(length) / 6.0;
    }

    Eigen::VectorXd w(length);
    const double center = static_cast<double>(length - 1) / 2.0;

    for (Eigen::Index n = 0; n < length; ++n) {
        double x = (n - center) / sigma;
        w(n) = std::exp(-0.5 * x * x);
    }

    return w;
}

}  // namespace windows
}  // namespace ssq
