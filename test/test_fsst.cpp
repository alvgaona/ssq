#include <gtest/gtest.h>

#include "ssq/fsst.hpp"
#include "ssq/windows.hpp"

#include <cmath>

namespace {

const double PI = 3.14159265358979323846;

// Generate a Hann window
Eigen::VectorXd hann_window(Eigen::Index n) {
    Eigen::VectorXd w(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        w(i) = 0.5 * (1.0 - std::cos(2.0 * PI * static_cast<double>(i) / (n - 1)));
    }
    return w;
}

// Generate a Kaiser window
Eigen::VectorXd kaiser_window(Eigen::Index n, double beta) {
    Eigen::VectorXd w(n);

    // Simple approximation of Kaiser window using I0 Bessel function approximation
    auto bessel_i0 = [](double x) {
        double sum = 1.0;
        double term = 1.0;
        for (int k = 1; k < 25; ++k) {
            term *= (x / (2.0 * k)) * (x / (2.0 * k));
            sum += term;
        }
        return sum;
    };

    double alpha = static_cast<double>(n - 1) / 2.0;
    double denom = bessel_i0(PI * beta);

    for (Eigen::Index i = 0; i < n; ++i) {
        double ratio = (static_cast<double>(i) - alpha) / alpha;
        w(i) = bessel_i0(PI * beta * std::sqrt(1.0 - ratio * ratio)) / denom;
    }

    return w;
}

// Generate a sinusoid
Eigen::VectorXd sine_signal(Eigen::Index n, double freq, double sample_rate) {
    Eigen::VectorXd sig(n);
    for (Eigen::Index i = 0; i < n; ++i) {
        sig(i) = std::sin(2.0 * PI * freq * static_cast<double>(i) / sample_rate);
    }
    return sig;
}

// Generate a linear chirp (frequency sweep)
Eigen::VectorXd chirp_signal(Eigen::Index n, double f0, double f1, double sample_rate) {
    Eigen::VectorXd sig(n);
    double duration = static_cast<double>(n) / sample_rate;
    double k = (f1 - f0) / duration;  // chirp rate

    for (Eigen::Index i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        double phase = 2.0 * PI * (f0 * t + 0.5 * k * t * t);
        sig(i) = std::sin(phase);
    }
    return sig;
}

}  // namespace

class FsstTest : public ::testing::Test {
protected:
    const Eigen::Index window_size = 128;
    const double sample_rate = 1000.0;
};

TEST_F(FsstTest, OutputDimensions) {
    Eigen::VectorXd win = kaiser_window(window_size, 0.5);
    Eigen::VectorXd sig = sine_signal(500, 100.0, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);

    // Number of frequency bins: nfft/2 + 1
    Eigen::Index expected_freqs = window_size / 2 + 1;
    // Number of time steps: signal length (hop = 1)
    Eigen::Index expected_times = sig.size();

    EXPECT_EQ(result.spectrum.rows(), expected_freqs);
    EXPECT_EQ(result.spectrum.cols(), expected_times);
    EXPECT_EQ(result.frequencies.size(), expected_freqs);
    EXPECT_EQ(result.times.size(), expected_times);
}

TEST_F(FsstTest, FrequencyAxisRange) {
    Eigen::VectorXd win = kaiser_window(window_size, 0.5);
    Eigen::VectorXd sig = sine_signal(200, 50.0, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);

    // Frequency axis should go from 0 to Nyquist
    EXPECT_NEAR(result.frequencies(0), 0.0, 1e-10);
    EXPECT_NEAR(result.frequencies(result.frequencies.size() - 1), sample_rate / 2.0, 1e-10);
}

TEST_F(FsstTest, TimeAxisRange) {
    Eigen::VectorXd win = kaiser_window(window_size, 0.5);
    Eigen::Index sig_len = 300;
    Eigen::VectorXd sig = sine_signal(sig_len, 50.0, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);

    // Time axis should start at 0
    EXPECT_NEAR(result.times(0), 0.0, 1e-10);

    // Last time point should be (sig_len - 1) / sample_rate
    EXPECT_NEAR(result.times(result.times.size() - 1), static_cast<double>(sig_len - 1) / sample_rate, 1e-10);
}

TEST_F(FsstTest, SinusoidConcentration) {
    // FSST should concentrate sinusoid energy at its frequency
    double test_freq = 100.0;
    Eigen::VectorXd win = hann_window(window_size);
    Eigen::VectorXd sig = sine_signal(500, test_freq, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);

    // Check a time point in the middle
    Eigen::Index t_mid = sig.size() / 2;

    // Find the frequency bin with maximum magnitude
    Eigen::Index max_bin = 0;
    double max_mag = 0.0;
    for (Eigen::Index k = 0; k < result.spectrum.rows(); ++k) {
        double mag = std::abs(result.spectrum(k, t_mid));
        if (mag > max_mag) {
            max_mag = mag;
            max_bin = k;
        }
    }

    // Peak should be at or very near the expected frequency
    double peak_freq = result.frequencies(max_bin);
    double df = sample_rate / static_cast<double>(window_size);
    EXPECT_NEAR(peak_freq, test_freq, df) << "FSST peak should be at " << test_freq << " Hz";

    // Energy should be concentrated (synchrosqueezing effect)
    // Count bins with significant energy (> 10% of max)
    int significant_bins = 0;
    for (Eigen::Index k = 0; k < result.spectrum.rows(); ++k) {
        double mag = std::abs(result.spectrum(k, t_mid));
        if (mag > max_mag * 0.1) {
            ++significant_bins;
        }
    }

    // FSST should concentrate energy into fewer bins compared to raw STFT
    EXPECT_LE(significant_bins, 5) << "FSST should concentrate sinusoid energy into few bins";
}

TEST_F(FsstTest, ChirpTracking) {
    // FSST should track the instantaneous frequency of a chirp
    double f0 = 50.0;
    double f1 = 150.0;
    Eigen::Index sig_len = 1000;
    Eigen::VectorXd win = hann_window(window_size);
    Eigen::VectorXd sig = chirp_signal(sig_len, f0, f1, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);

    // Check at several time points
    std::vector<Eigen::Index> test_times = {200, 400, 600, 800};
    double duration = static_cast<double>(sig_len) / sample_rate;

    for (Eigen::Index t : test_times) {
        // Expected instantaneous frequency at time t
        double time = static_cast<double>(t) / sample_rate;
        double expected_freq = f0 + (f1 - f0) * time / duration;

        // Find peak frequency at this time
        Eigen::Index max_bin = 0;
        double max_mag = 0.0;
        for (Eigen::Index k = 0; k < result.spectrum.rows(); ++k) {
            double mag = std::abs(result.spectrum(k, t));
            if (mag > max_mag) {
                max_mag = mag;
                max_bin = k;
            }
        }

        double peak_freq = result.frequencies(max_bin);

        // Allow tolerance of 2 frequency bins
        double df = sample_rate / static_cast<double>(window_size);
        EXPECT_NEAR(peak_freq, expected_freq, 2 * df) << "FSST should track chirp frequency at t=" << time;
    }
}

TEST_F(FsstTest, KaiserWindowConfiguration) {
    // Test with the exact configuration used in the project
    Eigen::VectorXd win = kaiser_window(128, 0.5);
    double fs = 1000.0;

    Eigen::VectorXd sig = sine_signal(2000, 100.0, fs);

    auto result = ssq::fsst(sig, fs, win);

    // Basic sanity checks
    EXPECT_EQ(result.frequencies.size(), 65);  // 128/2 + 1
    EXPECT_EQ(result.times.size(), 2000);
    EXPECT_NEAR(result.frequencies(0), 0.0, 1e-10);
    EXPECT_NEAR(result.frequencies(result.frequencies.size() - 1), 500.0, 1e-10);  // Nyquist = 1000/2
}

TEST_F(FsstTest, InverseReconstruction) {
    // ifsst should perfectly reconstruct sinusoids
    Eigen::VectorXd win = ssq::windows::gaussian(window_size);
    Eigen::VectorXd sig = sine_signal(1000, 50.0, sample_rate);

    auto result = ssq::fsst(sig, sample_rate, win);
    Eigen::VectorXd reconstructed = ssq::ifsst(result.spectrum, win);

    // Compare in the middle to avoid edge effects
    Eigen::Index mid_start = sig.size() / 4;
    Eigen::Index mid_end = 3 * sig.size() / 4;

    for (Eigen::Index i = mid_start; i < mid_end; ++i) {
        EXPECT_NEAR(reconstructed(i), sig(i), 1e-6)
            << "Reconstruction mismatch at index " << i;
    }
}

TEST_F(FsstTest, InverseReconstructionMultipleFrequencies) {
    // Test reconstruction with multiple frequency components
    Eigen::VectorXd win = ssq::windows::hann(window_size);
    Eigen::Index n = 1000;
    Eigen::VectorXd sig(n);

    for (Eigen::Index i = 0; i < n; ++i) {
        double t = static_cast<double>(i) / sample_rate;
        sig(i) = std::sin(2.0 * PI * 50.0 * t) + 0.5 * std::sin(2.0 * PI * 120.0 * t);
    }

    auto result = ssq::fsst(sig, sample_rate, win);
    Eigen::VectorXd reconstructed = ssq::ifsst(result.spectrum, win);

    // Check RMS error in the middle region
    double error_sum = 0.0;
    Eigen::Index mid_start = n / 4;
    Eigen::Index mid_end = 3 * n / 4;
    Eigen::Index count = mid_end - mid_start;

    for (Eigen::Index i = mid_start; i < mid_end; ++i) {
        double diff = reconstructed(i) - sig(i);
        error_sum += diff * diff;
    }

    double rms_error = std::sqrt(error_sum / count);
    EXPECT_LT(rms_error, 1e-5) << "RMS reconstruction error should be small";
}

TEST_F(FsstTest, InverseWithDifferentWindows) {
    // Test that all window types produce good reconstruction
    Eigen::VectorXd sig = sine_signal(800, 75.0, sample_rate);

    std::vector<Eigen::VectorXd> windows = {
        ssq::windows::kaiser(window_size),
        ssq::windows::hamming(window_size),
        ssq::windows::hann(window_size),
        ssq::windows::gaussian(window_size)
    };

    for (const auto& win : windows) {
        auto result = ssq::fsst(sig, sample_rate, win);
        Eigen::VectorXd reconstructed = ssq::ifsst(result.spectrum, win);

        // Check middle region
        Eigen::Index mid = sig.size() / 2;
        double error = std::abs(reconstructed(mid) - sig(mid));
        EXPECT_LT(error, 1e-5) << "Reconstruction should be accurate with any window";
    }
}
