#include "ssq/fftw_wrapper.hpp"

namespace ssq {

FftwManager& FftwManager::instance() {
    static FftwManager instance;
    return instance;
}

FftwManager::FftwManager() = default;

FftwManager::~FftwManager() {
    cleanup();
}

FftwPlan FftwManager::create_r2c_plan(int n, double* in, fftw_complex* out, unsigned flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    fftw_plan plan = fftw_plan_dft_r2c_1d(n, in, out, flags);
    return FftwPlan(plan);
}

FftwPlan FftwManager::create_c2r_plan(int n, fftw_complex* in, double* out, unsigned flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    fftw_plan plan = fftw_plan_dft_c2r_1d(n, in, out, flags);
    return FftwPlan(plan);
}

FftwPlan FftwManager::create_dft_plan(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    fftw_plan plan = fftw_plan_dft_1d(n, in, out, sign, flags);
    return FftwPlan(plan);
}

void FftwManager::execute(const FftwPlan& plan) {
    if (plan) {
        fftw_execute(plan.get());
    }
}

void FftwManager::cleanup() {
    std::lock_guard<std::mutex> lock(mutex_);
    fftw_cleanup();
}

}  // namespace ssq
