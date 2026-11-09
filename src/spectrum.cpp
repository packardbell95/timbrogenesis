#include "spectrum.hpp"

#include <fftw3.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

struct FftwBufferDeleter {
    void operator()(void* buffer) const noexcept
    {
        fftw_free(buffer);
    }
};

struct FftwPlanDeleter {
    void operator()(fftw_plan plan) const noexcept
    {
        if (plan != nullptr) {
            fftw_destroy_plan(plan);
        }
    }
};

using FftwInput = std::unique_ptr<double, FftwBufferDeleter>;
using FftwOutput = std::unique_ptr<fftw_complex, FftwBufferDeleter>;
using FftwPlan =
    std::unique_ptr<std::remove_pointer_t<fftw_plan>, FftwPlanDeleter>;

void validateStftSettings(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize)
{
    if (fftSize == 0) {
        throw std::invalid_argument("FFT size must be greater than zero.");
    }
    if (hopSize == 0) {
        throw std::invalid_argument("Hop size must be greater than zero.");
    }
    if (hopSize > fftSize) {
        throw std::invalid_argument("Hop size cannot exceed FFT size.");
    }
    if (audio.sampleRateHz == 0) {
        throw std::invalid_argument(
            "Audio sample rate must be greater than zero.");
    }
    if (fftSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(
            "FFT size exceeds the range supported by FFTW.");
    }
}

void validateReconstructibleStftSettings(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize)
{
    validateStftSettings(audio, fftSize, hopSize);
    // A symmetric Hann window is zero at both ends. Adjacent frames must place
    // a nonzero window sample at every position that can contain source audio.
    if (fftSize == 2 || (fftSize > 2 && hopSize > fftSize - 2)) {
        throw std::invalid_argument(
            "Hop size leaves samples uncovered by the Hann window.");
    }
}

void validateFrequencyMapping(std::size_t fftSize, std::uint32_t sampleRateHz)
{
    if (fftSize == 0) {
        throw std::invalid_argument("FFT size must be greater than zero.");
    }
    if (sampleRateHz == 0) {
        throw std::invalid_argument(
            "Audio sample rate must be greater than zero.");
    }
}

std::vector<double> createHannWindow(std::size_t fftSize)
{
    std::vector<double> window(fftSize, 1.0);
    if (fftSize == 1) {
        return window;
    }

    const double denominator = static_cast<double>(fftSize - 1);
    for (std::size_t index = 0; index < fftSize; ++index) {
        const double position = static_cast<double>(index) / denominator;
        window[index] =
            0.5 * (1.0 - std::cos(2.0 * std::numbers::pi * position));
    }
    return window;
}

std::size_t checkedMagnitudeCount(std::size_t frameCount, std::size_t binCount)
{
    if (frameCount != 0
        && binCount > std::numeric_limits<std::size_t>::max() / frameCount) {
        throw std::runtime_error("Spectrogram dimensions are too large.");
    }
    return frameCount * binCount;
}

std::size_t centeredFrameCount(
    std::size_t sampleCount, std::size_t fftSize, std::size_t hopSize)
{
    if (sampleCount == 0)
        return 0;
    const std::size_t padding = fftSize / 2;
    if (sampleCount > std::numeric_limits<std::size_t>::max() - 2 * padding)
        throw std::runtime_error(
            "Audio is too large for centered STFT padding.");
    const std::size_t paddedCount = sampleCount + 2 * padding;
    if (paddedCount <= fftSize)
        return 1;
    const std::size_t remaining = paddedCount - fftSize;
    return 1 + remaining / hopSize + (remaining % hopSize != 0 ? 1U : 0U);
}

std::size_t checkedCoefficientCount(
    std::size_t frameCount, std::size_t binCount)
{
    return checkedMagnitudeCount(frameCount, binCount);
}

}  // namespace

struct StftAnalyzer::Impl {
    std::size_t fftSize;
    std::size_t hopSize;
    std::size_t binCount;
    std::vector<double> window;
    FftwInput input;
    FftwOutput output;
    FftwPlan plan;

    Impl(std::size_t fftSizeValue, std::size_t hopSizeValue)
        : fftSize(fftSizeValue), hopSize(hopSizeValue),
          binCount(fftSizeValue / 2 + 1),
          window(createHannWindow(fftSizeValue)),
          input(
              static_cast<double*>(fftw_malloc(sizeof(double) * fftSizeValue))),
          output(static_cast<fftw_complex*>(
              fftw_malloc(sizeof(fftw_complex) * binCount))),
          plan(nullptr)
    {
        AudioBuffer dummy{{}, 1};
        validateStftSettings(dummy, fftSize, hopSize);
        if (!input || !output) {
            throw std::runtime_error("Failed to allocate FFTW workspace.");
        }
        plan.reset(fftw_plan_dft_r2c_1d(static_cast<int>(fftSize),
            input.get(),
            output.get(),
            FFTW_ESTIMATE));
        if (!plan) {
            throw std::runtime_error(
                "Failed to create FFTW real-to-complex plan.");
        }
    }
};

StftAnalyzer::StftAnalyzer(std::size_t fftSize, std::size_t hopSize)
    : impl_(std::make_unique<Impl>(fftSize, hopSize))
{
}

StftAnalyzer::~StftAnalyzer() = default;

std::size_t StftAnalyzer::fftSize() const noexcept
{
    return impl_->fftSize;
}
std::size_t StftAnalyzer::hopSize() const noexcept
{
    return impl_->hopSize;
}
std::size_t StftAnalyzer::binCount() const noexcept
{
    return impl_->binCount;
}

void StftAnalyzer::calculateFrameMagnitudes(const AudioBuffer& audio,
    std::size_t frameIndex,
    std::span<double> outputMagnitudes)
{
    validateStftSettings(audio, impl_->fftSize, impl_->hopSize);
    const std::size_t frameCount = audio.samples.empty()
        ? 0
        : 1 + (audio.samples.size() - 1) / impl_->hopSize;
    if (frameIndex >= frameCount) {
        throw std::out_of_range("STFT frame index is out of range.");
    }
    if (outputMagnitudes.size() != impl_->binCount) {
        throw std::invalid_argument("STFT frame output has the wrong size.");
    }
    const std::size_t frameStart = frameIndex * impl_->hopSize;
    const std::size_t copied =
        std::min(impl_->fftSize, audio.samples.size() - frameStart);
    std::fill_n(impl_->input.get(), impl_->fftSize, 0.0);
    for (std::size_t index = 0; index < copied; ++index) {
        impl_->input.get()[index] =
            audio.samples[frameStart + index] * impl_->window[index];
    }
    fftw_execute(impl_->plan.get());
    for (std::size_t bin = 0; bin < impl_->binCount; ++bin) {
        outputMagnitudes[bin] = std::hypot(
            impl_->output.get()[bin][0], impl_->output.get()[bin][1]);
    }
}

Spectrogram StftAnalyzer::calculate(const AudioBuffer& audio)
{
    validateStftSettings(audio, impl_->fftSize, impl_->hopSize);
    const std::size_t frames = audio.samples.empty()
        ? 0
        : 1 + (audio.samples.size() - 1) / impl_->hopSize;
    Spectrogram result{
        .frameCount = frames,
        .binCount = impl_->binCount,
        .fftSize = impl_->fftSize,
        .hopSize = impl_->hopSize,
        .sampleRateHz = audio.sampleRateHz,
        .magnitudes =
            std::vector<double>(checkedMagnitudeCount(frames, impl_->binCount)),
    };
    for (std::size_t frame = 0; frame < frames; ++frame) {
        calculateFrameMagnitudes(audio,
            frame,
            std::span<double>(result.magnitudes)
                .subspan(frame * impl_->binCount, impl_->binCount));
    }
    return result;
}

struct ComplexStftAnalyzer::Impl {
    std::size_t fftSize;
    std::size_t hopSize;
    std::size_t binCount;
    std::vector<double> window;
    FftwInput input;
    FftwOutput output;
    FftwPlan plan;

    Impl(std::size_t fftSizeValue, std::size_t hopSizeValue)
        : fftSize(fftSizeValue), hopSize(hopSizeValue),
          binCount(fftSizeValue / 2 + 1),
          window(createHannWindow(fftSizeValue)),
          input(
              static_cast<double*>(fftw_malloc(sizeof(double) * fftSizeValue))),
          output(static_cast<fftw_complex*>(
              fftw_malloc(sizeof(fftw_complex) * binCount))),
          plan(nullptr)
    {
        AudioBuffer dummy{{}, 1};
        validateReconstructibleStftSettings(dummy, fftSize, hopSize);
        if (!input || !output)
            throw std::runtime_error(
                "Failed to allocate complex FFTW workspace.");
        plan.reset(fftw_plan_dft_r2c_1d(static_cast<int>(fftSize),
            input.get(),
            output.get(),
            FFTW_ESTIMATE));
        if (!plan)
            throw std::runtime_error(
                "Failed to create complex STFT analysis plan.");
    }
};

ComplexStftAnalyzer::ComplexStftAnalyzer(
    std::size_t fftSize, std::size_t hopSize)
    : impl_(std::make_unique<Impl>(fftSize, hopSize))
{
}

ComplexStftAnalyzer::~ComplexStftAnalyzer() = default;

std::size_t ComplexStftAnalyzer::fftSize() const noexcept
{
    return impl_->fftSize;
}
std::size_t ComplexStftAnalyzer::hopSize() const noexcept
{
    return impl_->hopSize;
}
std::size_t ComplexStftAnalyzer::binCount() const noexcept
{
    return impl_->binCount;
}

ComplexSpectrogram ComplexStftAnalyzer::calculate(const AudioBuffer& audio)
{
    validateReconstructibleStftSettings(audio, impl_->fftSize, impl_->hopSize);
    const std::size_t frames = centeredFrameCount(
        audio.samples.size(), impl_->fftSize, impl_->hopSize);
    ComplexSpectrogram result{
        .frameCount = frames,
        .binCount = impl_->binCount,
        .fftSize = impl_->fftSize,
        .hopSize = impl_->hopSize,
        .sampleRateHz = audio.sampleRateHz,
        .originalSampleCount = audio.samples.size(),
        .coefficients = std::vector<std::complex<double>>(
            checkedCoefficientCount(frames, impl_->binCount)),
    };
    const std::size_t leftPadding = impl_->fftSize / 2;
    for (std::size_t frame = 0; frame < frames; ++frame) {
        std::fill_n(impl_->input.get(), impl_->fftSize, 0.0);
        const std::size_t frameStart = frame * impl_->hopSize;
        for (std::size_t index = 0; index < impl_->fftSize; ++index) {
            const std::size_t paddedIndex = frameStart + index;
            if (paddedIndex >= leftPadding) {
                const std::size_t sourceIndex = paddedIndex - leftPadding;
                if (sourceIndex < audio.samples.size()) {
                    impl_->input.get()[index] =
                        audio.samples[sourceIndex] * impl_->window[index];
                }
            }
        }
        fftw_execute(impl_->plan.get());
        for (std::size_t bin = 0; bin < impl_->binCount; ++bin) {
            result.coefficients[frame * impl_->binCount + bin] = {
                impl_->output.get()[bin][0], impl_->output.get()[bin][1]};
        }
    }
    return result;
}

struct ComplexStftSynthesizer::Impl {
    std::size_t fftSize;
    std::size_t binCount;
    std::vector<double> window;
    FftwOutput input;
    FftwInput output;
    FftwPlan plan;

    explicit Impl(std::size_t fftSizeValue)
        : fftSize(fftSizeValue), binCount(fftSizeValue / 2 + 1),
          window(createHannWindow(fftSizeValue)),
          input(static_cast<fftw_complex*>(
              fftw_malloc(sizeof(fftw_complex) * binCount))),
          output(
              static_cast<double*>(fftw_malloc(sizeof(double) * fftSizeValue))),
          plan(nullptr)
    {
        AudioBuffer dummy{{}, 1};
        validateStftSettings(dummy, fftSize, 1);
        if (!input || !output)
            throw std::runtime_error(
                "Failed to allocate inverse FFTW workspace.");
        plan.reset(fftw_plan_dft_c2r_1d(static_cast<int>(fftSize),
            input.get(),
            output.get(),
            FFTW_ESTIMATE));
        if (!plan)
            throw std::runtime_error(
                "Failed to create complex STFT synthesis plan.");
    }
};

ComplexStftSynthesizer::ComplexStftSynthesizer(std::size_t fftSize)
    : impl_(std::make_unique<Impl>(fftSize))
{
}

ComplexStftSynthesizer::~ComplexStftSynthesizer() = default;

std::size_t ComplexStftSynthesizer::fftSize() const noexcept
{
    return impl_->fftSize;
}

AudioBuffer ComplexStftSynthesizer::reconstruct(
    const ComplexSpectrogram& spectrogram)
{
    if (spectrogram.fftSize != impl_->fftSize)
        throw std::invalid_argument(
            "Spectrogram FFT size does not match synthesizer.");
    if (spectrogram.sampleRateHz == 0)
        throw std::invalid_argument(
            "Complex spectrogram has invalid sample rate.");
    const AudioBuffer validationAudio{{}, spectrogram.sampleRateHz};
    validateReconstructibleStftSettings(
        validationAudio, spectrogram.fftSize, spectrogram.hopSize);
    if (spectrogram.binCount != impl_->binCount)
        throw std::invalid_argument(
            "Complex spectrogram has invalid bin count.");
    if (spectrogram.coefficients.size()
        != checkedCoefficientCount(
            spectrogram.frameCount, spectrogram.binCount))
        throw std::invalid_argument(
            "Complex spectrogram coefficient count is invalid.");
    if (spectrogram.originalSampleCount == 0) {
        if (spectrogram.frameCount != 0)
            throw std::invalid_argument(
                "Empty complex spectrogram has frames.");
        return {{}, spectrogram.sampleRateHz};
    }
    const std::size_t expectedFrames =
        centeredFrameCount(spectrogram.originalSampleCount,
            spectrogram.fftSize,
            spectrogram.hopSize);
    if (spectrogram.frameCount != expectedFrames)
        throw std::invalid_argument(
            "Complex spectrogram frame count is invalid.");
    if (spectrogram.frameCount - 1
        > (std::numeric_limits<std::size_t>::max() - spectrogram.fftSize)
            / spectrogram.hopSize)
        throw std::runtime_error(
            "Reconstructed audio dimensions are too large.");
    const std::size_t paddedOutputCount = spectrogram.fftSize
        + (spectrogram.frameCount - 1) * spectrogram.hopSize;
    std::vector<double> output(paddedOutputCount, 0.0);
    std::vector<double> overlapWeights(paddedOutputCount, 0.0);
    const double inverseScale = 1.0 / static_cast<double>(spectrogram.fftSize);

    for (std::size_t frame = 0; frame < spectrogram.frameCount; ++frame) {
        for (std::size_t bin = 0; bin < impl_->binCount; ++bin) {
            const std::complex<double> value =
                spectrogram.coefficients[frame * impl_->binCount + bin];
            impl_->input.get()[bin][0] = value.real();
            impl_->input.get()[bin][1] = value.imag();
        }
        fftw_execute(impl_->plan.get());
        const std::size_t frameStart = frame * spectrogram.hopSize;
        for (std::size_t index = 0; index < impl_->fftSize; ++index) {
            const double window = impl_->window[index];
            output[frameStart + index] +=
                impl_->output.get()[index] * inverseScale * window;
            overlapWeights[frameStart + index] += window * window;
        }
    }
    constexpr double weightEpsilon = 1e-15;
    for (std::size_t index = 0; index < output.size(); ++index) {
        if (overlapWeights[index] > weightEpsilon)
            output[index] /= overlapWeights[index];
    }

    const std::size_t leftPadding = spectrogram.fftSize / 2;
    AudioBuffer result;
    result.sampleRateHz = spectrogram.sampleRateHz;
    result.samples.assign(
        output.begin() + static_cast<std::ptrdiff_t>(leftPadding),
        output.begin()
            + static_cast<std::ptrdiff_t>(
                leftPadding + spectrogram.originalSampleCount));
    return result;
}

Spectrogram calculateSpectrogram(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize)
{
    StftAnalyzer analyzer(fftSize, hopSize);
    return analyzer.calculate(audio);
}

ComplexSpectrogram calculateComplexSpectrogram(
    const AudioBuffer& audio, ComplexStftAnalyzer& analyzer)
{
    return analyzer.calculate(audio);
}

ComplexSpectrogram calculateComplexSpectrogram(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize)
{
    ComplexStftAnalyzer analyzer(fftSize, hopSize);
    return analyzer.calculate(audio);
}

AudioBuffer reconstructAudio(
    const ComplexSpectrogram& spectrogram, ComplexStftSynthesizer& synthesizer)
{
    return synthesizer.reconstruct(spectrogram);
}

AudioBuffer reconstructAudio(const ComplexSpectrogram& spectrogram)
{
    ComplexStftSynthesizer synthesizer(spectrogram.fftSize);
    return synthesizer.reconstruct(spectrogram);
}

double getSpectrogramMagnitude(const Spectrogram& spectrogram,
    std::size_t frameIndex,
    std::size_t binIndex)
{
    if (frameIndex >= spectrogram.frameCount
        || binIndex >= spectrogram.binCount) {
        throw std::out_of_range("Spectrogram index is out of range.");
    }
    if (spectrogram.binCount != 0
        && frameIndex > (std::numeric_limits<std::size_t>::max() - binIndex)
                / spectrogram.binCount) {
        throw std::out_of_range("Spectrogram index is out of range.");
    }

    return spectrogram.magnitudes.at(
        frameIndex * spectrogram.binCount + binIndex);
}

double binToFrequencyHz(
    std::size_t binIndex, std::size_t fftSize, std::uint32_t sampleRateHz)
{
    validateFrequencyMapping(fftSize, sampleRateHz);
    if (binIndex >= fftSize / 2 + 1) {
        throw std::out_of_range("FFT bin index is out of range.");
    }

    return static_cast<double>(binIndex) * static_cast<double>(sampleRateHz)
        / static_cast<double>(fftSize);
}

std::size_t frequencyToNearestBin(
    double frequencyHz, std::size_t fftSize, std::uint32_t sampleRateHz)
{
    validateFrequencyMapping(fftSize, sampleRateHz);
    const double nyquistFrequencyHz = static_cast<double>(sampleRateHz) / 2.0;
    if (!std::isfinite(frequencyHz) || frequencyHz < 0.0
        || frequencyHz > nyquistFrequencyHz) {
        throw std::invalid_argument(
            "Frequency must be finite and between zero and Nyquist.");
    }

    const double bin = frequencyHz * static_cast<double>(fftSize)
        / static_cast<double>(sampleRateHz);
    return static_cast<std::size_t>(std::round(bin));
}
