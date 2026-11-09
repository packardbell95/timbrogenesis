#pragma once

#include "audio.hpp"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

struct Spectrogram {
    std::size_t frameCount = 0;
    std::size_t binCount = 0;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
    std::uint32_t sampleRateHz = 0;
    std::vector<double> magnitudes;
};

// A centered, reconstruction-safe STFT used by spectral transformation modes.
// Coefficients are stored frame-major, with binCount entries per frame.
struct ComplexSpectrogram {
    std::size_t frameCount = 0;
    std::size_t binCount = 0;
    std::size_t fftSize = 0;
    std::size_t hopSize = 0;
    std::uint32_t sampleRateHz = 0;
    std::size_t originalSampleCount = 0;
    std::vector<std::complex<double>> coefficients;
};

class StftAnalyzer {
public:
    // The FFTW workspace is mutable and must only be used by one thread at a
    // time.
    StftAnalyzer(std::size_t fftSize, std::size_t hopSize);
    ~StftAnalyzer();

    StftAnalyzer(const StftAnalyzer&) = delete;
    StftAnalyzer& operator=(const StftAnalyzer&) = delete;
    StftAnalyzer(StftAnalyzer&&) = delete;
    StftAnalyzer& operator=(StftAnalyzer&&) = delete;

    Spectrogram calculate(const AudioBuffer& audio);
    void calculateFrameMagnitudes(const AudioBuffer& audio,
        std::size_t frameIndex,
        std::span<double> outputMagnitudes);
    std::size_t fftSize() const noexcept;
    std::size_t hopSize() const noexcept;
    std::size_t binCount() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ComplexStftAnalyzer {
public:
    // The FFTW workspace is mutable and must only be used by one thread at a
    // time.
    ComplexStftAnalyzer(std::size_t fftSize, std::size_t hopSize);
    ~ComplexStftAnalyzer();

    ComplexStftAnalyzer(const ComplexStftAnalyzer&) = delete;
    ComplexStftAnalyzer& operator=(const ComplexStftAnalyzer&) = delete;
    ComplexStftAnalyzer(ComplexStftAnalyzer&&) = delete;
    ComplexStftAnalyzer& operator=(ComplexStftAnalyzer&&) = delete;

    ComplexSpectrogram calculate(const AudioBuffer& audio);
    std::size_t fftSize() const noexcept;
    std::size_t hopSize() const noexcept;
    std::size_t binCount() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ComplexStftSynthesizer {
public:
    // The FFTW workspace is mutable and must only be used by one thread at a
    // time.
    explicit ComplexStftSynthesizer(std::size_t fftSize);
    ~ComplexStftSynthesizer();

    ComplexStftSynthesizer(const ComplexStftSynthesizer&) = delete;
    ComplexStftSynthesizer& operator=(const ComplexStftSynthesizer&) = delete;
    ComplexStftSynthesizer(ComplexStftSynthesizer&&) = delete;
    ComplexStftSynthesizer& operator=(ComplexStftSynthesizer&&) = delete;

    AudioBuffer reconstruct(const ComplexSpectrogram& spectrogram);
    std::size_t fftSize() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

Spectrogram calculateSpectrogram(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize);

ComplexSpectrogram calculateComplexSpectrogram(
    const AudioBuffer& audio, ComplexStftAnalyzer& analyzer);

ComplexSpectrogram calculateComplexSpectrogram(
    const AudioBuffer& audio, std::size_t fftSize, std::size_t hopSize);

AudioBuffer reconstructAudio(
    const ComplexSpectrogram& spectrogram, ComplexStftSynthesizer& synthesizer);

AudioBuffer reconstructAudio(const ComplexSpectrogram& spectrogram);

double getSpectrogramMagnitude(const Spectrogram& spectrogram,
    std::size_t frameIndex,
    std::size_t binIndex);

double binToFrequencyHz(
    std::size_t binIndex, std::size_t fftSize, std::uint32_t sampleRateHz);

// Halfway values round toward the higher-frequency bin.
std::size_t frequencyToNearestBin(
    double frequencyHz, std::size_t fftSize, std::uint32_t sampleRateHz);
