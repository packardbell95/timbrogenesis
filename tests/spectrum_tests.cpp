#include "spectrum.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

constexpr double tolerance = 1e-12;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireInvalidArgument(Function function, const std::string& testName)
{
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(
        testName + " did not throw std::invalid_argument.");
}

template <typename Function>
void requireOutOfRange(Function function, const std::string& testName)
{
    try {
        function();
    } catch (const std::out_of_range&) {
        return;
    }
    throw std::runtime_error(testName + " did not throw std::out_of_range.");
}

AudioBuffer createSignal(std::size_t sampleCount,
    std::uint32_t sampleRateHz,
    double firstFrequencyHz = 0.0,
    double firstAmplitude = 0.0,
    double secondFrequencyHz = 0.0,
    double secondAmplitude = 0.0)
{
    AudioBuffer audio;
    audio.sampleRateHz = sampleRateHz;
    audio.samples.resize(sampleCount);

    for (std::size_t sampleIndex = 0; sampleIndex < sampleCount;
         ++sampleIndex) {
        const double timeSeconds = static_cast<double>(sampleIndex)
            / static_cast<double>(sampleRateHz);
        audio.samples[sampleIndex] = firstAmplitude
                * std::sin(
                    2.0 * std::numbers::pi * firstFrequencyHz * timeSeconds)
            + secondAmplitude
                * std::sin(
                    2.0 * std::numbers::pi * secondFrequencyHz * timeSeconds);
    }
    return audio;
}

void requireRoundTrip(const AudioBuffer& original,
    std::size_t fftSize,
    std::size_t hopSize,
    double maximumTolerance = 1e-11,
    double rmsTolerance = 1e-12)
{
    const ComplexSpectrogram spectrum =
        calculateComplexSpectrogram(original, fftSize, hopSize);
    const AudioBuffer reconstructed = reconstructAudio(spectrum);
    require(reconstructed.sampleRateHz == original.sampleRateHz,
        "Complex STFT round trip changed the sample rate.");
    require(reconstructed.samples.size() == original.samples.size(),
        "Complex STFT round trip changed the sample count.");

    double maximumError = 0.0;
    double squaredError = 0.0;
    for (std::size_t index = 0; index < original.samples.size(); ++index) {
        const double error =
            std::abs(reconstructed.samples[index] - original.samples[index]);
        maximumError = std::max(maximumError, error);
        squaredError += error * error;
    }
    const double rmsError = original.samples.empty()
        ? 0.0
        : std::sqrt(
              squaredError / static_cast<double>(original.samples.size()));
    require(maximumError < maximumTolerance,
        "Complex STFT maximum round-trip error exceeded tolerance.");
    require(rmsError < rmsTolerance,
        "Complex STFT RMS round-trip error exceeded tolerance.");
}

std::size_t strongestBin(const Spectrogram& spectrogram,
    std::size_t frameIndex,
    std::size_t firstBin = 0)
{
    std::size_t strongest = firstBin;
    for (std::size_t binIndex = firstBin + 1; binIndex < spectrogram.binCount;
         ++binIndex) {
        if (getSpectrogramMagnitude(spectrogram, frameIndex, binIndex)
            > getSpectrogramMagnitude(spectrogram, frameIndex, strongest)) {
            strongest = binIndex;
        }
    }
    return strongest;
}

void testEmptyAndShortBuffers()
{
    const AudioBuffer empty{{}, 48'000};
    const Spectrogram emptySpectrogram = calculateSpectrogram(empty, 8, 4);
    require(emptySpectrogram.frameCount == 0,
        "Empty audio produced an STFT frame.");
    require(emptySpectrogram.binCount == 5,
        "Empty audio produced the wrong bin count.");
    require(emptySpectrogram.magnitudes.empty(),
        "Empty audio produced magnitudes.");

    const AudioBuffer shortAudio{{1.0, 0.0, -1.0}, 48'000};
    const Spectrogram shortSpectrogram = calculateSpectrogram(shortAudio, 8, 4);
    require(shortSpectrogram.frameCount == 1,
        "Audio shorter than one hop should produce one frame.");
    require(shortSpectrogram.magnitudes.size() == 5,
        "Short audio produced the wrong magnitude count.");
    for (double magnitude : shortSpectrogram.magnitudes) {
        require(std::isfinite(magnitude),
            "Zero-padding produced a non-finite magnitude.");
    }
}

void testFramePlacement()
{
    constexpr std::size_t fftSize = 8;
    constexpr std::size_t hopSize = 4;
    const AudioBuffer exactFrame = createSignal(fftSize, 48'000);
    const Spectrogram exactSpectrogram =
        calculateSpectrogram(exactFrame, fftSize, hopSize);
    require(exactSpectrogram.frameCount == 2,
        "An FFT-sized signal did not follow hop-based frame placement.");

    for (const std::size_t sampleCount : {0U, 1U, 3U, 4U, 5U, 8U, 9U, 17U}) {
        const AudioBuffer audio = createSignal(sampleCount, 48'000);
        const Spectrogram spectrogram =
            calculateSpectrogram(audio, fftSize, hopSize);
        const std::size_t expectedFrames =
            sampleCount == 0 ? 0 : 1 + (sampleCount - 1) / hopSize;
        require(spectrogram.frameCount == expectedFrames,
            "Frame count does not match the specified formula.");
    }

    AudioBuffer partial = createSignal(10, 48'000);
    std::fill(partial.samples.begin(), partial.samples.end(), 1.0);
    const Spectrogram partialSpectrogram = calculateSpectrogram(partial, 8, 6);
    require(partialSpectrogram.frameCount == 2,
        "A partial final frame was not included.");
    require(getSpectrogramMagnitude(partialSpectrogram, 1, 0) > 0.0,
        "A partial final frame was not populated before zero-padding.");
}

void testSilenceAndDc()
{
    const AudioBuffer silence = createSignal(64, 64);
    const Spectrogram silenceSpectrogram =
        calculateSpectrogram(silence, 64, 64);
    for (double magnitude : silenceSpectrogram.magnitudes) {
        require(std::abs(magnitude) < tolerance,
            "Silence produced nonzero spectral energy.");
    }

    AudioBuffer dc = createSignal(64, 64);
    std::fill(dc.samples.begin(), dc.samples.end(), 0.75);
    const Spectrogram dcSpectrogram = calculateSpectrogram(dc, 64, 64);
    require(strongestBin(dcSpectrogram, 0) == 0,
        "A DC signal did not peak at bin zero.");
}

void testAlignedAndUnalignedSines()
{
    constexpr std::uint32_t sampleRateHz = 4096;
    constexpr std::size_t fftSize = 256;
    constexpr std::size_t targetBin = 16;
    const double alignedFrequencyHz =
        binToFrequencyHz(targetBin, fftSize, sampleRateHz);
    const AudioBuffer aligned =
        createSignal(fftSize, sampleRateHz, alignedFrequencyHz, 1.0);
    const Spectrogram alignedSpectrogram =
        calculateSpectrogram(aligned, fftSize, fftSize);
    require(strongestBin(alignedSpectrogram, 0, 1) == targetBin,
        "An aligned sine did not peak at its expected bin.");

    const double unalignedFrequencyHz =
        binToFrequencyHz(targetBin, fftSize, sampleRateHz)
        + 0.5 * static_cast<double>(sampleRateHz)
            / static_cast<double>(fftSize);
    const AudioBuffer unaligned =
        createSignal(fftSize, sampleRateHz, unalignedFrequencyHz, 1.0);
    const Spectrogram unalignedSpectrogram =
        calculateSpectrogram(unaligned, fftSize, fftSize);
    const double lowerMagnitude =
        getSpectrogramMagnitude(unalignedSpectrogram, 0, targetBin);
    const double upperMagnitude =
        getSpectrogramMagnitude(unalignedSpectrogram, 0, targetBin + 1);
    const double distantMagnitude =
        getSpectrogramMagnitude(unalignedSpectrogram, 0, targetBin + 12);
    require(lowerMagnitude > distantMagnitude * 20.0,
        "An unaligned sine lacked energy in the lower neighboring bin.");
    require(upperMagnitude > distantMagnitude * 20.0,
        "An unaligned sine lacked energy in the upper neighboring bin.");
}

void testTwoFrequencyMixture()
{
    constexpr std::uint32_t sampleRateHz = 4096;
    constexpr std::size_t fftSize = 256;
    constexpr std::size_t firstBin = 12;
    constexpr std::size_t secondBin = 35;
    const AudioBuffer mixture = createSignal(fftSize,
        sampleRateHz,
        binToFrequencyHz(firstBin, fftSize, sampleRateHz),
        1.0,
        binToFrequencyHz(secondBin, fftSize, sampleRateHz),
        0.8);
    const Spectrogram spectrogram =
        calculateSpectrogram(mixture, fftSize, fftSize);

    const double distantMagnitude = getSpectrogramMagnitude(spectrogram, 0, 60);
    require(getSpectrogramMagnitude(spectrogram, 0, firstBin)
            > distantMagnitude * 20.0,
        "The first mixture frequency was not prominent.");
    require(getSpectrogramMagnitude(spectrogram, 0, secondBin)
            > distantMagnitude * 20.0,
        "The second mixture frequency was not prominent.");
}

void testFrequencyMapping()
{
    constexpr std::uint32_t sampleRateHz = 48'000;
    constexpr std::size_t fftSize = 2048;
    require(std::abs(binToFrequencyHz(0, fftSize, sampleRateHz)) < tolerance,
        "Bin zero did not map to zero Hz.");
    require(std::abs(
                binToFrequencyHz(fftSize / 2, fftSize, sampleRateHz) - 24'000.0)
            < tolerance,
        "The Nyquist bin mapped to the wrong frequency.");
    require(
        frequencyToNearestBin(
            binToFrequencyHz(19, fftSize, sampleRateHz), fftSize, sampleRateHz)
            == 19,
        "An exact-bin frequency mapped to the wrong bin.");

    const double binSpacingHz =
        static_cast<double>(sampleRateHz) / static_cast<double>(fftSize);
    require(
        frequencyToNearestBin(3.49 * binSpacingHz, fftSize, sampleRateHz) == 3,
        "A frequency below halfway rounded incorrectly.");
    require(
        frequencyToNearestBin(3.5 * binSpacingHz, fftSize, sampleRateHz) == 4,
        "A halfway frequency did not round toward the higher bin.");
}

void testMetadataAndIndexing()
{
    const AudioBuffer source = createSignal(20, 100, 10.0, 1.0);
    const AudioBuffer target = createSignal(20, 100, 20.0, 1.0);
    const Spectrogram sourceSpectrogram = calculateSpectrogram(source, 10, 5);
    const Spectrogram targetSpectrogram = calculateSpectrogram(target, 10, 5);

    require(sourceSpectrogram.frameCount == targetSpectrogram.frameCount
            && sourceSpectrogram.binCount == targetSpectrogram.binCount,
        "Compatible audio produced mismatched spectrogram dimensions.");
    require(sourceSpectrogram.fftSize == 10 && sourceSpectrogram.hopSize == 5
            && sourceSpectrogram.sampleRateHz == 100,
        "Spectrogram configuration metadata is incorrect.");
    require(sourceSpectrogram.magnitudes.size()
            == sourceSpectrogram.frameCount * sourceSpectrogram.binCount,
        "Spectrogram storage is not frame-major and contiguous.");
    requireOutOfRange([&sourceSpectrogram]() {
        getSpectrogramMagnitude(
            sourceSpectrogram, sourceSpectrogram.frameCount, 0);
    }, "Invalid spectrogram frame");
    requireOutOfRange(
        []() { binToFrequencyHz(6, 10, 100); }, "Invalid FFT bin");
}

void testInvalidSettings()
{
    const AudioBuffer audio = createSignal(16, 48'000);
    requireInvalidArgument(
        [&audio]() { calculateSpectrogram(audio, 0, 1); }, "Zero FFT size");
    requireInvalidArgument(
        [&audio]() { calculateSpectrogram(audio, 8, 0); }, "Zero hop size");
    requireInvalidArgument([&audio]() { calculateSpectrogram(audio, 8, 9); },
        "Hop larger than FFT");

    AudioBuffer invalidRate = audio;
    invalidRate.sampleRateHz = 0;
    requireInvalidArgument([&invalidRate]() {
        calculateSpectrogram(invalidRate, 8, 4);
    }, "Zero sample rate");
    requireInvalidArgument(
        []() { frequencyToNearestBin(-1.0, 8, 48'000); }, "Negative frequency");
    requireInvalidArgument([]() {
        frequencyToNearestBin(
            std::numeric_limits<double>::infinity(), 8, 48'000);
    }, "Infinite frequency");
}

void testReusableAnalyzerAndIndividualFrames()
{
    const AudioBuffer audio = createSignal(23, 100, 12.5, 0.7, 25.0, 0.2);
    const Spectrogram convenience = calculateSpectrogram(audio, 8, 4);
    StftAnalyzer analyzer(8, 4);
    const Spectrogram reused = analyzer.calculate(audio);
    require(convenience.magnitudes == reused.magnitudes,
        "Reusable STFT analysis changed spectral magnitudes.");
    require(analyzer.fftSize() == 8 && analyzer.hopSize() == 4
            && analyzer.binCount() == 5,
        "Reusable analyzer reported incorrect settings.");
    std::vector<double> frame(analyzer.binCount());
    for (std::size_t index = 0; index < reused.frameCount; ++index) {
        analyzer.calculateFrameMagnitudes(audio, index, frame);
        for (std::size_t bin = 0; bin < reused.binCount; ++bin)
            require(frame[bin] == getSpectrogramMagnitude(reused, index, bin),
                "Individual STFT frame differs from full analysis.");
    }
}

void testComplexSpectrogramMetadataAndStorage()
{
    const AudioBuffer audio = createSignal(123, 8'000, 500.0, 0.75);
    ComplexStftAnalyzer analyzer(32, 8);
    const ComplexSpectrogram spectrum =
        calculateComplexSpectrogram(audio, analyzer);
    require(spectrum.fftSize == 32 && spectrum.hopSize == 8,
        "Complex spectrogram has incorrect STFT settings.");
    require(spectrum.binCount == 17 && spectrum.sampleRateHz == 8'000,
        "Complex spectrogram has incorrect bin or sample-rate metadata.");
    require(spectrum.originalSampleCount == audio.samples.size(),
        "Complex spectrogram lost the original sample count.");
    require(
        spectrum.coefficients.size() == spectrum.frameCount * spectrum.binCount,
        "Complex coefficients are not frame-major and contiguous.");
    require(analyzer.fftSize() == 32 && analyzer.hopSize() == 8
            && analyzer.binCount() == 17,
        "Complex analyzer reported incorrect settings.");
    require(std::any_of(spectrum.coefficients.begin(),
                spectrum.coefficients.end(),
                [](const std::complex<double>& value) {
        return std::abs(value) > tolerance;
    }),
        "Complex analysis discarded a non-silent signal.");
}

void testComplexRoundTripSignals()
{
    constexpr std::uint32_t sampleRateHz = 8'000;
    constexpr std::size_t fftSize = 128;
    constexpr std::size_t hopSize = 32;

    requireRoundTrip(createSignal(800, sampleRateHz), fftSize, hopSize);
    requireRoundTrip(
        createSignal(800, sampleRateHz, 437.5, 0.73), fftSize, hopSize);
    requireRoundTrip(
        createSignal(800, sampleRateHz, 437.5, 0.55, 1'093.75, 0.31),
        fftSize,
        hopSize);

    AudioBuffer impulse = createSignal(257, sampleRateHz);
    impulse.samples[113] = 0.91;
    requireRoundTrip(impulse, fftSize, hopSize);

    const AudioBuffer shortInput{{0.8, -0.4, 0.2, -0.1, 0.05}, sampleRateHz};
    requireRoundTrip(shortInput, 32, 8);

    AudioBuffer nonMultiple =
        createSignal(333, sampleRateHz, 613.0, 0.61, 1'337.0, 0.19);
    requireRoundTrip(nonMultiple, fftSize, hopSize);

    AudioBuffer boundaries = createSignal(191, sampleRateHz);
    boundaries.samples.front() = 0.87;
    boundaries.samples.back() = -0.63;
    requireRoundTrip(boundaries, fftSize, hopSize);
}

void testComplexRoundTripDeterminismAndReuse()
{
    const AudioBuffer audio =
        createSignal(379, 12'000, 731.0, 0.62, 2'111.0, 0.21);
    ComplexStftAnalyzer analyzer(64, 16);
    ComplexStftSynthesizer synthesizer(64);
    const ComplexSpectrogram first = analyzer.calculate(audio);
    const ComplexSpectrogram second = analyzer.calculate(audio);
    require(first.coefficients == second.coefficients,
        "Repeated complex analysis was not deterministic.");
    const AudioBuffer firstAudio = synthesizer.reconstruct(first);
    const AudioBuffer secondAudio = synthesizer.reconstruct(second);
    require(firstAudio.samples == secondAudio.samples,
        "Repeated complex synthesis was not deterministic.");
    require(synthesizer.fftSize() == 64,
        "Complex synthesizer reported the wrong FFT size.");
}

void testEmptyComplexRoundTripAndValidation()
{
    const AudioBuffer empty{{}, 48'000};
    const ComplexSpectrogram spectrum =
        calculateComplexSpectrogram(empty, 64, 16);
    require(spectrum.frameCount == 0 && spectrum.coefficients.empty(),
        "Empty audio produced complex STFT frames.");
    const AudioBuffer reconstructed = reconstructAudio(spectrum);
    require(reconstructed.samples.empty()
            && reconstructed.sampleRateHz == empty.sampleRateHz,
        "Empty audio did not survive a complex STFT round trip.");

    ComplexSpectrogram invalid = calculateComplexSpectrogram(
        createSignal(64, 48'000, 1'000.0, 0.5), 32, 8);
    invalid.coefficients.pop_back();
    requireInvalidArgument([&invalid]() { (void)reconstructAudio(invalid); },
        "Invalid complex coefficient count");

    const ComplexSpectrogram valid = calculateComplexSpectrogram(
        createSignal(64, 48'000, 1'000.0, 0.5), 32, 8);
    ComplexStftSynthesizer wrongSize(64);
    requireInvalidArgument([&valid, &wrongSize]() {
        (void)wrongSize.reconstruct(valid);
    }, "Mismatched inverse FFT size");
    requireInvalidArgument([]() { ComplexStftAnalyzer analyzer(8, 7); },
        "Complex STFT hop with uncovered Hann samples");
    requireInvalidArgument([]() { ComplexStftAnalyzer analyzer(2, 1); },
        "Zero-valued two-sample Hann window");
}

}  // namespace

int main()
{
    try {
        testEmptyAndShortBuffers();
        testFramePlacement();
        testSilenceAndDc();
        testAlignedAndUnalignedSines();
        testTwoFrequencyMixture();
        testFrequencyMapping();
        testMetadataAndIndexing();
        testInvalidSettings();
        testReusableAnalyzerAndIndividualFrames();
        testComplexSpectrogramMetadataAndStorage();
        testComplexRoundTripSignals();
        testComplexRoundTripDeterminismAndReuse();
        testEmptyComplexRoundTripAndValidation();
        std::cout << "All spectrum tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Spectrum test failure: " << error.what() << '\n';
        return 1;
    }
}
