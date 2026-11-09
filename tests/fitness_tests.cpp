#include "fitness.hpp"
#include "synthesis.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

constexpr double tolerance = 1e-12;
constexpr std::uint32_t sampleRateHz = 8192;
constexpr std::size_t fftSize = 1024;
constexpr std::size_t hopSize = 512;

void require(bool condition, const std::string& message)
{
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void requireThrows(Function function, const std::string& testName)
{
    try {
        function();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(testName + " did not throw.");
}

AudioBuffer createSine(double frequencyHz,
    double amplitude = 1.0,
    std::size_t sampleCount = fftSize)
{
    AudioBuffer audio;
    audio.sampleRateHz = sampleRateHz;
    audio.samples.resize(sampleCount);
    for (std::size_t sampleIndex = 0; sampleIndex < sampleCount;
         ++sampleIndex) {
        const double timeSeconds = static_cast<double>(sampleIndex)
            / static_cast<double>(sampleRateHz);
        audio.samples[sampleIndex] = amplitude
            * std::sin(2.0 * std::numbers::pi * frequencyHz * timeSeconds);
    }
    return audio;
}

FitnessReference createReference(const AudioBuffer& target)
{
    return createFitnessReference(
        calculateSpectrogram(target, fftSize, hopSize));
}

void testMagnitudeToLog()
{
    require(std::isfinite(magnitudeToLog(0.0)),
        "Zero magnitude did not produce a finite log value.");
    require(std::abs(magnitudeToLog(2.0) - std::log(2.0 + 1e-12)) < tolerance,
        "Positive magnitude produced the wrong log value.");

    requireThrows([]() { magnitudeToLog(-0.1); }, "Negative magnitude");
    requireThrows([]() {
        magnitudeToLog(std::numeric_limits<double>::quiet_NaN());
    }, "NaN magnitude");
    requireThrows([]() {
        magnitudeToLog(std::numeric_limits<double>::infinity());
    }, "Infinite magnitude");
}

void testDynamicRangeFloor()
{
    const double floor = 1e-4;
    require(magnitudeToLog(0.0, floor) == magnitudeToLog(floor * 0.5, floor)
            && magnitudeToLog(floor, floor) == std::log(floor),
        "Magnitudes below the configured floor were not equivalent.");
    require(magnitudeToLog(0.5, floor) == std::log(0.5),
        "A magnitude above the floor lost its normal logarithm.");

    const Spectrogram target =
        calculateSpectrogram(createSine(440.0), fftSize, hopSize);
    const FitnessReference reference = createFitnessReference(target, 80.0);
    const double maximum =
        *std::max_element(target.magnitudes.begin(), target.magnitudes.end());
    require(std::abs(reference.magnitudeFloor - maximum * 1e-4)
            < maximum * tolerance,
        "Target-relative 80 dB floor is incorrect.");
    const AudioBuffer silence{std::vector<double>(fftSize, 0.0), sampleRateHz};
    const FitnessReference silentReference = createReference(silence);
    require(silentReference.magnitudeFloor == 1e-12
            && std::isfinite(std::log(silentReference.magnitudeFloor)),
        "Silent target did not receive the absolute fallback floor.");
    requireThrows([&target]() { createFitnessReference(target, 0.0); },
        "Zero dynamic range");
}

void testSignedResidualAndHotspots()
{
    const AudioBuffer tone = createSine(440.0);
    const AudioBuffer silence{std::vector<double>(fftSize, 0.0), sampleRateHz};
    const Spectrogram toneSpectrum =
        calculateSpectrogram(tone, fftSize, hopSize);
    const Spectrogram silenceSpectrum =
        calculateSpectrogram(silence, fftSize, hopSize);
    const FitnessReference toneReference = createFitnessReference(toneSpectrum);
    const SpectralResidual missing =
        calculateSpectralResidual(silenceSpectrum, toneReference);
    const auto missingHotspots = findResidualHotspots(missing);
    require(!missingHotspots.empty() && missingHotspots.front().residual > 0.0
            && std::abs(static_cast<long long>(missingHotspots.front().binIndex)
                   - 55)
                <= 1,
        "A missing sine did not create a positive hotspot near its frequency.");

    const FitnessReference silenceReference =
        createFitnessReference(silenceSpectrum);
    const auto excessHotspots = findResidualHotspots(
        calculateSpectralResidual(toneSpectrum, silenceReference));
    require(!excessHotspots.empty() && excessHotspots.front().residual < 0.0
            && std::abs(
                   static_cast<long long>(excessHotspots.front().binIndex) - 55)
                <= 1,
        "An excess sine did not create a negative hotspot near its frequency.");

    const SpectralResidual exact =
        calculateSpectralResidual(toneSpectrum, toneReference);
    require(findResidualHotspots(exact).empty(),
        "An exact match produced a residual hotspot.");

    SpectralResidual synthetic{.frameCount = 3,
        .binCount = 5,
        .fftSize = 8,
        .hopSize = 4,
        .sampleRateHz = 80,
        .values = std::vector<double>(15, 0.0)};
    synthetic.values[1 * 5 + 2] = 10.0;
    synthetic.values[1 * 5 + 3] = 9.0;
    synthetic.values[0 * 5 + 0] = -8.0;
    HotspotSettings settings;
    settings.maximumHotspots = 2;
    settings.suppressionFrames = 1;
    settings.suppressionBins = 1;
    const auto suppressed = findResidualHotspots(synthetic, settings);
    require(suppressed.size() == 2 && suppressed[0].binIndex == 2
            && suppressed[1].binIndex == 0,
        "Hotspot suppression did not spread selections across regions.");
}

void testIdenticalAndEmptySpectrograms()
{
    const Spectrogram identical{
        .frameCount = 1,
        .binCount = 3,
        .fftSize = 4,
        .hopSize = 2,
        .sampleRateHz = 8,
        .magnitudes = {0.0, 1.0, 2.0},
    };
    require(calculateSpectralFitness(identical, identical) == 0.0,
        "Identical spectrograms did not have zero fitness.");

    const AudioBuffer emptyAudio{{}, sampleRateHz};
    const Spectrogram empty =
        calculateSpectrogram(emptyAudio, fftSize, hopSize);
    const FitnessReference emptyReference = createFitnessReference(empty);
    require(calculateSpectralFitness(empty, emptyReference) == 0.0,
        "Compatible empty spectrograms did not have zero fitness.");
}

void testCompleteAudioPipeline()
{
    const AudioBuffer silence{{}, sampleRateHz};
    AudioBuffer populatedSilence = silence;
    populatedSilence.samples.assign(fftSize, 0.0);
    const FitnessReference silenceReference = createReference(populatedSilence);
    require(calculateAudioFitness(populatedSilence, silenceReference) == 0.0,
        "Silence compared with itself did not have zero fitness.");

    const AudioBuffer sine440 = createSine(440.0);
    const FitnessReference sineReference = createReference(sine440);
    require(calculateAudioFitness(sine440, sineReference) == 0.0,
        "A sine compared with itself did not have zero fitness.");

    const AudioBuffer sine880 = createSine(880.0);
    require(calculateAudioFitness(sine880, sineReference) > 0.0,
        "Different sine frequencies produced zero fitness.");
}

void testSimilarityRanking()
{
    const AudioBuffer target = createSine(440.0);
    const FitnessReference targetReference = createReference(target);
    const double similarFitness =
        calculateAudioFitness(createSine(445.0), targetReference);
    const double dissimilarFitness =
        calculateAudioFitness(createSine(880.0), targetReference);

    require(similarFitness < dissimilarFitness,
        "A nearby sine did not rank closer than a distant sine.");
}

void testAmplitudeAndTimingDifferences()
{
    const AudioBuffer target = createSine(440.0, 0.5);
    const FitnessReference targetReference = createReference(target);
    require(
        calculateAudioFitness(createSine(440.0, 0.25), targetReference) > 0.0,
        "An amplitude difference produced zero fitness.");

    AudioBuffer earlyTone;
    earlyTone.sampleRateHz = sampleRateHz;
    earlyTone.samples.assign(sampleRateHz, 0.0);
    AudioBuffer lateTone = earlyTone;
    const SineAtom earlyAtom{
        .startSeconds = 0.0,
        .durationSeconds = 0.4,
        .frequencyHz = 440.0,
        .amplitude = 0.5,
        .phaseRadians = 0.0,
    };
    SineAtom lateAtom = earlyAtom;
    lateAtom.startSeconds = 0.6;
    applySineAtom(earlyTone, earlyAtom);
    applySineAtom(lateTone, lateAtom);

    const FitnessReference earlyReference = createReference(earlyTone);
    require(calculateAudioFitness(lateTone, earlyReference) > 0.0,
        "A timing difference produced zero fitness.");
}

void testReferenceEquivalence()
{
    const Spectrogram candidate =
        calculateSpectrogram(createSine(445.0), fftSize, hopSize);
    const Spectrogram target =
        calculateSpectrogram(createSine(440.0), fftSize, hopSize);
    const FitnessReference reference = createFitnessReference(target);
    const double directFitness = calculateSpectralFitness(candidate, target);
    const double cachedFitness = calculateSpectralFitness(candidate, reference);

    require(std::abs(directFitness - cachedFitness) < tolerance,
        "Cached and direct target scoring produced different fitness.");
    require(reference.logMagnitudes.size() == target.magnitudes.size(),
        "The cached target has the wrong value count.");
}

void testMonotonicSanity()
{
    const AudioBuffer target = createSine(440.0, 1.0);
    const FitnessReference reference = createReference(target);
    AudioBuffer silence = target;
    std::fill(silence.samples.begin(), silence.samples.end(), 0.0);

    const double exactFitness = calculateAudioFitness(target, reference);
    const double partialFitness =
        calculateAudioFitness(createSine(440.0, 0.5), reference);
    const double silenceFitness = calculateAudioFitness(silence, reference);

    require(exactFitness < partialFitness,
        "An exact target did not beat a partial-amplitude target.");
    require(partialFitness < silenceFitness,
        "A partial-amplitude target did not beat silence.");
}

void testCompatibilityAndFiniteValidation()
{
    const Spectrogram candidate =
        calculateSpectrogram(createSine(440.0), fftSize, hopSize);
    const FitnessReference reference = createFitnessReference(candidate);

    const Spectrogram differentFrames = calculateSpectrogram(
        createSine(440.0, 1.0, fftSize * 2), fftSize, hopSize);
    requireThrows([&differentFrames, &reference]() {
        calculateSpectralFitness(differentFrames, reference);
    }, "Different frame count");

    FitnessReference differentSettings = reference;
    differentSettings.hopSize = hopSize / 2;
    requireThrows([&candidate, &differentSettings]() {
        calculateSpectralFitness(candidate, differentSettings);
    }, "Different analysis settings");

    Spectrogram differentBins = candidate;
    differentBins.fftSize = fftSize + 2;
    differentBins.binCount = differentBins.fftSize / 2 + 1;
    differentBins.magnitudes.resize(
        differentBins.frameCount * differentBins.binCount, 0.0);
    requireThrows([&differentBins, &reference]() {
        calculateSpectralFitness(differentBins, reference);
    }, "Different bin count");

    Spectrogram nonFiniteCandidate = candidate;
    nonFiniteCandidate.magnitudes[0] = std::numeric_limits<double>::quiet_NaN();
    requireThrows([&nonFiniteCandidate, &reference]() {
        calculateSpectralFitness(nonFiniteCandidate, reference);
    }, "Non-finite candidate magnitude");

    FitnessReference nonFiniteReference = reference;
    nonFiniteReference.logMagnitudes[0] =
        std::numeric_limits<double>::infinity();
    requireThrows([&candidate, &nonFiniteReference]() {
        calculateSpectralFitness(candidate, nonFiniteReference);
    }, "Non-finite target log magnitude");

    Spectrogram malformed = candidate;
    malformed.magnitudes.pop_back();
    requireThrows([&malformed]() { createFitnessReference(malformed); },
        "Malformed spectrogram");
}

void testFitnessStateAndReusableAnalyzer()
{
    const AudioBuffer target = createSine(440.0);
    const AudioBuffer candidate = createSine(445.0);
    const Spectrogram targetSpectrum =
        calculateSpectrogram(target, fftSize, hopSize);
    const FitnessReference reference = createFitnessReference(targetSpectrum);
    const Spectrogram candidateSpectrum =
        calculateSpectrogram(candidate, fftSize, hopSize);
    const FitnessState state = createFitnessState(candidateSpectrum, reference);
    require(std::abs(state.fitness
                - calculateSpectralFitness(candidateSpectrum, reference))
            < tolerance,
        "Per-frame fitness state differs from full fitness.");
    double total = 0.0;
    for (double error : state.frameSquaredErrors)
        total += error;
    require(std::abs(total - state.totalSquaredError) < tolerance,
        "Per-frame errors do not sum to the cached total.");
    StftAnalyzer analyzer(fftSize, hopSize);
    require(std::abs(calculateAudioFitness(candidate, reference, analyzer)
                - state.fitness)
            < tolerance,
        "Reusable-analyzer audio fitness differs from full fitness.");
}

}  // namespace

int main()
{
    try {
        testMagnitudeToLog();
        testDynamicRangeFloor();
        testSignedResidualAndHotspots();
        testIdenticalAndEmptySpectrograms();
        testCompleteAudioPipeline();
        testSimilarityRanking();
        testAmplitudeAndTimingDifferences();
        testReferenceEquivalence();
        testMonotonicSanity();
        testCompatibilityAndFiniteValidation();
        testFitnessStateAndReusableAnalyzer();
        std::cout << "All fitness tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Fitness test failure: " << error.what() << '\n';
        return 1;
    }
}
