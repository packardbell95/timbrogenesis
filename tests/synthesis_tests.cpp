#include "synthesis.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

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

AudioBuffer makeBuffer(std::size_t sampleCount = 10,
    std::uint32_t sampleRateHz = 10,
    double initialValue = 0.0)
{
    AudioBuffer audio;
    audio.sampleRateHz = sampleRateHz;
    audio.samples.assign(sampleCount, initialValue);
    return audio;
}

SineAtom makeValidAtom()
{
    return SineAtom{
        .startSeconds = 0.2,
        .durationSeconds = 0.5,
        .frequencyHz = 0.0,
        .amplitude = 0.8,
        .phaseRadians = std::numbers::pi / 2.0,
    };
}

void testHannWindow()
{
    require(std::abs(hannWindow(0, 5)) < tolerance,
        "Hann window must begin at zero.");
    require(std::abs(hannWindow(2, 5) - 1.0) < tolerance,
        "Hann window must reach one at its center.");
    require(std::abs(hannWindow(4, 5)) < tolerance,
        "Hann window must end at zero.");
    require(std::abs(hannWindow(0, 1) - 1.0) < tolerance,
        "A one-sample Hann window must equal one.");
}

void testLocalizedSineAtom()
{
    AudioBuffer audio = makeBuffer();
    applySineAtom(audio, makeValidAtom());

    require(std::abs(audio.samples[0]) < tolerance,
        "A sine atom modified a sample before its start.");
    require(std::abs(audio.samples[1]) < tolerance,
        "A sine atom modified a sample before its start.");
    require(std::abs(audio.samples[2]) < tolerance,
        "The atom's Hann envelope must begin at zero.");
    require(std::abs(audio.samples[4] - 0.8) < tolerance,
        "The atom did not reach its expected center amplitude.");
    require(std::abs(audio.samples[6]) < tolerance,
        "The atom's Hann envelope must end at zero.");
    require(std::abs(audio.samples[7]) < tolerance,
        "A sine atom modified a sample after its end.");
}

void testPositiveAndNegativeAmplitude()
{
    AudioBuffer positive = makeBuffer();
    AudioBuffer negative = makeBuffer();
    SineAtom positiveAtom = makeValidAtom();
    positiveAtom.frequencyHz = 2.0;
    positiveAtom.phaseRadians = 0.3;
    SineAtom negativeAtom = positiveAtom;
    negativeAtom.amplitude = -positiveAtom.amplitude;

    applySineAtom(positive, positiveAtom);
    applySineAtom(negative, negativeAtom);

    bool foundNonZeroSample = false;
    for (std::size_t index = 0; index < positive.samples.size(); ++index) {
        foundNonZeroSample =
            foundNonZeroSample || std::abs(positive.samples[index]) > tolerance;
        require(std::abs(positive.samples[index] + negative.samples[index])
                < tolerance,
            "Negative amplitude did not negate the positive atom.");
    }
    require(foundNonZeroSample, "Positive amplitude produced no signal.");
}

void testPhaseAndLocalTime()
{
    AudioBuffer zeroPhase = makeBuffer();
    AudioBuffer quarterTurnPhase = makeBuffer();
    SineAtom atom = makeValidAtom();
    atom.phaseRadians = 0.0;
    applySineAtom(zeroPhase, atom);
    atom.phaseRadians = std::numbers::pi / 2.0;
    applySineAtom(quarterTurnPhase, atom);

    require(std::abs(zeroPhase.samples[4]) < tolerance,
        "A zero-frequency, zero-phase atom should be zero.");
    require(std::abs(quarterTurnPhase.samples[4] - 0.8) < tolerance,
        "Phase was not applied to the oscillator.");

    AudioBuffer early = makeBuffer(20);
    AudioBuffer late = makeBuffer(20);
    SineAtom earlyAtom = makeValidAtom();
    earlyAtom.frequencyHz = 2.0;
    earlyAtom.phaseRadians = 0.1;
    SineAtom lateAtom = earlyAtom;
    lateAtom.startSeconds = 1.0;
    applySineAtom(early, earlyAtom);
    applySineAtom(late, lateAtom);

    for (std::size_t offset = 0; offset < 5; ++offset) {
        require(std::abs(early.samples[2 + offset] - late.samples[10 + offset])
                < tolerance,
            "Sine phase depends on absolute position instead of local time.");
    }
}

void testTruncationAndStartBeyondBuffer()
{
    AudioBuffer truncated = makeBuffer();
    SineAtom atom = makeValidAtom();
    atom.startSeconds = 0.7;
    atom.durationSeconds = 0.5;
    atom.amplitude = 0.5;
    applySineAtom(truncated, atom);

    require(std::abs(truncated.samples[6]) < tolerance,
        "A truncated atom modified a sample before its start.");
    require(std::abs(truncated.samples[8] - 0.5) < tolerance,
        "A truncated atom did not process the available region.");
    require(std::abs(truncated.samples[9]) < tolerance,
        "A truncated atom did not taper at the effective end.");

    AudioBuffer beyond = makeBuffer();
    const std::vector<double> original = beyond.samples;
    atom.startSeconds = 1.5;
    atom.durationSeconds = 0.5;
    applySineAtom(beyond, atom);
    require(beyond.samples == original,
        "An atom starting beyond the buffer changed the audio.");
}

void testInvalidInputs()
{
    AudioBuffer audio = makeBuffer();

    SineAtom atom = makeValidAtom();
    atom.startSeconds = -0.1;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "Negative start");

    atom = makeValidAtom();
    atom.durationSeconds = 0.0;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "Zero duration");

    atom.durationSeconds = -0.1;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "Negative duration");

    atom = makeValidAtom();
    atom.frequencyHz = -1.0;
    requireInvalidArgument([&audio, &atom]() { applySineAtom(audio, atom); },
        "Negative frequency");

    atom.frequencyHz = 5.01;
    requireInvalidArgument([&audio, &atom]() { applySineAtom(audio, atom); },
        "Frequency above Nyquist");

    atom.frequencyHz = 5.0;
    applySineAtom(audio, atom);

    AudioBuffer invalidRate = makeBuffer();
    invalidRate.sampleRateHz = 0;
    atom = makeValidAtom();
    requireInvalidArgument([&invalidRate, &atom]() {
        applySineAtom(invalidRate, atom);
    }, "Zero sample rate");

    AudioBuffer empty = makeBuffer(0);
    requireInvalidArgument(
        [&empty, &atom]() { applySineAtom(empty, atom); }, "Empty audio");
}

void testNonFiniteFields()
{
    AudioBuffer audio = makeBuffer();
    const double infinity = std::numeric_limits<double>::infinity();
    const double notANumber = std::numeric_limits<double>::quiet_NaN();

    SineAtom atom = makeValidAtom();
    atom.startSeconds = infinity;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "Infinite start");

    atom = makeValidAtom();
    atom.durationSeconds = notANumber;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "NaN duration");

    atom = makeValidAtom();
    atom.frequencyHz = infinity;
    requireInvalidArgument([&audio, &atom]() { applySineAtom(audio, atom); },
        "Infinite frequency");

    atom = makeValidAtom();
    atom.amplitude = notANumber;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "NaN amplitude");

    atom = makeValidAtom();
    atom.phaseRadians = infinity;
    requireInvalidArgument(
        [&audio, &atom]() { applySineAtom(audio, atom); }, "Infinite phase");
}

void testNoClipping()
{
    AudioBuffer audio = makeBuffer(10, 10, 0.9);
    SineAtom atom = makeValidAtom();
    atom.amplitude = 0.4;
    applySineAtom(audio, atom);

    require(audio.samples[4] > 1.0,
        "Synthesis clipped an intermediate sample to 1.0.");
    require(std::abs(audio.samples[4] - 1.3) < tolerance,
        "Unexpected unclipped sample value.");
}

void testSampleRangeAndExactRemoval()
{
    AudioBuffer audio = makeBuffer();
    SineAtom atom = makeValidAtom();
    const SampleRange range = sineAtomSampleRange(audio, atom);
    require(range.beginSample == 2 && range.endSample == 7,
        "Sine atom sample range disagrees with synthesis rounding.");
    const AudioBuffer original = audio;
    applySineAtom(audio, atom);
    removeSineAtom(audio, atom);
    for (std::size_t index = 0; index < audio.samples.size(); ++index)
        require(std::abs(audio.samples[index] - original.samples[index])
                < tolerance,
            "Removing an atom did not cancel its contribution.");
}

}  // namespace

int main()
{
    try {
        testHannWindow();
        testLocalizedSineAtom();
        testPositiveAndNegativeAmplitude();
        testPhaseAndLocalTime();
        testTruncationAndStartBeyondBuffer();
        testInvalidInputs();
        testNonFiniteFields();
        testNoClipping();
        testSampleRangeAndExactRemoval();
        std::cout << "All synthesis tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Synthesis test failure: " << error.what() << '\n';
        return 1;
    }
}
