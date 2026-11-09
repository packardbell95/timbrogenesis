#include "mutation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <type_traits>

namespace {

void validateMutationSettingsImplementation(
    const AudioBuffer& audio, const SineMutationSettings& settings)
{
    if (audio.sampleRateHz == 0) {
        throw std::invalid_argument("Cannot generate a mutation for audio with "
                                    "an invalid sample rate.");
    }
    if (audio.samples.empty()) {
        throw std::invalid_argument(
            "Cannot generate a mutation for empty audio.");
    }
    if (!std::isfinite(settings.minimumFrequencyHz)
        || settings.minimumFrequencyHz <= 0.0) {
        throw std::invalid_argument(
            "Minimum mutation frequency must be finite and greater than zero.");
    }
    if (!std::isfinite(settings.maximumFrequencyHz)
        || settings.maximumFrequencyHz < settings.minimumFrequencyHz) {
        throw std::invalid_argument("Maximum mutation frequency must be finite "
                                    "and at least the minimum.");
    }
    if (!std::isfinite(settings.minimumDurationSeconds)
        || settings.minimumDurationSeconds <= 0.0) {
        throw std::invalid_argument(
            "Minimum mutation duration must be finite and greater than zero.");
    }
    if (!std::isfinite(settings.maximumDurationSeconds)
        || settings.maximumDurationSeconds < settings.minimumDurationSeconds) {
        throw std::invalid_argument("Maximum mutation duration must be finite "
                                    "and at least the minimum.");
    }
    if (!std::isfinite(settings.maximumAbsoluteAmplitude)
        || settings.maximumAbsoluteAmplitude < 0.0) {
        throw std::invalid_argument(
            "Maximum mutation amplitude must be finite and non-negative.");
    }

    const double nyquistFrequencyHz =
        static_cast<double>(audio.sampleRateHz) / 2.0;
    if (settings.minimumFrequencyHz > nyquistFrequencyHz) {
        throw std::invalid_argument(
            "Minimum mutation frequency exceeds the Nyquist frequency.");
    }
}

double uniformReal(
    double minimum, double maximum, std::mt19937_64& randomGenerator)
{
    if (minimum == maximum) {
        return minimum;
    }
    std::uniform_real_distribution<double> distribution(minimum, maximum);
    return distribution(randomGenerator);
}

const GenomeAtom& requireAtom(const Genome& genome, AtomId id)
{
    const GenomeAtom* atom = findGenomeAtom(genome, id);
    if (atom == nullptr) {
        throw std::invalid_argument(
            "Genome mutation references an unknown atom ID.");
    }
    return *atom;
}

std::size_t randomAtomIndex(
    const Genome& genome, std::mt19937_64& randomGenerator)
{
    std::uniform_int_distribution<std::size_t> distribution(
        0, genome.atoms.size() - 1);
    return distribution(randomGenerator);
}

}  // namespace

RandomSeed createRandomSeed()
{
    std::random_device randomDevice;
    const auto high = static_cast<RandomSeed>(randomDevice()) << 32U;
    const auto low = static_cast<RandomSeed>(randomDevice());
    return high ^ low;
}

void validateSineMutationSettings(
    const AudioBuffer& audio, const SineMutationSettings& settings)
{
    validateMutationSettingsImplementation(audio, settings);
}

SineAtom createRandomSineAtom(const AudioBuffer& audio,
    const SineMutationSettings& settings,
    std::mt19937_64& randomGenerator)
{
    validateSineMutationSettings(audio, settings);

    const double audioDurationSeconds =
        static_cast<double>(audio.samples.size())
        / static_cast<double>(audio.sampleRateHz);
    const double maximumDurationSeconds =
        std::min(settings.maximumDurationSeconds, audioDurationSeconds);
    const double minimumDurationSeconds =
        std::min(settings.minimumDurationSeconds, maximumDurationSeconds);
    const double durationSeconds = uniformReal(
        minimumDurationSeconds, maximumDurationSeconds, randomGenerator);
    const double maximumStartSeconds =
        std::max(0.0, audioDurationSeconds - durationSeconds);
    const double startSeconds =
        uniformReal(0.0, maximumStartSeconds, randomGenerator);

    const double nyquistFrequencyHz =
        static_cast<double>(audio.sampleRateHz) / 2.0;
    const double maximumFrequencyHz =
        std::min(settings.maximumFrequencyHz, nyquistFrequencyHz);
    const double logFrequencyHz =
        uniformReal(std::log(settings.minimumFrequencyHz),
            std::log(maximumFrequencyHz),
            randomGenerator);

    return SineAtom{
        .startSeconds = startSeconds,
        .durationSeconds = durationSeconds,
        .frequencyHz = std::exp(logFrequencyHz),
        .amplitude = uniformReal(-settings.maximumAbsoluteAmplitude,
            settings.maximumAbsoluteAmplitude,
            randomGenerator),
        .phaseRadians =
            uniformReal(0.0, 2.0 * std::numbers::pi, randomGenerator),
    };
}

const GenomeAtom* findGenomeAtom(const Genome& genome, AtomId atomId)
{
    const auto found = std::find_if(genome.atoms.begin(),
        genome.atoms.end(),
        [atomId](const GenomeAtom& atom) { return atom.id == atomId; });
    return found == genome.atoms.end() ? nullptr : &*found;
}

void validateGenome(const AudioBuffer& source, const Genome& genome)
{
    for (std::size_t index = 0; index < genome.atoms.size(); ++index) {
        const SampleRange range =
            sineAtomSampleRange(source, genome.atoms[index].atom);
        if (range.beginSample == range.endSample)
            throw std::invalid_argument(
                "Genome atoms must affect at least one sample.");
        for (std::size_t other = index + 1; other < genome.atoms.size();
             ++other) {
            if (genome.atoms[index].id == genome.atoms[other].id) {
                throw std::invalid_argument("Genome atom IDs must be unique.");
            }
        }
    }
}

AudioBuffer renderGenome(const AudioBuffer& source, const Genome& genome)
{
    validateGenome(source, genome);
    AudioBuffer rendered = source;
    for (const GenomeAtom& atom : genome.atoms) {
        applySineAtom(rendered, atom.atom);
    }
    return rendered;
}

void applyGenomeMutationToAudio(AudioBuffer& renderedAudio,
    const Genome& genome,
    const GenomeMutation& mutation)
{
    std::visit([&](const auto& operation) {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T, AddAtomMutation>) {
            if (findGenomeAtom(genome, operation.atom.id))
                throw std::invalid_argument(
                    "Cannot add a duplicate genome atom ID.");
            const SampleRange range =
                sineAtomSampleRange(renderedAudio, operation.atom.atom);
            if (range.beginSample == range.endSample)
                throw std::invalid_argument(
                    "Added genome atoms must affect audio.");
            applySineAtom(renderedAudio, operation.atom.atom);
        } else {
            const SineAtom oldAtom = requireAtom(genome, operation.atomId).atom;
            if constexpr (std::is_same_v<T, RemoveAtomMutation>) {
                removeSineAtom(renderedAudio, oldAtom);
            } else {
                SineAtom replacement = oldAtom;
                if constexpr (std::is_same_v<T, MoveAtomMutation>)
                    replacement.startSeconds = operation.startSeconds;
                else if constexpr (std::is_same_v<T, ChangeFrequencyMutation>)
                    replacement.frequencyHz = operation.frequencyHz;
                else if constexpr (std::is_same_v<T, ChangeAmplitudeMutation>)
                    replacement.amplitude = operation.amplitude;
                else if constexpr (std::is_same_v<T, ChangeDurationMutation>)
                    replacement.durationSeconds = operation.durationSeconds;
                else
                    replacement.phaseRadians = operation.phaseRadians;
                removeSineAtom(renderedAudio, oldAtom);
                applySineAtom(renderedAudio, replacement);
            }
        }
    }, mutation);
}

void applyGenomeMutation(
    Genome& genome, AudioBuffer& renderedAudio, const GenomeMutation& mutation)
{
    std::visit([&](const auto& operation) {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T, AddAtomMutation>) {
            if (findGenomeAtom(genome, operation.atom.id) != nullptr) {
                throw std::invalid_argument(
                    "Cannot add a duplicate genome atom ID.");
            }
            const SampleRange range =
                sineAtomSampleRange(renderedAudio, operation.atom.atom);
            if (range.beginSample == range.endSample)
                throw std::invalid_argument(
                    "Added genome atoms must affect audio.");
            applySineAtom(renderedAudio, operation.atom.atom);
            genome.atoms.push_back(operation.atom);
        } else {
            const GenomeAtom& current = requireAtom(genome, operation.atomId);
            const SineAtom oldAtom = current.atom;
            if constexpr (std::is_same_v<T, RemoveAtomMutation>) {
                removeSineAtom(renderedAudio, oldAtom);
                genome.atoms.erase(std::remove_if(genome.atoms.begin(),
                                       genome.atoms.end(),
                                       [&](const GenomeAtom& atom) {
                    return atom.id == operation.atomId;
                }),
                    genome.atoms.end());
            } else {
                SineAtom replacement = oldAtom;
                if constexpr (std::is_same_v<T, MoveAtomMutation>) {
                    replacement.startSeconds = operation.startSeconds;
                } else if constexpr (std::is_same_v<T,
                                         ChangeFrequencyMutation>) {
                    replacement.frequencyHz = operation.frequencyHz;
                } else if constexpr (std::is_same_v<T,
                                         ChangeAmplitudeMutation>) {
                    replacement.amplitude = operation.amplitude;
                } else if constexpr (std::is_same_v<T,
                                         ChangeDurationMutation>) {
                    replacement.durationSeconds = operation.durationSeconds;
                } else if constexpr (std::is_same_v<T, ChangePhaseMutation>) {
                    replacement.phaseRadians = operation.phaseRadians;
                }
                if constexpr (std::is_same_v<T, MoveAtomMutation>) {
                    if (operation.previousStartSeconds != oldAtom.startSeconds)
                        throw std::invalid_argument(
                            "Move mutation previous value is stale.");
                } else if constexpr (std::is_same_v<T,
                                         ChangeFrequencyMutation>) {
                    if (operation.previousFrequencyHz != oldAtom.frequencyHz)
                        throw std::invalid_argument(
                            "Frequency mutation previous value is stale.");
                } else if constexpr (std::is_same_v<T,
                                         ChangeAmplitudeMutation>) {
                    if (operation.previousAmplitude != oldAtom.amplitude)
                        throw std::invalid_argument(
                            "Amplitude mutation previous value is stale.");
                } else if constexpr (std::is_same_v<T,
                                         ChangeDurationMutation>) {
                    if (operation.previousDurationSeconds
                        != oldAtom.durationSeconds)
                        throw std::invalid_argument(
                            "Duration mutation previous value is stale.");
                } else if constexpr (std::is_same_v<T, ChangePhaseMutation>) {
                    if (operation.previousPhaseRadians != oldAtom.phaseRadians)
                        throw std::invalid_argument(
                            "Phase mutation previous value is stale.");
                }
                AudioBuffer validation = renderedAudio;
                applySineAtom(validation, replacement);
                removeSineAtom(renderedAudio, oldAtom);
                applySineAtom(renderedAudio, replacement);
                for (GenomeAtom& atom : genome.atoms) {
                    if (atom.id == operation.atomId) {
                        atom.atom = replacement;
                        break;
                    }
                }
            }
        }
    }, mutation);
}

GenomeMutation createRandomGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    AtomId addAtomId,
    const SineMutationSettings& sineSettings,
    const GenomeMutationSettings& settings,
    std::mt19937_64& randomGenerator)
{
    validateSineMutationSettings(audio, sineSettings);
    const bool invalidScale = !std::isfinite(settings.maximumMoveSeconds)
        || settings.maximumMoveSeconds < 0.0
        || !std::isfinite(settings.maximumFrequencyRatio)
        || settings.maximumFrequencyRatio < 1.0
        || !std::isfinite(settings.maximumAmplitudeDelta)
        || settings.maximumAmplitudeDelta < 0.0
        || !std::isfinite(settings.maximumDurationDeltaSeconds)
        || settings.maximumDurationDeltaSeconds < 0.0
        || !std::isfinite(settings.maximumPhaseDeltaRadians)
        || settings.maximumPhaseDeltaRadians < 0.0;
    if (invalidScale)
        throw std::invalid_argument("Genome mutation scales are invalid.");
    const double allWeights = settings.addWeight + settings.removeWeight
        + settings.moveWeight + settings.frequencyWeight
        + settings.amplitudeWeight + settings.durationWeight
        + settings.phaseWeight;
    if (!std::isfinite(allWeights) || settings.addWeight < 0.0
        || settings.removeWeight < 0.0 || settings.moveWeight < 0.0
        || settings.frequencyWeight < 0.0 || settings.amplitudeWeight < 0.0
        || settings.durationWeight < 0.0 || settings.phaseWeight < 0.0)
        throw std::invalid_argument(
            "Genome mutation weights must be finite and non-negative.");
    if (genome.atoms.empty()) {
        return AddAtomMutation{{addAtomId,
            createRandomSineAtom(audio, sineSettings, randomGenerator)}};
    }
    const double total = allWeights;
    if (total <= 0.0)
        throw std::invalid_argument(
            "At least one genome mutation weight must be positive.");
    double choice = uniformReal(0.0, total, randomGenerator);
    if (choice < settings.addWeight) {
        return AddAtomMutation{{addAtomId,
            createRandomSineAtom(audio, sineSettings, randomGenerator)}};
    }
    choice -= settings.addWeight;
    const GenomeAtom& selected =
        genome.atoms[randomAtomIndex(genome, randomGenerator)];
    if ((choice -= settings.removeWeight) < 0.0) {
        return RemoveAtomMutation{selected.id};
    }
    const double duration = static_cast<double>(audio.samples.size())
        / static_cast<double>(audio.sampleRateHz);
    if ((choice -= settings.moveWeight) < 0.0) {
        const double maximumStart =
            std::max(0.0, duration - selected.atom.durationSeconds);
        return MoveAtomMutation{selected.id,
            selected.atom.startSeconds,
            std::clamp(selected.atom.startSeconds
                    + uniformReal(-settings.maximumMoveSeconds,
                        settings.maximumMoveSeconds,
                        randomGenerator),
                0.0,
                maximumStart)};
    }
    if ((choice -= settings.frequencyWeight) < 0.0) {
        const double ratio = uniformReal(1.0 / settings.maximumFrequencyRatio,
            settings.maximumFrequencyRatio,
            randomGenerator);
        return ChangeFrequencyMutation{selected.id,
            selected.atom.frequencyHz,
            std::clamp(selected.atom.frequencyHz * ratio,
                sineSettings.minimumFrequencyHz,
                std::min(sineSettings.maximumFrequencyHz,
                    static_cast<double>(audio.sampleRateHz) / 2.0))};
    }
    if ((choice -= settings.amplitudeWeight) < 0.0) {
        return ChangeAmplitudeMutation{selected.id,
            selected.atom.amplitude,
            std::clamp(selected.atom.amplitude
                    + uniformReal(-settings.maximumAmplitudeDelta,
                        settings.maximumAmplitudeDelta,
                        randomGenerator),
                -sineSettings.maximumAbsoluteAmplitude,
                sineSettings.maximumAbsoluteAmplitude)};
    }
    if ((choice -= settings.durationWeight) < 0.0) {
        const double maximumDuration =
            std::min(sineSettings.maximumDurationSeconds,
                duration - selected.atom.startSeconds);
        return ChangeDurationMutation{selected.id,
            selected.atom.durationSeconds,
            std::clamp(selected.atom.durationSeconds
                    + uniformReal(-settings.maximumDurationDeltaSeconds,
                        settings.maximumDurationDeltaSeconds,
                        randomGenerator),
                std::min(sineSettings.minimumDurationSeconds, maximumDuration),
                maximumDuration)};
    }
    return ChangePhaseMutation{selected.id,
        selected.atom.phaseRadians,
        selected.atom.phaseRadians
            + uniformReal(-settings.maximumPhaseDeltaRadians,
                settings.maximumPhaseDeltaRadians,
                randomGenerator)};
}

namespace {

struct LocalFrameRange {
    std::size_t begin = 0;
    std::size_t end = 0;
};

LocalFrameRange atomFrames(const AudioBuffer& audio,
    const SineAtom& atom,
    std::size_t fftSize,
    std::size_t hopSize)
{
    if (fftSize == 0 || hopSize == 0 || hopSize > fftSize)
        throw std::invalid_argument(
            "Atom relevance STFT settings are invalid.");
    const SampleRange samples = sineAtomSampleRange(audio, atom);
    const std::size_t frameCount =
        audio.samples.empty() ? 0 : 1 + (audio.samples.size() - 1) / hopSize;
    if (samples.beginSample == samples.endSample)
        return {};
    const std::size_t begin = samples.beginSample < fftSize
        ? 0
        : (samples.beginSample - fftSize) / hopSize + 1;
    const std::size_t end = std::min(frameCount,
        samples.endSample / hopSize
            + (samples.endSample % hopSize != 0 ? 1U : 0U));
    return {std::min(begin, frameCount), end};
}

void validateResidualForAudio(
    const AudioBuffer& audio, const SpectralResidual& residual)
{
    if (residual.fftSize == 0 || residual.hopSize == 0
        || residual.hopSize > residual.fftSize)
        throw std::invalid_argument(
            "Spectral residual is incompatible with audio.");
    const std::size_t expectedFrames = audio.samples.empty()
        ? 0
        : 1 + (audio.samples.size() - 1) / residual.hopSize;
    if (residual.binCount != residual.fftSize / 2 + 1
        || residual.frameCount != expectedFrames
        || residual.sampleRateHz != audio.sampleRateHz
        || (residual.frameCount != 0
            && residual.binCount
                > std::numeric_limits<std::size_t>::max() / residual.frameCount)
        || residual.values.size() != residual.frameCount * residual.binCount)
        throw std::invalid_argument(
            "Spectral residual is incompatible with audio.");
}

void validateSearchSettings(const GuidedSearchSettings& settings)
{
    const bool invalid = !std::isfinite(settings.guidedMutationRatio)
        || settings.guidedMutationRatio < 0.0
        || settings.guidedMutationRatio > 1.0
        || !std::isfinite(settings.guidedFrequencyRadiusBins)
        || settings.guidedFrequencyRadiusBins < 0.0
        || !std::isfinite(settings.guidedTimeRadiusHops)
        || settings.guidedTimeRadiusHops < 0.0
        || !std::isfinite(settings.durationPersistenceThresholdRatio)
        || settings.durationPersistenceThresholdRatio < 0.0
        || settings.durationPersistenceThresholdRatio > 1.0
        || !std::isfinite(settings.atomSelectionExplorationWeight)
        || settings.atomSelectionExplorationWeight <= 0.0
        || !std::isfinite(settings.refinementBudgetRatio)
        || settings.refinementBudgetRatio < 0.0
        || settings.refinementBudgetRatio >= 1.0
        || settings.refinementParentCount == 0
        || !std::isfinite(settings.refinementFrequencyRadiusBins)
        || settings.refinementFrequencyRadiusBins < 0.0
        || !std::isfinite(settings.refinementTimeRadiusHops)
        || settings.refinementTimeRadiusHops < 0.0
        || !std::isfinite(settings.refinementDurationRatio)
        || settings.refinementDurationRatio < 0.0
        || settings.refinementDurationRatio > 1.0;
    if (invalid)
        throw std::invalid_argument("Guided search settings are invalid.");
}

std::size_t chooseWeightedIndex(
    const std::vector<double>& weights, std::mt19937_64& generator)
{
    double total = 0.0;
    for (double weight : weights)
        total += weight;
    if (!std::isfinite(total) || total <= 0.0)
        throw std::invalid_argument("Selection weights are invalid.");
    double choice = uniformReal(0.0, total, generator);
    for (std::size_t index = 0; index < weights.size(); ++index) {
        if (choice < weights[index])
            return index;
        choice -= weights[index];
    }
    return weights.size() - 1;
}

std::pair<std::size_t, std::size_t> persistentFrames(
    const SpectralResidual& residual,
    std::size_t frame,
    std::size_t bin,
    double thresholdRatio)
{
    const double center = residual.values.at(frame * residual.binCount + bin);
    const double threshold = std::abs(center) * thresholdRatio;
    const bool positive = center >= 0.0;
    std::size_t begin = frame;
    std::size_t end = frame + 1;
    while (begin > 0) {
        const double value =
            residual.values[(begin - 1) * residual.binCount + bin];
        if ((value >= 0.0) != positive || std::abs(value) < threshold)
            break;
        --begin;
    }
    while (end < residual.frameCount) {
        const double value = residual.values[end * residual.binCount + bin];
        if ((value >= 0.0) != positive || std::abs(value) < threshold)
            break;
        ++end;
    }
    return {begin, end};
}

double wrapPhase(double phase)
{
    constexpr double turn = 2.0 * std::numbers::pi;
    phase = std::fmod(phase, turn);
    return phase < 0.0 ? phase + turn : phase;
}

}  // namespace

std::vector<double> calculateAtomSelectionWeights(const AudioBuffer& audio,
    const Genome& genome,
    const FitnessState& fitness,
    std::size_t fftSize,
    std::size_t hopSize,
    double explorationWeight)
{
    if (!std::isfinite(explorationWeight) || explorationWeight <= 0.0)
        throw std::invalid_argument(
            "Atom exploration weight must be positive.");
    const std::size_t frameCount =
        audio.samples.empty() ? 0 : 1 + (audio.samples.size() - 1) / hopSize;
    if (fitness.frameSquaredErrors.size() != frameCount)
        throw std::invalid_argument(
            "Fitness state does not match audio frame count.");
    std::vector<double> scores;
    scores.reserve(genome.atoms.size());
    double sum = 0.0;
    for (const GenomeAtom& atom : genome.atoms) {
        const LocalFrameRange frames =
            atomFrames(audio, atom.atom, fftSize, hopSize);
        double score = 0.0;
        for (std::size_t frame = frames.begin; frame < frames.end; ++frame)
            score += fitness.frameSquaredErrors[frame];
        score = std::sqrt(std::max(0.0, score));
        scores.push_back(score);
        sum += score;
    }
    const double typical = scores.empty()
        ? 1.0
        : std::max(sum / static_cast<double>(scores.size()), 1e-12);
    for (double& score : scores)
        score += explorationWeight * typical;
    return scores;
}

SineAtom createGuidedSineAtom(const AudioBuffer& audio,
    const SpectralResidual& residual,
    const ResidualHotspot& hotspot,
    const SineMutationSettings& sineSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator)
{
    validateSineMutationSettings(audio, sineSettings);
    validateSearchSettings(searchSettings);
    validateResidualForAudio(audio, residual);
    if (hotspot.frameIndex >= residual.frameCount
        || hotspot.binIndex >= residual.binCount || residual.fftSize == 0
        || residual.hopSize == 0 || residual.sampleRateHz != audio.sampleRateHz)
        throw std::invalid_argument(
            "Residual hotspot is incompatible with audio.");
    const auto persistence = persistentFrames(residual,
        hotspot.frameIndex,
        hotspot.binIndex,
        searchSettings.durationPersistenceThresholdRatio);
    const double audioDuration = static_cast<double>(audio.samples.size())
        / static_cast<double>(audio.sampleRateHz);
    const double persistentDuration =
        static_cast<double>(
            (persistence.second - persistence.first) * residual.hopSize)
        / static_cast<double>(audio.sampleRateHz);
    const double maximumDuration =
        std::min(sineSettings.maximumDurationSeconds, audioDuration);
    const double minimumDuration =
        std::min(sineSettings.minimumDurationSeconds, maximumDuration);
    const double preferredDuration =
        std::clamp(persistentDuration, minimumDuration, maximumDuration);
    const double duration =
        std::clamp(preferredDuration * uniformReal(0.75, 1.25, randomGenerator),
            minimumDuration,
            maximumDuration);

    const double binSpacing = static_cast<double>(audio.sampleRateHz)
        / static_cast<double>(residual.fftSize);
    const double frequency = std::clamp(hotspot.frequencyHz
            + uniformReal(
                -searchSettings.guidedFrequencyRadiusBins * binSpacing,
                searchSettings.guidedFrequencyRadiusBins * binSpacing,
                randomGenerator),
        sineSettings.minimumFrequencyHz,
        std::min(sineSettings.maximumFrequencyHz,
            static_cast<double>(audio.sampleRateHz) / 2.0));
    const double hopSeconds = static_cast<double>(residual.hopSize)
        / static_cast<double>(audio.sampleRateHz);
    const double maximumStart = std::max(0.0, audioDuration - duration);
    const double start = std::clamp(hotspot.timeSeconds - duration / 2.0
            + uniformReal(-searchSettings.guidedTimeRadiusHops * hopSeconds,
                searchSettings.guidedTimeRadiusHops * hopSeconds,
                randomGenerator),
        0.0,
        maximumStart);
    return {start,
        duration,
        frequency,
        uniformReal(-sineSettings.maximumAbsoluteAmplitude,
            sineSettings.maximumAbsoluteAmplitude,
            randomGenerator),
        uniformReal(0.0, 2.0 * std::numbers::pi, randomGenerator)};
}

GenomeMutation createGuidedGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    AtomId addAtomId,
    const SineMutationSettings& sineSettings,
    const GenomeMutationSettings& genomeSettings,
    const GuidedSearchSettings& searchSettings,
    const SpectralResidual& residual,
    const ResidualHotspot& hotspot,
    const FitnessState& fitness,
    std::mt19937_64& randomGenerator)
{
    validateSearchSettings(searchSettings);
    validateResidualForAudio(audio, residual);
    if (genome.atoms.empty())
        return AddAtomMutation{{addAtomId,
            createGuidedSineAtom(audio,
                residual,
                hotspot,
                sineSettings,
                searchSettings,
                randomGenerator)}};
    const double total = genomeSettings.addWeight + genomeSettings.removeWeight
        + genomeSettings.moveWeight + genomeSettings.frequencyWeight
        + genomeSettings.amplitudeWeight + genomeSettings.durationWeight
        + genomeSettings.phaseWeight;
    if (!std::isfinite(total) || total <= 0.0 || genomeSettings.addWeight < 0.0
        || genomeSettings.removeWeight < 0.0 || genomeSettings.moveWeight < 0.0
        || genomeSettings.frequencyWeight < 0.0
        || genomeSettings.amplitudeWeight < 0.0
        || genomeSettings.durationWeight < 0.0
        || genomeSettings.phaseWeight < 0.0)
        throw std::invalid_argument("Genome mutation weights are invalid.");
    double choice = uniformReal(0.0, total, randomGenerator);
    if (choice < genomeSettings.addWeight)
        return AddAtomMutation{{addAtomId,
            createGuidedSineAtom(audio,
                residual,
                hotspot,
                sineSettings,
                searchSettings,
                randomGenerator)}};
    choice -= genomeSettings.addWeight;
    std::vector<double> weights = calculateAtomSelectionWeights(audio,
        genome,
        fitness,
        residual.fftSize,
        residual.hopSize,
        searchSettings.atomSelectionExplorationWeight);
    const bool removal = choice < genomeSettings.removeWeight;
    if (removal) {
        for (std::size_t index = 0; index < genome.atoms.size(); ++index) {
            const GenomeAtom& atom = genome.atoms[index];
            const LocalFrameRange frames = atomFrames(
                audio, atom.atom, residual.fftSize, residual.hopSize);
            const std::size_t bin = frequencyToNearestBin(
                atom.atom.frequencyHz, residual.fftSize, residual.sampleRateHz);
            double negative = 0.0;
            for (std::size_t frame = frames.begin; frame < frames.end; ++frame)
                negative += std::max(
                    0.0, -residual.values[frame * residual.binCount + bin]);
            weights[index] *= 1.0 + negative;
        }
    }
    const GenomeAtom& selected =
        genome.atoms[chooseWeightedIndex(weights, randomGenerator)];
    if ((choice -= genomeSettings.removeWeight) < 0.0)
        return RemoveAtomMutation{selected.id};

    const LocalFrameRange frames =
        atomFrames(audio, selected.atom, residual.fftSize, residual.hopSize);
    const std::size_t currentBin = frequencyToNearestBin(
        selected.atom.frequencyHz, residual.fftSize, residual.sampleRateHz);
    std::size_t bestFrame = frames.begin;
    std::size_t bestBin = currentBin;
    double bestMagnitude = -1.0;
    const std::size_t binRadius = 8;
    const std::size_t firstBin =
        currentBin > binRadius ? currentBin - binRadius : 0;
    const std::size_t lastBin =
        std::min(residual.binCount - 1, currentBin + binRadius);
    for (std::size_t frame = frames.begin; frame < frames.end; ++frame) {
        for (std::size_t bin = firstBin; bin <= lastBin; ++bin) {
            const double magnitude =
                std::abs(residual.values[frame * residual.binCount + bin]);
            if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestFrame = frame;
                bestBin = bin;
            }
        }
    }
    const double binSpacing = static_cast<double>(audio.sampleRateHz)
        / static_cast<double>(residual.fftSize);
    if ((choice -= genomeSettings.moveWeight) < 0.0) {
        bestMagnitude = -1.0;
        for (std::size_t frame = 0; frame < residual.frameCount; ++frame) {
            const double magnitude = std::abs(
                residual.values[frame * residual.binCount + currentBin]);
            if (magnitude > bestMagnitude) {
                bestMagnitude = magnitude;
                bestFrame = frame;
            }
        }
        const double center = (static_cast<double>(bestFrame * residual.hopSize)
                                  + static_cast<double>(residual.fftSize) / 2.0)
            / static_cast<double>(audio.sampleRateHz);
        const double audioDuration = static_cast<double>(audio.samples.size())
            / static_cast<double>(audio.sampleRateHz);
        const double maximumStart =
            std::max(0.0, audioDuration - selected.atom.durationSeconds);
        return MoveAtomMutation{selected.id,
            selected.atom.startSeconds,
            std::clamp(center - selected.atom.durationSeconds / 2.0
                    + uniformReal(-0.5, 0.5, randomGenerator)
                        * static_cast<double>(residual.hopSize)
                        / static_cast<double>(audio.sampleRateHz),
                0.0,
                maximumStart)};
    }
    if ((choice -= genomeSettings.frequencyWeight) < 0.0) {
        return ChangeFrequencyMutation{selected.id,
            selected.atom.frequencyHz,
            std::clamp(binToFrequencyHz(
                           bestBin, residual.fftSize, residual.sampleRateHz)
                    + uniformReal(-0.5, 0.5, randomGenerator) * binSpacing,
                sineSettings.minimumFrequencyHz,
                std::min(sineSettings.maximumFrequencyHz,
                    static_cast<double>(audio.sampleRateHz) / 2.0))};
    }
    if ((choice -= genomeSettings.amplitudeWeight) < 0.0) {
        double signedResidual = 0.0;
        for (std::size_t frame = frames.begin; frame < frames.end; ++frame)
            signedResidual +=
                residual.values[frame * residual.binCount + currentBin];
        const double direction = signedResidual >= 0.0 ? 1.0 : -1.0;
        const double sign = selected.atom.amplitude < 0.0 ? -1.0 : 1.0;
        const double nextMagnitude =
            std::clamp(std::abs(selected.atom.amplitude)
                    + direction
                        * uniformReal(0.0,
                            genomeSettings.maximumAmplitudeDelta,
                            randomGenerator),
                0.0,
                sineSettings.maximumAbsoluteAmplitude);
        return ChangeAmplitudeMutation{
            selected.id, selected.atom.amplitude, sign * nextMagnitude};
    }
    if ((choice -= genomeSettings.durationWeight) < 0.0) {
        const auto persistence = persistentFrames(residual,
            bestFrame,
            bestBin,
            searchSettings.durationPersistenceThresholdRatio);
        const double suggested =
            static_cast<double>(
                (persistence.second - persistence.first) * residual.hopSize)
            / static_cast<double>(audio.sampleRateHz);
        const double audioDuration = static_cast<double>(audio.samples.size())
            / static_cast<double>(audio.sampleRateHz);
        const double maximum = std::min(sineSettings.maximumDurationSeconds,
            audioDuration - selected.atom.startSeconds);
        return ChangeDurationMutation{selected.id,
            selected.atom.durationSeconds,
            std::clamp(suggested * uniformReal(0.85, 1.15, randomGenerator),
                std::min(sineSettings.minimumDurationSeconds, maximum),
                maximum)};
    }
    return ChangePhaseMutation{selected.id,
        selected.atom.phaseRadians,
        wrapPhase(selected.atom.phaseRadians
            + uniformReal(-genomeSettings.maximumPhaseDeltaRadians,
                genomeSettings.maximumPhaseDeltaRadians,
                randomGenerator))};
}

GenomeMutation refineGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    const GenomeMutation& mutation,
    std::size_t variantIndex,
    std::size_t fftSize,
    std::size_t hopSize,
    const SineMutationSettings& sineSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator)
{
    validateSearchSettings(searchSettings);
    if (fftSize == 0 || hopSize == 0 || hopSize > fftSize)
        throw std::invalid_argument("Refinement STFT settings are invalid.");
    return std::visit([&](const auto& operation) -> GenomeMutation {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T, AddAtomMutation>) {
            SineAtom atom = operation.atom.atom;
            if (variantIndex < 4) {
                atom.phaseRadians =
                    static_cast<double>(variantIndex) * std::numbers::pi / 2.0;
            } else if (variantIndex < 12) {
                constexpr double scales[] = {
                    -1.0, -0.5, -0.25, -0.125, 0.125, 0.25, 0.5, 1.0};
                atom.amplitude = scales[variantIndex - 4]
                    * sineSettings.maximumAbsoluteAmplitude;
            } else {
                const double binSpacing =
                    static_cast<double>(audio.sampleRateHz)
                    / static_cast<double>(fftSize);
                atom.frequencyHz = std::clamp(atom.frequencyHz
                        + uniformReal(
                            -searchSettings.refinementFrequencyRadiusBins
                                * binSpacing,
                            searchSettings.refinementFrequencyRadiusBins
                                * binSpacing,
                            randomGenerator),
                    sineSettings.minimumFrequencyHz,
                    std::min(sineSettings.maximumFrequencyHz,
                        static_cast<double>(audio.sampleRateHz) / 2.0));
                atom.durationSeconds = std::clamp(atom.durationSeconds
                        * uniformReal(
                            1.0 - searchSettings.refinementDurationRatio,
                            1.0 + searchSettings.refinementDurationRatio,
                            randomGenerator),
                    sineSettings.minimumDurationSeconds,
                    std::min(sineSettings.maximumDurationSeconds,
                        static_cast<double>(audio.samples.size())
                            / static_cast<double>(audio.sampleRateHz)));
                const double maximumStart = std::max(0.0,
                    static_cast<double>(audio.samples.size())
                            / static_cast<double>(audio.sampleRateHz)
                        - atom.durationSeconds);
                atom.startSeconds = std::clamp(atom.startSeconds
                        + uniformReal(-searchSettings.refinementTimeRadiusHops,
                              searchSettings.refinementTimeRadiusHops,
                              randomGenerator)
                            * static_cast<double>(hopSize)
                            / static_cast<double>(audio.sampleRateHz),
                    0.0,
                    maximumStart);
            }
            return AddAtomMutation{{operation.atom.id, atom}};
        } else if constexpr (std::is_same_v<T, ChangeAmplitudeMutation>) {
            constexpr double offsets[] = {-1.0, -0.5, -0.25, 0.25, 0.5, 1.0};
            const GenomeAtom& atom = requireAtom(genome, operation.atomId);
            const double value = variantIndex < 6 ? atom.atom.amplitude
                    + offsets[variantIndex]
                        * sineSettings.maximumAbsoluteAmplitude * 0.25
                                                  : operation.amplitude
                    + uniformReal(-0.1, 0.1, randomGenerator)
                        * sineSettings.maximumAbsoluteAmplitude;
            return ChangeAmplitudeMutation{operation.atomId,
                atom.atom.amplitude,
                std::clamp(value,
                    -sineSettings.maximumAbsoluteAmplitude,
                    sineSettings.maximumAbsoluteAmplitude)};
        } else if constexpr (std::is_same_v<T, ChangePhaseMutation>) {
            constexpr double offsets[] = {-std::numbers::pi / 2.0,
                -std::numbers::pi / 4.0,
                std::numbers::pi / 4.0,
                std::numbers::pi / 2.0};
            const GenomeAtom& atom = requireAtom(genome, operation.atomId);
            return ChangePhaseMutation{operation.atomId,
                atom.atom.phaseRadians,
                wrapPhase(variantIndex < 4
                        ? atom.atom.phaseRadians + offsets[variantIndex]
                        : operation.phaseRadians
                            + uniformReal(-std::numbers::pi / 8.0,
                                std::numbers::pi / 8.0,
                                randomGenerator))};
        } else if constexpr (std::is_same_v<T, MoveAtomMutation>) {
            const GenomeAtom& atom = requireAtom(genome, operation.atomId);
            const double maximumStart = std::max(0.0,
                static_cast<double>(audio.samples.size())
                        / static_cast<double>(audio.sampleRateHz)
                    - atom.atom.durationSeconds);
            return MoveAtomMutation{operation.atomId,
                atom.atom.startSeconds,
                std::clamp(operation.startSeconds
                        + uniformReal(-1.0, 1.0, randomGenerator)
                            * searchSettings.refinementTimeRadiusHops
                            * static_cast<double>(hopSize)
                            / static_cast<double>(audio.sampleRateHz),
                    0.0,
                    maximumStart)};
        } else if constexpr (std::is_same_v<T, ChangeFrequencyMutation>) {
            const GenomeAtom& atom = requireAtom(genome, operation.atomId);
            const double binSpacing = static_cast<double>(audio.sampleRateHz)
                / static_cast<double>(fftSize);
            return ChangeFrequencyMutation{operation.atomId,
                atom.atom.frequencyHz,
                std::clamp(operation.frequencyHz
                        + uniformReal(-1.0, 1.0, randomGenerator)
                            * searchSettings.refinementFrequencyRadiusBins
                            * binSpacing,
                    sineSettings.minimumFrequencyHz,
                    std::min(sineSettings.maximumFrequencyHz,
                        static_cast<double>(audio.sampleRateHz) / 2.0))};
        } else if constexpr (std::is_same_v<T, ChangeDurationMutation>) {
            const GenomeAtom& atom = requireAtom(genome, operation.atomId);
            const double maximum = std::min(sineSettings.maximumDurationSeconds,
                static_cast<double>(audio.samples.size())
                        / static_cast<double>(audio.sampleRateHz)
                    - atom.atom.startSeconds);
            return ChangeDurationMutation{operation.atomId,
                atom.atom.durationSeconds,
                std::clamp(operation.durationSeconds
                        * uniformReal(
                            1.0 - searchSettings.refinementDurationRatio,
                            1.0 + searchSettings.refinementDurationRatio,
                            randomGenerator),
                    std::min(sineSettings.minimumDurationSeconds, maximum),
                    maximum)};
        } else {
            return operation;
        }
    }, mutation);
}
