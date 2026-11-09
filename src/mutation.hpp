#pragma once

#include "audio.hpp"
#include "fitness.hpp"
#include "synthesis.hpp"

#include <cstdint>
#include <random>
#include <variant>
#include <vector>

using RandomSeed = std::uint64_t;

struct SineMutationSettings {
    double minimumFrequencyHz = 20.0;
    double maximumFrequencyHz = 20'000.0;
    double minimumDurationSeconds = 0.01;
    double maximumDurationSeconds = 0.50;
    double maximumAbsoluteAmplitude = 0.25;
};

using AtomId = std::uint64_t;

struct GenomeAtom {
    AtomId id = 0;
    SineAtom atom;
    bool operator==(const GenomeAtom&) const = default;
};

struct Genome {
    std::vector<GenomeAtom> atoms;
};

struct AddAtomMutation {
    GenomeAtom atom;
    bool operator==(const AddAtomMutation&) const = default;
};
struct RemoveAtomMutation {
    AtomId atomId = 0;
    bool operator==(const RemoveAtomMutation&) const = default;
};
struct MoveAtomMutation {
    AtomId atomId = 0;
    double previousStartSeconds = 0.0;
    double startSeconds = 0.0;
    bool operator==(const MoveAtomMutation&) const = default;
};
struct ChangeFrequencyMutation {
    AtomId atomId = 0;
    double previousFrequencyHz = 0.0;
    double frequencyHz = 0.0;
    bool operator==(const ChangeFrequencyMutation&) const = default;
};
struct ChangeAmplitudeMutation {
    AtomId atomId = 0;
    double previousAmplitude = 0.0;
    double amplitude = 0.0;
    bool operator==(const ChangeAmplitudeMutation&) const = default;
};
struct ChangeDurationMutation {
    AtomId atomId = 0;
    double previousDurationSeconds = 0.0;
    double durationSeconds = 0.0;
    bool operator==(const ChangeDurationMutation&) const = default;
};
struct ChangePhaseMutation {
    AtomId atomId = 0;
    double previousPhaseRadians = 0.0;
    double phaseRadians = 0.0;
    bool operator==(const ChangePhaseMutation&) const = default;
};

using GenomeMutation = std::variant<AddAtomMutation,
    RemoveAtomMutation,
    MoveAtomMutation,
    ChangeFrequencyMutation,
    ChangeAmplitudeMutation,
    ChangeDurationMutation,
    ChangePhaseMutation>;

struct GenomeMutationSettings {
    double addWeight = 0.50;
    double removeWeight = 0.10;
    double moveWeight = 0.10;
    double frequencyWeight = 0.10;
    double amplitudeWeight = 0.10;
    double durationWeight = 0.05;
    double phaseWeight = 0.05;
    double maximumMoveSeconds = 0.10;
    double maximumFrequencyRatio = 1.25;
    double maximumAmplitudeDelta = 0.05;
    double maximumDurationDeltaSeconds = 0.05;
    double maximumPhaseDeltaRadians = 0.7853981633974483;
};

struct GuidedSearchSettings {
    double guidedMutationRatio = 0.80;
    HotspotSettings hotspots;
    double guidedFrequencyRadiusBins = 2.0;
    double guidedTimeRadiusHops = 2.0;
    double durationPersistenceThresholdRatio = 0.25;
    double atomSelectionExplorationWeight = 0.05;
    double refinementBudgetRatio = 0.40;
    std::size_t refinementParentCount = 16;
    double refinementFrequencyRadiusBins = 0.5;
    double refinementTimeRadiusHops = 0.5;
    double refinementDurationRatio = 0.15;
};

enum class MutationProposalSource {
    Random,
    GuidedHotspot,
    Refinement,
    ParameterSearch,
};

RandomSeed createRandomSeed();

void validateSineMutationSettings(
    const AudioBuffer& audio, const SineMutationSettings& settings);

SineAtom createRandomSineAtom(const AudioBuffer& audio,
    const SineMutationSettings& settings,
    std::mt19937_64& randomGenerator);

const GenomeAtom* findGenomeAtom(const Genome& genome, AtomId atomId);
void validateGenome(const AudioBuffer& source, const Genome& genome);
AudioBuffer renderGenome(const AudioBuffer& source, const Genome& genome);
void applyGenomeMutationToAudio(AudioBuffer& renderedAudio,
    const Genome& genome,
    const GenomeMutation& mutation);
void applyGenomeMutation(
    Genome& genome, AudioBuffer& renderedAudio, const GenomeMutation& mutation);
GenomeMutation createRandomGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    AtomId addAtomId,
    const SineMutationSettings& sineSettings,
    const GenomeMutationSettings& genomeSettings,
    std::mt19937_64& randomGenerator);

std::vector<double> calculateAtomSelectionWeights(const AudioBuffer& audio,
    const Genome& genome,
    const FitnessState& fitness,
    std::size_t fftSize,
    std::size_t hopSize,
    double explorationWeight);

SineAtom createGuidedSineAtom(const AudioBuffer& audio,
    const SpectralResidual& residual,
    const ResidualHotspot& hotspot,
    const SineMutationSettings& sineSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator);

GenomeMutation createGuidedGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    AtomId addAtomId,
    const SineMutationSettings& sineSettings,
    const GenomeMutationSettings& genomeSettings,
    const GuidedSearchSettings& searchSettings,
    const SpectralResidual& residual,
    const ResidualHotspot& hotspot,
    const FitnessState& fitness,
    std::mt19937_64& randomGenerator);

GenomeMutation refineGenomeMutation(const AudioBuffer& audio,
    const Genome& genome,
    const GenomeMutation& mutation,
    std::size_t variantIndex,
    std::size_t fftSize,
    std::size_t hopSize,
    const SineMutationSettings& sineSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator);
