#pragma once

#include "audio.hpp"
#include "fitness.hpp"
#include "mutation.hpp"
#include "synthesis.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <random>
#include <string_view>
#include <vector>

enum class GenerationLimitMode {
    Attempts,
    SuccessfulWrites,
};

struct EvolutionSettings {
    std::size_t candidatesPerGeneration = 1000;
    std::size_t maximumGenerations = 100;
    GenerationLimitMode generationLimitMode = GenerationLimitMode::Attempts;
    std::size_t maximumAcceptedMutationsPerGeneration = 8;
    std::optional<double> targetFitness;
    std::optional<std::size_t> stagnationGenerations;
};

struct FrameRange {
    std::size_t beginFrame = 0;
    std::size_t endFrame = 0;
};

FrameRange affectedFrameRange(const AudioBuffer& audio,
    const SineAtom& atom,
    std::size_t fftSize,
    std::size_t hopSize);
bool frameRangesOverlap(const FrameRange& left, const FrameRange& right);

struct GenerationResult {
    AudioBuffer audio;
    double previousFitness = 0.0;
    double fitness = 0.0;
    bool improved = false;
    std::vector<SineAtom> winningMutations;
    std::size_t candidatesTested = 0;
};

struct GenerationSummary {
    std::size_t generation = 0;
    std::optional<double> previousFitness;
    double fitness = 0.0;
    double improvement = 0.0;
    bool improved = false;
    std::size_t candidatesTested = 0;
    std::vector<GenomeMutation> winningMutations;
    std::size_t guidedCandidatesTested = 0;
    std::size_t randomCandidatesTested = 0;
    std::size_t refinementCandidatesTested = 0;
    std::size_t guidedWinnersAccepted = 0;
    std::size_t randomWinnersAccepted = 0;
    std::size_t refinementWinnersAccepted = 0;
};

enum class EvolutionStopReason {
    MaximumGenerations,
    MaximumSuccessfulWrites,
    TargetFitnessReached,
    Stagnation,
};

struct EvolutionRunResult {
    AudioBuffer finalAudio;
    double initialFitness = 0.0;
    double finalFitness = 0.0;
    std::size_t completedGenerations = 0;
    std::size_t successfulWrites = 0;
    EvolutionStopReason stopReason = EvolutionStopReason::MaximumGenerations;
    std::vector<GenerationSummary> history;
    Genome finalGenome;
    std::size_t guidedCandidatesTested = 0;
    std::size_t randomCandidatesTested = 0;
    std::size_t refinementCandidatesTested = 0;
    std::size_t guidedWinnersAccepted = 0;
    std::size_t randomWinnersAccepted = 0;
    std::size_t refinementWinnersAccepted = 0;
};

struct EvolutionState {
    Genome genome;
    AudioBuffer renderedAudio;
    Spectrogram spectrogram;
    FitnessState fitness;
    AtomId nextAtomId = 1;
};

struct GenomeGenerationResult {
    double previousFitness = 0.0;
    double fitness = 0.0;
    std::vector<GenomeMutation> winningMutations;
    std::size_t candidatesTested = 0;
    std::size_t guidedCandidatesTested = 0;
    std::size_t randomCandidatesTested = 0;
    std::size_t refinementCandidatesTested = 0;
    std::size_t guidedWinnersAccepted = 0;
    std::size_t randomWinnersAccepted = 0;
    std::size_t refinementWinnersAccepted = 0;
};

EvolutionState createEvolutionState(const AudioBuffer& source,
    const FitnessReference& target,
    StftAnalyzer& analyzer);

FrameRange affectedFrameRange(const AudioBuffer& audio,
    const Genome& genome,
    const GenomeMutation& mutation,
    std::size_t fftSize,
    std::size_t hopSize);

GenomeGenerationResult evaluateGenomeMutations(EvolutionState& state,
    const FitnessReference& target,
    const std::vector<GenomeMutation>& mutations,
    std::size_t maximumAcceptedMutations,
    StftAnalyzer& analyzer);

using GenerationObserver =
    std::function<void(const GenerationSummary&, const AudioBuffer&)>;

std::string_view evolutionStopReasonValue(EvolutionStopReason reason);
std::string_view generationLimitModeValue(GenerationLimitMode mode);

std::size_t updateStagnationCount(std::size_t currentCount, bool improved);

std::optional<EvolutionStopReason> determineStopReason(double currentFitness,
    std::size_t completedGenerations,
    std::size_t successfulWrites,
    std::size_t generationsWithoutImprovement,
    const EvolutionSettings& settings);

GenerationResult evaluateMutations(const AudioBuffer& currentAudio,
    double currentFitness,
    const FitnessReference& target,
    const std::vector<SineAtom>& mutations,
    std::size_t maximumAcceptedMutations = 8);

GenerationResult evolveOneGeneration(const AudioBuffer& currentAudio,
    double currentFitness,
    const FitnessReference& target,
    const EvolutionSettings& evolutionSettings,
    const SineMutationSettings& mutationSettings,
    std::mt19937_64& randomGenerator);

EvolutionRunResult evolve(const AudioBuffer& source,
    double initialFitness,
    const FitnessReference& target,
    const EvolutionSettings& settings,
    const SineMutationSettings& mutationSettings,
    std::mt19937_64& randomGenerator,
    const GenerationObserver& observer = {});

EvolutionRunResult evolve(const AudioBuffer& source,
    double initialFitness,
    const FitnessReference& target,
    const EvolutionSettings& settings,
    const SineMutationSettings& mutationSettings,
    const GenomeMutationSettings& genomeMutationSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator,
    StftAnalyzer& analyzer,
    const GenerationObserver& observer = {});
