#include "evolution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

void validateParent(const AudioBuffer& audio, double fitness)
{
    if (audio.samples.empty() || audio.sampleRateHz == 0)
        throw std::invalid_argument("Cannot evolve empty or invalid audio.");
    if (!std::isfinite(fitness))
        throw std::invalid_argument("Current fitness must be finite.");
}

void validateEvolutionSettings(const EvolutionSettings& settings)
{
    if (settings.maximumGenerations == 0
        || settings.maximumGenerations
            == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("Maximum generation count is invalid.");
    if (settings.candidatesPerGeneration == 0)
        throw std::invalid_argument(
            "Candidate count must be greater than zero.");
    if (settings.maximumAcceptedMutationsPerGeneration == 0)
        throw std::invalid_argument(
            "Maximum accepted mutation count must be positive.");
    if (settings.targetFitness
        && (!std::isfinite(*settings.targetFitness)
            || *settings.targetFitness < 0.0))
        throw std::invalid_argument(
            "Target fitness must be finite and non-negative.");
    if (settings.stagnationGenerations && *settings.stagnationGenerations == 0)
        throw std::invalid_argument(
            "Stagnation generation count must be positive.");
}

void notifyObserver(const GenerationObserver& observer,
    const GenerationSummary& summary,
    const AudioBuffer& audio)
{
    if (!observer)
        return;
    try {
        observer(summary, audio);
    } catch (const std::exception& error) {
        throw std::runtime_error("Evolution failed while persisting generation "
            + std::to_string(summary.generation) + ": " + error.what());
    }
}

FrameRange combineRanges(FrameRange left, FrameRange right)
{
    if (left.beginFrame == left.endFrame)
        return right;
    if (right.beginFrame == right.endFrame)
        return left;
    return {std::min(left.beginFrame, right.beginFrame),
        std::max(left.endFrame, right.endFrame)};
}

const GenomeAtom& requireAtom(const Genome& genome, AtomId id)
{
    const GenomeAtom* atom = findGenomeAtom(genome, id);
    if (!atom)
        throw std::invalid_argument(
            "Genome mutation references an unknown atom ID.");
    return *atom;
}

std::optional<AtomId> editedAtomId(const GenomeMutation& mutation)
{
    return std::visit([](const auto& operation) -> std::optional<AtomId> {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T, AddAtomMutation>)
            return operation.atom.id;
        else
            return operation.atomId;
    }, mutation);
}

FrameRange mutationFrameRange(const AudioBuffer& audio,
    const Genome& genome,
    const GenomeMutation& mutation,
    std::size_t fftSize,
    std::size_t hopSize)
{
    return std::visit([&](const auto& operation) -> FrameRange {
        using T = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<T, AddAtomMutation>) {
            return affectedFrameRange(
                audio, operation.atom.atom, fftSize, hopSize);
        } else {
            const SineAtom oldAtom = requireAtom(genome, operation.atomId).atom;
            if constexpr (std::is_same_v<T, RemoveAtomMutation>) {
                return affectedFrameRange(audio, oldAtom, fftSize, hopSize);
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
                else if constexpr (std::is_same_v<T, ChangePhaseMutation>)
                    replacement.phaseRadians = operation.phaseRadians;
                return combineRanges(
                    affectedFrameRange(audio, oldAtom, fftSize, hopSize),
                    affectedFrameRange(audio, replacement, fftSize, hopSize));
            }
        }
    }, mutation);
}

struct EvaluatedMutation {
    GenomeMutation mutation;
    FrameRange frames;
    std::size_t candidateIndex = 0;
    double fitness = 0.0;
    double improvement = 0.0;
    MutationProposalSource source = MutationProposalSource::Random;
};

struct CandidateProposal {
    GenomeMutation mutation;
    MutationProposalSource source = MutationProposalSource::Random;
};

double scoreCandidate(const EvolutionState& state,
    const FitnessReference& target,
    const GenomeMutation& mutation,
    FrameRange frames,
    StftAnalyzer& analyzer)
{
    AudioBuffer candidateAudio = state.renderedAudio;
    applyGenomeMutationToAudio(candidateAudio, state.genome, mutation);
    double total = state.fitness.totalSquaredError;
    std::vector<double> magnitudes(target.binCount);
    for (std::size_t frame = frames.beginFrame; frame < frames.endFrame;
         ++frame) {
        analyzer.calculateFrameMagnitudes(candidateAudio, frame, magnitudes);
        total -= state.fitness.frameSquaredErrors.at(frame);
        total += calculateFrameSquaredError(magnitudes, target, frame);
    }
    const std::size_t values = target.frameCount * target.binCount;
    return values == 0 ? 0.0 : total / static_cast<double>(values);
}

std::vector<EvaluatedMutation> scoreCandidates(const EvolutionState& state,
    const FitnessReference& target,
    const std::vector<CandidateProposal>& proposals,
    std::size_t indexOffset,
    StftAnalyzer& analyzer)
{
    std::vector<EvaluatedMutation> improving;
    for (std::size_t index = 0; index < proposals.size(); ++index) {
        const GenomeMutation& mutation = proposals[index].mutation;
        const FrameRange frames = mutationFrameRange(state.renderedAudio,
            state.genome,
            mutation,
            target.fftSize,
            target.hopSize);
        const double fitness =
            scoreCandidate(state, target, mutation, frames, analyzer);
        if (!std::isfinite(fitness))
            throw std::runtime_error("Candidate fitness is not finite.");
        if (fitness < state.fitness.fitness) {
            improving.push_back({mutation,
                frames,
                indexOffset + index,
                fitness,
                state.fitness.fitness - fitness,
                proposals[index].source});
        }
    }
    std::stable_sort(improving.begin(),
        improving.end(),
        [](const EvaluatedMutation& left, const EvaluatedMutation& right) {
        if (left.improvement != right.improvement)
            return left.improvement > right.improvement;
        return left.candidateIndex < right.candidateIndex;
    });
    return improving;
}

std::vector<EvaluatedMutation> selectCompatibleCandidates(
    std::vector<EvaluatedMutation> improving, std::size_t maximumAccepted)
{
    std::stable_sort(improving.begin(),
        improving.end(),
        [](const EvaluatedMutation& left, const EvaluatedMutation& right) {
        if (left.improvement != right.improvement)
            return left.improvement > right.improvement;
        return left.candidateIndex < right.candidateIndex;
    });
    std::vector<EvaluatedMutation> selected;
    for (const EvaluatedMutation& candidate : improving) {
        if (selected.size() == maximumAccepted)
            break;
        const auto candidateAtom = editedAtomId(candidate.mutation);
        const bool conflict = std::any_of(selected.begin(),
            selected.end(),
            [&](const EvaluatedMutation& winner) {
            const auto winnerAtom = editedAtomId(winner.mutation);
            return frameRangesOverlap(candidate.frames, winner.frames)
                || (candidateAtom && winnerAtom == candidateAtom);
        });
        if (!conflict)
            selected.push_back(candidate);
    }
    return selected;
}

void commitMutation(EvolutionState& state,
    const FitnessReference& target,
    const EvaluatedMutation& selected,
    StftAnalyzer& analyzer)
{
    applyGenomeMutation(state.genome, state.renderedAudio, selected.mutation);
    std::vector<double> magnitudes(target.binCount);
    for (std::size_t frame = selected.frames.beginFrame;
         frame < selected.frames.endFrame;
         ++frame) {
        analyzer.calculateFrameMagnitudes(
            state.renderedAudio, frame, magnitudes);
        const double replacement =
            calculateFrameSquaredError(magnitudes, target, frame);
        state.fitness.totalSquaredError -=
            state.fitness.frameSquaredErrors[frame];
        state.fitness.totalSquaredError += replacement;
        state.fitness.frameSquaredErrors[frame] = replacement;
        std::copy(magnitudes.begin(),
            magnitudes.end(),
            state.spectrogram.magnitudes.begin()
                + static_cast<std::ptrdiff_t>(frame * target.binCount));
    }
    const std::size_t values = target.frameCount * target.binCount;
    state.fitness.fitness = values == 0
        ? 0.0
        : state.fitness.totalSquaredError / static_cast<double>(values);
}

std::vector<EvaluatedMutation> commitSelectedMutations(EvolutionState& state,
    const FitnessReference& target,
    std::vector<EvaluatedMutation> selected,
    StftAnalyzer& analyzer)
{
    const double previous = state.fitness.fitness;
    const EvolutionState parent = state;
    for (const EvaluatedMutation& winner : selected)
        commitMutation(state, target, winner, analyzer);
    if (!selected.empty() && !(state.fitness.fitness < previous)) {
        state = parent;
        commitMutation(state, target, selected.front(), analyzer);
        if (!(state.fitness.fitness < previous)) {
            state = parent;
            selected.clear();
        } else {
            selected.resize(1);
        }
    }
    return selected;
}

GenomeGenerationResult evaluateGenomeMutationsImplementation(
    EvolutionState& state,
    const FitnessReference& target,
    const std::vector<GenomeMutation>& mutations,
    std::size_t maximumAccepted,
    StftAnalyzer& analyzer)
{
    const double previous = state.fitness.fitness;
    std::vector<CandidateProposal> proposals;
    proposals.reserve(mutations.size());
    for (const GenomeMutation& mutation : mutations)
        proposals.push_back({mutation, MutationProposalSource::Random});
    std::vector<EvaluatedMutation> selected = selectCompatibleCandidates(
        scoreCandidates(state, target, proposals, 0, analyzer),
        maximumAccepted);
    selected =
        commitSelectedMutations(state, target, std::move(selected), analyzer);
    std::vector<GenomeMutation> winners;
    winners.reserve(selected.size());
    for (EvaluatedMutation& winner : selected)
        winners.push_back(std::move(winner.mutation));
    return {previous,
        state.fitness.fitness,
        std::move(winners),
        mutations.size(),
        0,
        mutations.size(),
        0,
        0,
        selected.size(),
        0};
}

GenomeGenerationResult evolveGuidedGeneration(const AudioBuffer& source,
    EvolutionState& state,
    const FitnessReference& target,
    const EvolutionSettings& evolutionSettings,
    const SineMutationSettings& sineSettings,
    const GenomeMutationSettings& genomeSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator,
    StftAnalyzer& analyzer)
{
    if (!std::isfinite(searchSettings.guidedMutationRatio)
        || searchSettings.guidedMutationRatio < 0.0
        || searchSettings.guidedMutationRatio > 1.0
        || !std::isfinite(searchSettings.refinementBudgetRatio)
        || searchSettings.refinementBudgetRatio < 0.0
        || searchSettings.refinementBudgetRatio >= 1.0
        || searchSettings.refinementParentCount == 0)
        throw std::invalid_argument("Guided search ratios are invalid.");

    const std::size_t totalBudget = evolutionSettings.candidatesPerGeneration;
    const std::size_t refinementBudget = totalBudget <= 1
        ? 0
        : std::min(totalBudget - 1,
              static_cast<std::size_t>(
                  std::llround(static_cast<double>(totalBudget)
                      * searchSettings.refinementBudgetRatio)));
    const std::size_t stageOneBudget = totalBudget - refinementBudget;
    const SpectralResidual residual =
        calculateSpectralResidual(state.spectrogram, target);
    const std::vector<ResidualHotspot> hotspots =
        findResidualHotspots(residual, searchSettings.hotspots);
    const std::size_t requestedGuided = std::min(stageOneBudget,
        static_cast<std::size_t>(
            std::llround(static_cast<double>(stageOneBudget)
                * searchSettings.guidedMutationRatio)));

    std::vector<CandidateProposal> stageOne;
    stageOne.reserve(stageOneBudget);
    std::size_t guidedTested = 0;
    std::size_t randomTested = 0;
    for (std::size_t index = 0; index < stageOneBudget; ++index) {
        const AtomId proposalId = state.nextAtomId++;
        if (index < requestedGuided && !hotspots.empty()) {
            stageOne.push_back({createGuidedGenomeMutation(source,
                                    state.genome,
                                    proposalId,
                                    sineSettings,
                                    genomeSettings,
                                    searchSettings,
                                    residual,
                                    hotspots[index % hotspots.size()],
                                    state.fitness,
                                    randomGenerator),
                MutationProposalSource::GuidedHotspot});
            ++guidedTested;
        } else {
            stageOne.push_back({createRandomGenomeMutation(source,
                                    state.genome,
                                    proposalId,
                                    sineSettings,
                                    genomeSettings,
                                    randomGenerator),
                MutationProposalSource::Random});
            ++randomTested;
        }
    }
    std::vector<EvaluatedMutation> stageOneImproving =
        scoreCandidates(state, target, stageOne, 0, analyzer);

    std::vector<const EvaluatedMutation*> refinementParents;
    refinementParents.reserve(searchSettings.refinementParentCount);
    for (const EvaluatedMutation& candidate : stageOneImproving) {
        if (!std::holds_alternative<RemoveAtomMutation>(candidate.mutation))
            refinementParents.push_back(&candidate);
        if (refinementParents.size() == searchSettings.refinementParentCount)
            break;
    }
    const std::size_t parentCount = refinementParents.size();
    std::vector<CandidateProposal> refinements;
    if (parentCount != 0) {
        refinements.reserve(refinementBudget);
        for (std::size_t index = 0; index < refinementBudget; ++index) {
            constexpr std::size_t structuredVariantCount = 12;
            const std::size_t parentIndex =
                (index / structuredVariantCount) % parentCount;
            const std::size_t cycle =
                index / (structuredVariantCount * parentCount);
            const std::size_t variantIndex =
                index % structuredVariantCount + cycle * structuredVariantCount;
            const EvaluatedMutation& parent = *refinementParents[parentIndex];
            const bool parameterSearch =
                (std::holds_alternative<AddAtomMutation>(parent.mutation)
                    && variantIndex < structuredVariantCount)
                || std::holds_alternative<ChangeAmplitudeMutation>(
                    parent.mutation)
                || std::holds_alternative<ChangePhaseMutation>(parent.mutation);
            refinements.push_back({refineGenomeMutation(source,
                                       state.genome,
                                       parent.mutation,
                                       variantIndex,
                                       target.fftSize,
                                       target.hopSize,
                                       sineSettings,
                                       searchSettings,
                                       randomGenerator),
                parameterSearch ? MutationProposalSource::ParameterSearch
                                : MutationProposalSource::Refinement});
        }
    }
    std::vector<EvaluatedMutation> refinementImproving =
        scoreCandidates(state, target, refinements, stageOne.size(), analyzer);
    stageOneImproving.insert(stageOneImproving.end(),
        refinementImproving.begin(),
        refinementImproving.end());
    std::vector<EvaluatedMutation> selected =
        selectCompatibleCandidates(std::move(stageOneImproving),
            evolutionSettings.maximumAcceptedMutationsPerGeneration);
    const double previous = state.fitness.fitness;
    selected =
        commitSelectedMutations(state, target, std::move(selected), analyzer);

    std::size_t guidedWinners = 0;
    std::size_t randomWinners = 0;
    std::size_t refinementWinners = 0;
    std::vector<GenomeMutation> winners;
    winners.reserve(selected.size());
    for (EvaluatedMutation& winner : selected) {
        if (winner.source == MutationProposalSource::GuidedHotspot)
            ++guidedWinners;
        else if (winner.source == MutationProposalSource::Random)
            ++randomWinners;
        else
            ++refinementWinners;
        winners.push_back(std::move(winner.mutation));
    }
    return {previous,
        state.fitness.fitness,
        std::move(winners),
        stageOne.size() + refinements.size(),
        guidedTested,
        randomTested,
        refinements.size(),
        guidedWinners,
        randomWinners,
        refinementWinners};
}

EvolutionState createEvolutionStateImplementation(const AudioBuffer& source,
    const FitnessReference& target,
    StftAnalyzer& analyzer)
{
    EvolutionState state;
    state.renderedAudio = source;
    state.spectrogram = analyzer.calculate(source);
    state.fitness = createFitnessState(state.spectrogram, target);
    return state;
}

}  // namespace

FrameRange affectedFrameRange(const AudioBuffer& audio,
    const SineAtom& atom,
    std::size_t fftSize,
    std::size_t hopSize)
{
    if (fftSize == 0 || hopSize == 0 || hopSize > fftSize)
        throw std::invalid_argument(
            "Invalid STFT settings for affected range.");
    const SampleRange samples = sineAtomSampleRange(audio, atom);
    const std::size_t frameCount =
        audio.samples.empty() ? 0 : 1 + (audio.samples.size() - 1) / hopSize;
    if (samples.beginSample == samples.endSample)
        return {0, 0};
    const std::size_t begin = samples.beginSample < fftSize
        ? 0
        : (samples.beginSample - fftSize) / hopSize + 1;
    const std::size_t end = std::min(frameCount,
        samples.endSample / hopSize
            + (samples.endSample % hopSize != 0 ? 1U : 0U));
    return {std::min(begin, frameCount), end};
}

bool frameRangesOverlap(const FrameRange& left, const FrameRange& right)
{
    return left.beginFrame < right.endFrame && right.beginFrame < left.endFrame;
}

EvolutionState createEvolutionState(const AudioBuffer& source,
    const FitnessReference& target,
    StftAnalyzer& analyzer)
{
    return createEvolutionStateImplementation(source, target, analyzer);
}

FrameRange affectedFrameRange(const AudioBuffer& audio,
    const Genome& genome,
    const GenomeMutation& mutation,
    std::size_t fftSize,
    std::size_t hopSize)
{
    return mutationFrameRange(audio, genome, mutation, fftSize, hopSize);
}

GenomeGenerationResult evaluateGenomeMutations(EvolutionState& state,
    const FitnessReference& target,
    const std::vector<GenomeMutation>& mutations,
    std::size_t maximumAcceptedMutations,
    StftAnalyzer& analyzer)
{
    if (maximumAcceptedMutations == 0)
        throw std::invalid_argument(
            "Maximum accepted mutation count must be positive.");
    return evaluateGenomeMutationsImplementation(
        state, target, mutations, maximumAcceptedMutations, analyzer);
}

std::string_view evolutionStopReasonValue(EvolutionStopReason reason)
{
    switch (reason) {
    case EvolutionStopReason::MaximumGenerations:
        return "maximum-generations";
    case EvolutionStopReason::MaximumSuccessfulWrites:
        return "maximum-successful-writes";
    case EvolutionStopReason::TargetFitnessReached:
        return "target-fitness";
    case EvolutionStopReason::Stagnation:
        return "stagnation";
    }
    throw std::invalid_argument("Unknown evolution stop reason.");
}

std::string_view generationLimitModeValue(GenerationLimitMode mode)
{
    switch (mode) {
    case GenerationLimitMode::Attempts:
        return "attempts";
    case GenerationLimitMode::SuccessfulWrites:
        return "successful-writes";
    }
    throw std::invalid_argument("Unknown generation limit mode.");
}

std::size_t updateStagnationCount(std::size_t currentCount, bool improved)
{
    if (improved)
        return 0;
    return currentCount == std::numeric_limits<std::size_t>::max()
        ? currentCount
        : currentCount + 1;
}

std::optional<EvolutionStopReason> determineStopReason(double currentFitness,
    std::size_t completedGenerations,
    std::size_t successfulWrites,
    std::size_t generationsWithoutImprovement,
    const EvolutionSettings& settings)
{
    validateEvolutionSettings(settings);
    if (!std::isfinite(currentFitness))
        throw std::invalid_argument("Current fitness must be finite.");
    if (settings.targetFitness && currentFitness <= *settings.targetFitness)
        return EvolutionStopReason::TargetFitnessReached;
    if (settings.stagnationGenerations
        && generationsWithoutImprovement >= *settings.stagnationGenerations)
        return EvolutionStopReason::Stagnation;
    if (settings.generationLimitMode == GenerationLimitMode::Attempts
        && completedGenerations >= settings.maximumGenerations)
        return EvolutionStopReason::MaximumGenerations;
    if (settings.generationLimitMode == GenerationLimitMode::SuccessfulWrites
        && successfulWrites >= settings.maximumGenerations)
        return EvolutionStopReason::MaximumSuccessfulWrites;
    return std::nullopt;
}

GenerationResult evaluateMutations(const AudioBuffer& currentAudio,
    double currentFitness,
    const FitnessReference& target,
    const std::vector<SineAtom>& mutations,
    std::size_t maximumAcceptedMutations)
{
    validateParent(currentAudio, currentFitness);
    if (maximumAcceptedMutations == 0)
        throw std::invalid_argument(
            "Maximum accepted mutation count must be positive.");
    StftAnalyzer analyzer(target.fftSize, target.hopSize);
    EvolutionState state = createEvolutionState(currentAudio, target, analyzer);
    std::vector<GenomeMutation> operations;
    AtomId id = 1;
    for (const SineAtom& atom : mutations)
        operations.push_back(AddAtomMutation{{id++, atom}});
    GenomeGenerationResult generation = evaluateGenomeMutations(
        state, target, operations, maximumAcceptedMutations, analyzer);
    std::vector<SineAtom> winners;
    for (const GenomeMutation& mutation : generation.winningMutations)
        winners.push_back(std::get<AddAtomMutation>(mutation).atom.atom);
    return {std::move(state.renderedAudio),
        currentFitness,
        state.fitness.fitness,
        !winners.empty(),
        std::move(winners),
        mutations.size()};
}

GenerationResult evolveOneGeneration(const AudioBuffer& currentAudio,
    double currentFitness,
    const FitnessReference& target,
    const EvolutionSettings& settings,
    const SineMutationSettings& mutationSettings,
    std::mt19937_64& randomGenerator)
{
    validateParent(currentAudio, currentFitness);
    validateEvolutionSettings(settings);
    std::vector<SineAtom> mutations;
    mutations.reserve(settings.candidatesPerGeneration);
    for (std::size_t index = 0; index < settings.candidatesPerGeneration;
         ++index)
        mutations.push_back(createRandomSineAtom(
            currentAudio, mutationSettings, randomGenerator));
    return evaluateMutations(currentAudio,
        currentFitness,
        target,
        mutations,
        settings.maximumAcceptedMutationsPerGeneration);
}

EvolutionRunResult evolve(const AudioBuffer& source,
    double initialFitness,
    const FitnessReference& target,
    const EvolutionSettings& settings,
    const SineMutationSettings& mutationSettings,
    const GenomeMutationSettings& genomeMutationSettings,
    const GuidedSearchSettings& searchSettings,
    std::mt19937_64& randomGenerator,
    StftAnalyzer& analyzer,
    const GenerationObserver& observer)
{
    validateParent(source, initialFitness);
    validateEvolutionSettings(settings);
    validateSineMutationSettings(source, mutationSettings);
    if (analyzer.fftSize() != target.fftSize
        || analyzer.hopSize() != target.hopSize)
        throw std::invalid_argument(
            "Evolution analyzer settings do not match target.");
    EvolutionState state = createEvolutionState(source, target, analyzer);
    std::vector<GenerationSummary> history;
    history.reserve(settings.maximumGenerations + 1);
    history.push_back(
        {0, std::nullopt, state.fitness.fitness, 0.0, false, 0, {}});
    notifyObserver(observer, history.back(), state.renderedAudio);
    std::size_t completed = 0;
    std::size_t successfulWrites = 1;
    std::size_t stagnant = 0;
    std::size_t guidedCandidates = 0;
    std::size_t randomCandidates = 0;
    std::size_t refinementCandidates = 0;
    std::size_t guidedWinners = 0;
    std::size_t randomWinners = 0;
    std::size_t refinementWinners = 0;
    auto stop = determineStopReason(
        state.fitness.fitness, completed, successfulWrites, stagnant, settings);
    while (!stop) {
        GenomeGenerationResult generation = evolveGuidedGeneration(source,
            state,
            target,
            settings,
            mutationSettings,
            genomeMutationSettings,
            searchSettings,
            randomGenerator,
            analyzer);
        ++completed;
        const bool improved = !generation.winningMutations.empty();
        if (improved)
            ++successfulWrites;
        stagnant = updateStagnationCount(stagnant, improved);
        guidedCandidates += generation.guidedCandidatesTested;
        randomCandidates += generation.randomCandidatesTested;
        refinementCandidates += generation.refinementCandidatesTested;
        guidedWinners += generation.guidedWinnersAccepted;
        randomWinners += generation.randomWinnersAccepted;
        refinementWinners += generation.refinementWinnersAccepted;
        history.push_back({
            .generation = completed,
            .previousFitness = generation.previousFitness,
            .fitness = state.fitness.fitness,
            .improvement = generation.previousFitness - state.fitness.fitness,
            .improved = improved,
            .candidatesTested = generation.candidatesTested,
            .winningMutations = std::move(generation.winningMutations),
            .guidedCandidatesTested = generation.guidedCandidatesTested,
            .randomCandidatesTested = generation.randomCandidatesTested,
            .refinementCandidatesTested = generation.refinementCandidatesTested,
            .guidedWinnersAccepted = generation.guidedWinnersAccepted,
            .randomWinnersAccepted = generation.randomWinnersAccepted,
            .refinementWinnersAccepted = generation.refinementWinnersAccepted,
        });
        notifyObserver(observer, history.back(), state.renderedAudio);
        stop = determineStopReason(state.fitness.fitness,
            completed,
            successfulWrites,
            stagnant,
            settings);
    }
    return {
        .finalAudio = std::move(state.renderedAudio),
        .initialFitness = initialFitness,
        .finalFitness = state.fitness.fitness,
        .completedGenerations = completed,
        .successfulWrites = successfulWrites,
        .stopReason = *stop,
        .history = std::move(history),
        .finalGenome = std::move(state.genome),
        .guidedCandidatesTested = guidedCandidates,
        .randomCandidatesTested = randomCandidates,
        .refinementCandidatesTested = refinementCandidates,
        .guidedWinnersAccepted = guidedWinners,
        .randomWinnersAccepted = randomWinners,
        .refinementWinnersAccepted = refinementWinners,
    };
}

EvolutionRunResult evolve(const AudioBuffer& source,
    double initialFitness,
    const FitnessReference& target,
    const EvolutionSettings& settings,
    const SineMutationSettings& mutationSettings,
    std::mt19937_64& randomGenerator,
    const GenerationObserver& observer)
{
    StftAnalyzer analyzer(target.fftSize, target.hopSize);
    return evolve(source,
        initialFitness,
        target,
        settings,
        mutationSettings,
        GenomeMutationSettings{},
        GuidedSearchSettings{},
        randomGenerator,
        analyzer,
        observer);
}
