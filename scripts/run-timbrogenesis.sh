#!/usr/bin/env bash

set -euo pipefail

program="./build/timbrogenesis"
prepared_root="prepared"
requested_sample_rate=""
length_policy="longest"
channel_mode="first"

usage()
{
    cat <<'EOF'
Usage:
    run-evolution.sh [options] SOURCE TARGET [-- PROGRAM_OPTIONS...]

Prepares two audio files for timbrogenesis and then runs it.

Input preparation:
    - Uses the first audio stream from each input.
    - Converts both files to mono WAV.
    - Uses the first channel by default.
    - Resamples both files to an identical sample rate.
    - Pads or trims them to an identical sample count.
    - Writes 32-bit floating-point WAV files.

Options:
    --program PATH
        Path to timbrogenesis.
        Default: ./build/timbrogenesis

    --sample-rate HZ
        Force both files to this sample rate.

        If omitted, the lower of the two input sample rates is used. This
        avoids unnecessary upsampling and reduces evolutionary processing
        cost.

    --length longest
        Pad the shorter recording with silence to match the longer recording.
        This is the default.

    --length shortest
        Trim the longer recording to match the shorter recording.

    --mixdown
        Mix all channels to mono instead of extracting the first channel.

    --prepared-root PATH
        Directory under which prepared inputs are retained.
        Default: prepared

    -h, --help
        Show this help.

Arguments after "--" are passed directly to timbrogenesis.

Examples:

    ./scripts/run-evolution.sh source.flac target.mp3

    ./scripts/run-evolution.sh \
        source.wav target.wav \
        -- \
        --maximum-generations 50 \
        --candidates-per-generation 2000

    ./scripts/run-evolution.sh \
        --sample-rate 48000 \
        --length shortest \
        source.flac target.mp3 \
        -- \
        --output output/test-run \
        --seed 12345
EOF
}

fail()
{
    echo "Error: $*" >&2
    exit 1
}

require_command()
{
    local command_name="$1"

    if ! command -v "$command_name" >/dev/null 2>&1; then
        fail "Required command not found: $command_name"
    fi
}

probe_sample_rate()
{
    local input_path="$1"

    ffprobe \
        -v error \
        -select_streams a:0 \
        -show_entries stream=sample_rate \
        -of default=noprint_wrappers=1:nokey=1 \
        "$input_path"
}

probe_sample_count()
{
    local input_path="$1"

    ffprobe \
        -v error \
        -select_streams a:0 \
        -show_entries stream=duration_ts \
        -of default=noprint_wrappers=1:nokey=1 \
        "$input_path"
}

probe_channel_count()
{
    local input_path="$1"

    ffprobe \
        -v error \
        -select_streams a:0 \
        -show_entries stream=channels \
        -of default=noprint_wrappers=1:nokey=1 \
        "$input_path"
}

validate_positive_integer()
{
    local value="$1"
    local description="$2"

    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        fail "$description must be a positive integer."
    fi
}

prepare_intermediate()
{
    local input_path="$1"
    local output_path="$2"
    local sample_rate_hz="$3"

    if [[ "$channel_mode" == "mixdown" ]]; then
        ffmpeg \
            -nostdin \
            -hide_banner \
            -loglevel error \
            -y \
            -i "$input_path" \
            -map 0:a:0 \
            -vn \
            -sn \
            -dn \
            -ac 1 \
            -ar "$sample_rate_hz" \
            -c:a pcm_f32le \
            "$output_path"
    else
        ffmpeg \
            -nostdin \
            -hide_banner \
            -loglevel error \
            -y \
            -i "$input_path" \
            -map 0:a:0 \
            -vn \
            -sn \
            -dn \
            -af "pan=mono|c0=c0,aresample=${sample_rate_hz}" \
            -c:a pcm_f32le \
            "$output_path"
    fi
}

force_sample_count()
{
    local input_path="$1"
    local output_path="$2"
    local sample_count="$3"

    ffmpeg \
        -nostdin \
        -hide_banner \
        -loglevel error \
        -y \
        -i "$input_path" \
        -map 0:a:0 \
        -vn \
        -sn \
        -dn \
        -af \
        "atrim=end_sample=${sample_count},"\
"apad=whole_len=${sample_count},"\
"atrim=end_sample=${sample_count}" \
        -c:a pcm_f32le \
        "$output_path"
}

verify_prepared_file()
{
    local input_path="$1"
    local expected_sample_rate_hz="$2"
    local expected_sample_count="$3"

    local actual_channels
    local actual_sample_rate_hz
    local actual_sample_count

    actual_channels="$(probe_channel_count "$input_path")"
    actual_sample_rate_hz="$(probe_sample_rate "$input_path")"
    actual_sample_count="$(probe_sample_count "$input_path")"

    if [[ "$actual_channels" != "1" ]]; then
        fail "$input_path is not mono after conversion."
    fi

    if [[ "$actual_sample_rate_hz" != "$expected_sample_rate_hz" ]]; then
        fail \
            "$input_path has sample rate ${actual_sample_rate_hz}, expected "\
"${expected_sample_rate_hz}."
    fi

    if [[ "$actual_sample_count" != "$expected_sample_count" ]]; then
        fail \
            "$input_path has ${actual_sample_count} samples, expected "\
"${expected_sample_count}."
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --program)
            [[ $# -ge 2 ]] || fail "--program requires a path."
            program="$2"
            shift 2
            ;;

        --sample-rate)
            [[ $# -ge 2 ]] || fail "--sample-rate requires a value."
            requested_sample_rate="$2"
            shift 2
            ;;

        --length)
            [[ $# -ge 2 ]] || fail "--length requires a value."

            case "$2" in
                longest|shortest)
                    length_policy="$2"
                    ;;
                *)
                    fail "--length must be 'longest' or 'shortest'."
                    ;;
            esac

            shift 2
            ;;

        --mixdown)
            channel_mode="mixdown"
            shift
            ;;

        --prepared-root)
            [[ $# -ge 2 ]] || fail "--prepared-root requires a path."
            prepared_root="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        --)
            fail "SOURCE and TARGET must appear before '--'."
            ;;

        -*)
            fail "Unknown wrapper option: $1"
            ;;

        *)
            break
            ;;
    esac
done

[[ $# -ge 2 ]] || {
    usage >&2
    exit 1
}

source_input="$1"
target_input="$2"
shift 2

if [[ $# -gt 0 && "$1" == "--" ]]; then
    shift
fi

program_arguments=("$@")

require_command ffmpeg
require_command ffprobe

[[ -f "$source_input" ]] || fail "Source file not found: $source_input"
[[ -f "$target_input" ]] || fail "Target file not found: $target_input"
[[ -x "$program" ]] || fail "Program is not executable: $program"

source_sample_rate_hz="$(probe_sample_rate "$source_input")"
target_sample_rate_hz="$(probe_sample_rate "$target_input")"

validate_positive_integer \
    "$source_sample_rate_hz" \
    "Source sample rate"

validate_positive_integer \
    "$target_sample_rate_hz" \
    "Target sample rate"

if [[ -n "$requested_sample_rate" ]]; then
    validate_positive_integer \
        "$requested_sample_rate" \
        "Requested sample rate"

    sample_rate_hz="$requested_sample_rate"
elif (( source_sample_rate_hz < target_sample_rate_hz )); then
    sample_rate_hz="$source_sample_rate_hz"
else
    sample_rate_hz="$target_sample_rate_hz"
fi

mkdir -p "$prepared_root"

timestamp="$(date '+%Y-%m-%d_%H%M%S')"

prepared_directory="$(
    mktemp -d \
        "${prepared_root%/}/${timestamp}-XXXXXX"
)"

source_intermediate="$prepared_directory/.source-intermediate.wav"
target_intermediate="$prepared_directory/.target-intermediate.wav"

prepared_source="$prepared_directory/source.wav"
prepared_target="$prepared_directory/target.wav"

cleanup()
{
    rm -f \
        "$source_intermediate" \
        "$target_intermediate"
}

trap cleanup EXIT

echo "Preparing audio..."
echo "  Source:      $source_input"
echo "  Target:      $target_input"
echo "  Sample rate: ${sample_rate_hz} Hz"
echo "  Channels:    $channel_mode"
echo "  Length:      $length_policy"

prepare_intermediate \
    "$source_input" \
    "$source_intermediate" \
    "$sample_rate_hz"

prepare_intermediate \
    "$target_input" \
    "$target_intermediate" \
    "$sample_rate_hz"

source_sample_count="$(probe_sample_count "$source_intermediate")"
target_sample_count="$(probe_sample_count "$target_intermediate")"

validate_positive_integer \
    "$source_sample_count" \
    "Converted source sample count"

validate_positive_integer \
    "$target_sample_count" \
    "Converted target sample count"

if [[ "$length_policy" == "shortest" ]]; then
    if (( source_sample_count < target_sample_count )); then
        final_sample_count="$source_sample_count"
    else
        final_sample_count="$target_sample_count"
    fi
else
    if (( source_sample_count > target_sample_count )); then
        final_sample_count="$source_sample_count"
    else
        final_sample_count="$target_sample_count"
    fi
fi

force_sample_count \
    "$source_intermediate" \
    "$prepared_source" \
    "$final_sample_count"

force_sample_count \
    "$target_intermediate" \
    "$prepared_target" \
    "$final_sample_count"

verify_prepared_file \
    "$prepared_source" \
    "$sample_rate_hz" \
    "$final_sample_count"

verify_prepared_file \
    "$prepared_target" \
    "$sample_rate_hz" \
    "$final_sample_count"

duration_seconds="$(
    awk \
        -v samples="$final_sample_count" \
        -v rate="$sample_rate_hz" \
        'BEGIN { printf "%.6f", samples / rate }'
)"

echo
echo "Prepared inputs:"
echo "  Directory:    $prepared_directory"
echo "  Sample rate:  ${sample_rate_hz} Hz"
echo "  Sample count: $final_sample_count"
echo "  Duration:     ${duration_seconds} s"
echo "  Source:       $prepared_source"
echo "  Target:       $prepared_target"
echo
echo "Starting evolution..."
echo

"$program" \
    "$prepared_source" \
    "$prepared_target" \
    --pre-conversion-source "$source_input" \
    --pre-conversion-target "$target_input" \
    "${program_arguments[@]}"
