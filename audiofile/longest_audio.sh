#!/bin/bash

# Check if a directory is provided as an argument
if [ -z "$1" ]; then
    echo "Usage: $0 /path/to/directory"
    exit 1
fi

# Set the directory variable to the user-provided directory
DIRECTORY=$1

# Initialize a counter for audio files longer than 10 seconds
long_audio_count=0

# Verify that the directory contains audio files (supports multiple extensions)
shopt -s nullglob
audio_files=("$DIRECTORY"/*.{wav,mp3,ogg,aac,flac})
shopt -u nullglob

# Check if there are any audio files in the directory
if [ ${#audio_files[@]} -eq 0 ]; then
    echo "No audio files found in the specified directory."
    exit 1
fi

# Iterate through all supported audio files in the directory
for file in "${audio_files[@]}"; do
    # Check if the file exists (this should always be true due to array check)
    if [ -e "$file" ]; then
        # Get the duration of the audio file using ffprobe
        duration=$(ffprobe -i "$file" -show_entries format=duration -v quiet -of csv="p=0" 2>/dev/null)

        # Handle cases where ffprobe may fail (e.g., corrupted files)
        if [ -z "$duration" ]; then
            echo "Failed to get duration for: $file"
            continue
        fi

        # Convert duration to an integer for comparison (rounding)
        duration=${duration%.*}

        # Check if the duration is greater than 10 seconds
        if [ "$duration" -gt 20 ]; then
            long_audio_count=$((long_audio_count + 1))
        fi
    fi
done

# Print the number of audio files longer than 10 seconds
echo "Number of audio files longer than 10 seconds: $long_audio_count"

