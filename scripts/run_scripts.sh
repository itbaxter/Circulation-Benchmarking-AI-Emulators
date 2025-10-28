#!/bin/bash

# Script to run specified scripts with a given data directory.
# Usage: ./run_scripts.sh DATA_DIR script1.py [script2.sh ...]

# The first argument is the data directory.
DATA_DIR="$1"

# Validate that DATA_DIR is provided and exists.
if [ -z "$DATA_DIR" ]; then
    echo "Error: Data directory must be specified as the first argument."
    echo "Usage: $0 DATA_DIR script1 [script2 ...]"
    exit 1
elif [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Convert to absolute path and export for scripts that might need it.
DATA_DIR=$(realpath "$DATA_DIR")
export DATA_DIR

# The rest of the arguments are the scripts to run.
shift # Remove DATA_DIR from argument list
SCRIPTS=("./QBO/plot_qbo_functions.py" "./WK99/ace2_analysis_like_ncl.py" "./WK99/era5_analysis_like_ncl.py" "./WK99/ngcm_analysis_like_ncl.py" "./WK99/plot_wk_diagram_analysis.py" "./RH91/eddy_co_spectra_v2.py" "./LH23/plot_lubis_main.py" "./LH23/plot_lubis_cross_correlations.py")

# Check if any scripts were provided.
if [ ${#SCRIPTS[@]} -eq 0 ]; then
    echo "Error: No scripts specified to run."
    echo "Usage: $0 DATA_DIR script1 [script2 ...]"
    exit 1
fi

echo "=========================================="
echo "Running specified scripts"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Scripts to run: ${SCRIPTS[@]}"
echo "=========================================="

# Counters for tracking
total_scripts=0
successful_scripts=0
failed_scripts=0

# Process each specified script
for script_path in "${SCRIPTS[@]}"; do
    # Check if the script is a WK99 analysis script
    if [[ "$script_path" == *"./WK99/"*analysis_like_ncl.py ]]; then
        # Extract the prefix (e.g., "era5") from the script name
        script_name=$(basename "$script_path")
        prefix="${script_name%_analysis_like_ncl.py}"
        
        echo $prefix
        # Check if a corresponding .nc data file exists in the WK99 data directory.
        # nullglob ensures the array is empty if no files match.
        shopt -s nullglob
        data_files=(./WK99/data/${prefix}_*.nc)
        shopt -u nullglob
        
        if [ ${#data_files[@]} -gt 0 ]; then
            echo ""
            echo "----------------------------------------"
            echo "Skipping analysis (data exists for $prefix): $script_path"
            echo "----------------------------------------"
            continue
        fi
    fi

    total_scripts=$((total_scripts + 1))

    # Validate that the script file exists
    if [ ! -f "$script_path" ]; then
        echo ""
        echo "----------------------------------------"
        echo "Error: Script '$script_path' not found, skipping."
        echo "----------------------------------------"
        failed_scripts=$((failed_scripts + 1))
        continue
    fi

    echo ""
    echo "----------------------------------------"
    echo "Running: $script_path"
    echo "----------------------------------------"

    # Change to the script's directory to handle relative paths correctly
    script_dir=$(dirname "$script_path")
    script_name=$(basename "$script_path")
    pushd "$script_dir" > /dev/null

    # Determine how to run the script based on its extension
    if [[ "$script_name" == *.py ]]; then
        # Run Python script, passing DATA_DIR as an argument
        if python3 "$script_name" --data_dir "$DATA_DIR"; then
            echo "Success: $script_path"
            successful_scripts=$((successful_scripts + 1))
        else
            echo "Failed: $script_path"
            failed_scripts=$((failed_scripts + 1))
        fi
    elif [[ "$script_name" == *.sh ]]; then
        # Run shell script, passing DATA_DIR as an argument
        if bash "$script_name" "$DATA_DIR"; then
            echo "Success: $script_path"
            successful_scripts=$((successful_scripts + 1))
        else
            echo "Failed: $script_path"
            failed_scripts=$((failed_scripts + 1))
        fi
    else
        echo "Error: Unknown script type for '$script_path'. Can only run .py or .sh files."
        failed_scripts=$((failed_scripts + 1))
    fi

    popd > /dev/null
done

# Print summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total scripts attempted: $total_scripts"
echo "Successful: $successful_scripts"
echo "Failed: $failed_scripts"
echo "=========================================="

# Exit with error if any scripts failed
if [ $failed_scripts -gt 0 ]; then
    exit 1
fi
