#!/bin/bash

# Script to run all Python scripts in subdirectories with specified data directory
# Usage: ./run_scripts.sh [DATA_DIR] [SCRIPT_DIR]

# Set default directories
DATA_DIR="${1:-$(pwd)}"
SCRIPT_DIR="${2:-./}"

# Validate that directories exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Script directory '$SCRIPT_DIR' does not exist"
    exit 1
fi

# Convert to absolute paths
DATA_DIR=$(realpath "$DATA_DIR")
SCRIPT_DIR=$(realpath "$SCRIPT_DIR")

echo "=========================================="
echo "Running plotting scripts"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "=========================================="

# Export DATA_DIR as environment variable for scripts to use
export DATA_DIR

# Counter for tracking
total_scripts=0
successful_scripts=0
failed_scripts=0

# Define the order of subdirectories to process
SUBDIRS=("QBO" "WK99" "RH91" "LH23") 

# Process each subdirectory in order
for subdir in "${SUBDIRS[@]}"; do
    subdir_path="$SCRIPT_DIR/$subdir"
    
    # Check if subdirectory exists
    if [ ! -d "$subdir_path" ]; then
        echo ""
        echo "⚠ Warning: Subdirectory '$subdir' not found, skipping..."
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "PROCESSING DIRECTORY: $subdir"
    echo "=========================================="
    
    # Find and run all Python scripts in this subdirectory
    while IFS= read -r -d '' script; do
        total_scripts=$((total_scripts + 1))
        
        # Get relative path for display
        rel_path="${script#$SCRIPT_DIR/}"
        
        echo ""
        echo "----------------------------------------"
        echo "Running: $rel_path"
        echo "----------------------------------------"
        
        # Change to the script's directory
        script_dir=$(dirname "$script")
        pushd "$script_dir" > /dev/null
        
        # Run the script with DATA_DIR as argument if it accepts arguments
        # Otherwise, rely on the exported environment variable
        if python3 "$(basename "$script")" "$DATA_DIR" 2>&1; then
            echo "✓ Success: $rel_path"
            successful_scripts=$((successful_scripts + 1))
        else
            echo "✗ Failed: $rel_path"
            failed_scripts=$((failed_scripts + 1))
        fi
        
        popd > /dev/null
        
    done < <(find "$subdir_path" -maxdepth 1 -type f -name "*.py" -print0 | sort -z)
    
    # Also run shell scripts if any exist in this subdirectory
    while IFS= read -r -d '' script; do
        total_scripts=$((total_scripts + 1))
        
        rel_path="${script#$SCRIPT_DIR/}"
        
        echo ""
        echo "----------------------------------------"
        echo "Running: $rel_path"
        echo "----------------------------------------"
        
        script_dir=$(dirname "$script")
        pushd "$script_dir" > /dev/null
        
        if bash "$(basename "$script")" "$DATA_DIR" 2>&1; then
            echo "✓ Success: $rel_path"
            successful_scripts=$((successful_scripts + 1))
        else
            echo "✗ Failed: $rel_path"
            failed_scripts=$((failed_scripts + 1))
        fi
        
        popd > /dev/null
        
    done < <(find "$subdir_path" -maxdepth 1 -type f -name "*.sh" -print0 | sort -z)
    
    echo ""
    echo "✓ Completed directory: $subdir"
done

# Print summary
echo ""
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total scripts: $total_scripts"
echo "Successful: $successful_scripts"
echo "Failed: $failed_scripts"
echo "=========================================="

# Exit with error if any scripts failed
if [ $failed_scripts -gt 0 ]; then
    exit 1
fi
