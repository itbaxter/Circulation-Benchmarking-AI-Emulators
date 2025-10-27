begin

; Define input directory and pattern
input_dir = "../../../Circulation-Benchmarking-AI-Emulators-data/ace2/sh_uwind/"
file_pattern = "*.nc"  ; Pattern to match files

; Get list of files in directory
files = systemfunc("ls " + input_dir + file_pattern)
nfiles = dimsizes(files)

print("Found " + nfiles + " files to process")
print("")

; Loop through all files
do ifile = 0, nfiles-1

    print("Processing file " + (ifile+1) + " of " + nfiles + "...")
    
    ; Get full file path
    input_file = files(ifile)
    print("Input file: " + input_file)
    
    ; Extract basename from filename (remove path and .nc extension)
    filename_only = systemfunc("basename " + input_file + " .nc")
    
    ; Create source identifier from filename
    source = filename_only
    
    print("Source ID: " + source)
    print("Reading the input file...")

    ; Open input file
    f1 = addfile(input_file, "r")

    ; Get all variable names
    var_names = getfilevarnames(f1)
    do iv = 0, dimsizes(var_names)-1
        wind_var = var_names(iv)
    end do
    printVarSummary(var_names)

    ; Read wind data - adjust variable name as needed
    u = f1->$wind_var$(time|:, {lat|-80:-20})
    printVarSummary(u)

    u@_FillValue = -999

    print("Weighting latitude...")

    rad = 4.0 * atan(1.0) / 180.0
    wgt_u = tofloat(sqrt(cos(u&lat * rad)))  ; Convert to float to match u

    printVarSummary(wgt_u)
    u = (/u * conform(u, wgt_u, 1)/)

    print("Calculating EOFs...")

    neof = 2

    eof = eofunc_n_Wrap(u, neof, False, 0)
    pc = eofunc_ts_n_Wrap(u, eof, False, 0)

    eof(0, :) = (/-1.0 * eof(0, :)/)
    pc(0, :) = (/-1.0 * pc(0, :)/)

    print("Projecting...")

    printVarSummary(pc)
    z1 = pc(:, :)

    print("Saving z1 to NetCDF...")

    ; Create output filename using original basename
    output_file = "z1/z1_" + source + ".nc"

    ; Remove existing file if it exists
    system("rm -f " + output_file)

    ; Create NetCDF file
    fout = addfile(output_file, "c")

    ; Set global attributes
    fout@title = "Z1 time series from EOF analysis"
    fout@description = "First principal component (z1) from EOF analysis of zonal wind anomalies"
    fout@source = source
    fout@original_file = input_file
    fout@created = systemfunc("date")
    fout@latitude_range = "80S-20S"
    fout@pressure_levels = "Column averaged"

    ; Create a clean z1 array without any coordinates
    z1_values = new(dimsizes(z1), "float")
    z1_values = (/z1/)
    z1_values!0 = "mode"
    z1_values!1 = "time"

    ; Create simple time coordinate
    time_coord = ispan(0, dimsizes(z1(0,:))-1, 1)
    time_coord!0 = "time"
    time_coord@long_name = "time index"
    time_coord@units = "days since start"

    ; Write variables to file
    fout->time = time_coord
    fout->z1 = z1_values

    print("z1 saved to: " + output_file)

    ; Print some basic statistics
    print("z1 statistics:")
    print("  Min: " + min(z1))
    print("  Max: " + max(z1))
    print("  Mean: " + avg(z1))
    print("  Std: " + dim_stddev(z1))
    print("  Length: " + dimsizes(z1))

    ; Clean up variables for next iteration
    delete(f1)
    delete(u)
    delete(wgt_u)
    delete(eof)
    delete(pc)
    delete(z1)
    delete(z1_values)
    delete(time_coord)
    delete(fout)

    print("Completed processing: " + source)
    print("")

end do  ; End of file loop

print("All " + nfiles + " files processed successfully!")
print("Output files saved in z1/ directory with original basenames")

end
