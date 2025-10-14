begin

print("Reading the input file...")

source = "ERA5"
f1 = addfile("/project/tas1/itbaxter/aimip/benchmarking_ai_variability/data/era5/sh_uwind/u_anomalies_era5.1979-2020_nharm4.nc", "r")

u = f1->u(time|:, {lat|-80:-20})
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

; Create output filename with timestamp
output_file = "z1/z1_" + source + "_" + systemfunc("date +%Y%m%d") + ".nc"

; Remove existing file if it exists
system("rm -f " + output_file)

; Create NetCDF file
fout = addfile(output_file, "c")

; Set global attributes
fout@title = "Z1 time series from EOF analysis"
fout@description = "First principal component (z1) from EOF analysis of CESM2-WACCM zonal wind anomalies"
fout@source = source
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

end
