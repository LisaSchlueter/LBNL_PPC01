# convert .ecsv files from SkuTek "FemtoDAQ Vireo" to LEGEND-style hdf5-files. 
using LegendDataManagement
using TypedTables
using Plots 
using CSV, JSON
using Unitful
using Dates

# include relevant functions
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_IO.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(2)
channel = ChannelId(1)
category = :cal 

# input for function: csv folder, heading 
csv_folder = asic.tier[DataTier(:raw_csv), category , period, run]
timestep = 0.01u"µs"

# skutek_csv_to_lh5(asic, period, run, category, channel, csv_folder; timestep = timestep)

## DEBUG 
# # start funcion 
# # # METADATA: read json file with DAQ settings
# # files_json = filter(x -> occursin(".json", x), files)
# # if isempty(files_json)
# #    @info "No json file with DAQ settings found"
# #    DAQ_settings = Dict()
# # else
# #     DAQ_settings = JSON.parsefile(files_json[1])
# #     @info "Read DAQ settings from json file"
# # end
# # time_absolut_iso = DateTime(split(DAQ_settings["Datetime"], "Time: ")[2], dateformat"yyyy-mm-dd HH:MM:SS")
# # time_absolut_unix = datetime2unix(time_absolut_iso)

# ## FILE BY FILE 
# files_csv = readdir(csv_folder, join = true)
# filter!(x -> occursin(".ecsv", x), files_csv)

# # READ HEADER of first file to get the absolute time of the measurement
# header = CSV.read(files_csv[1], DataFrame; delim=':', header = false, limit = header_rows - 5, silencewarnings = true);
# timestamp_abs_str = header[findfirst(contains.(header[!,1], "Datetime")),2]
# timestamp_abs_iso  = DateTime(split(timestamp_abs_str, "Time: ")[2], dateformat"yyyy-mm-dd HH:MM:SS")
# timestamp_abs_unix = datetime2unix(timestamp_abs_iso)

# # load first file for relative timestamp start 
# timestamp_evt_start = Int64(CSV.read(files_csv[1], DataFrame; delim='\t', comment = "#", header = false)[!,1][1])
# eventnumber_max = 0

# for i in eachindex(files_csv)
#     global eventnumber_max
#     # read data and rename columns for better readability 
#     data = CSV.read(files_csv[i], DataFrame; delim='\t',  comment = "#", header = false) 
#     nchannel = size(data,2) - 2
#     column_names = ["timestamp", "channellist", "ch" .* string.(1:nchannel)...]
#     rename!(data, Symbol.(column_names))
    
#     # Timestamp: Absolute time of the measurement in unix time. (seconds since 1970-01-01 00:00:00 UTC)
#     # The timestamp per waveform is given in units of clock cycles since the last reset of the FPGA. 
#     timestamp_rel = ustrip(uconvert(u"s",timestep)).* (Vector(Int64.(data.timestamp)) .- timestamp_evt_start)
#     timestamp_unix = timestamp_abs_unix .+ round.(Int64,timestamp_rel)

#     # Read channels 
#     # channel_list = map(x-> Int8.(x), JSON.parse.(data[!,2]))
#     channel1 = map(x -> Int32.(x), JSON.parse.(data.ch1))
#     channel2 = map(x -> Int32.(x), JSON.parse.(data.ch2))
#     ch_diff = channel1 .- channel2

#     # convert to waveforms 
#     nsamples = length(channel1[1])
#     times = 0.0u"µs":timestep:((nsamples - 1)*timestep)
#     wvfs = ArrayOfRDWaveforms([RDWaveform(t, signal) for (t, signal) in zip(fill(times, length(channel1)), channel1)])

#     # save to lh5 files 

#     h5folder = asic.tier[DataTier(:raw), category, period, run] * "/"
#     if !ispath(h5folder)
#         mkpath(h5folder)
#         @info "created folder: $h5folder"
#     end

#     filekey = string(FileKey(asic.name, period, run, category, Timestamp(timestamp_unix[1])))
#     h5name = h5folder * filekey * "-tier_raw.lh5"
#     eventnumber = eventnumber_max .+ collect(1:length(wvfs))

#     fid = lh5open(h5name, "w")
#     fid["$channel/raw/waveform"]  = wvfs
#     fid["$channel/raw/daqenergy"] = maximum.(extrema.(wvfs.signal)) .- minimum.(extrema.(wvfs.signal)) #DAQ energy not available in oscilloscope, approx with difference between max and min, needed for compatibility with LEGEND functions
#     fid["$channel/raw/eventnumber"]  = eventnumber
#     fid["$channel/raw/timestamp"]  = timestamp_unix 
#     fid["$channel/raw/baseline"] = fill(NaN, length(wvfs)) # not available in csv files, but needed for compatibility with LEGEND functions
#     @info "saved $(length(wvfs)) waveforms in .lh5 files with filekey: $filekey , folder: $h5folder"
#     close(fid)

#     eventnumber_max = maximum(eventnumber)
# end 