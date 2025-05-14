using LegendDataManagement
using CSV, JSON
using Unitful
using TypedTables
using DataFrames
using Dates
using RadiationDetectorSignals
using RadiationDetectorDSP
using Makie, LegendMakie, CairoMakie
using IntervalSets

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(2)
channel = ChannelId(1)
category = :cal 
timestep = 10.0u"ns"
csv_folder = asic.tier[DataTier(:raw_csv), category , period, run]
files_csv = readdir(csv_folder, join = true)
filter!(x -> occursin(".ecsv", x), files_csv)

# read first file
function read_file(filename::String)
    # read HEADER of first file to get the absolute time of the measurement
    header = CSV.read(filename, DataFrame; delim=':', header = false, limit = 1, silencewarnings = true);
    timestamp_abs_unix  = datetime2unix(DateTime(split(header[1,2], "Time: ")[2], dateformat"yyyy-mm-dd HH:MM:SS"))

    # load first file for relative timestamp start 
    timestamp_evt_start = Int64(CSV.read(filename, DataFrame; delim='\t', comment = "#", header = false)[!,1][1])
    
    f = CSV.read(filename, DataFrame; delim='\t',  comment = "#", header = false) 

    nchannel = size(f,2) - 2
    column_names = ["timestamp", "channellist", "ch" .* string.(1:nchannel)...]
    rename!(f, Symbol.(column_names))

    # Timestamp: Absolute time of the measurement in unix time. (seconds since 1970-01-01 00:00:00 UTC)
    # The timestamp per waveform is given in units of clock cycles since the last reset of the FPGA. 
    timestamp_rel = ustrip(uconvert(u"s",timestep)).* (Vector(Int64.(f.timestamp)) .- timestamp_evt_start)
    timestamp_unix = timestamp_abs_unix .+ round.(Int64,timestamp_rel)

    # Read channels 
    channel1 = map(x -> Int32.(x), JSON.parse.(f.ch1))
    channel2 = map(x -> Int32.(x), JSON.parse.(f.ch2))
    nsamples = length(channel1[1])
    times = 0.0u"µs":timestep:((nsamples - 1)*timestep)


    # convert to waveforms 
    wvfs_ch1 = ArrayOfRDWaveforms([RDWaveform(t, signal) for (t, signal) in zip(fill(times, length(channel1)), channel1)])
    wvfs_ch2 = ArrayOfRDWaveforms([RDWaveform(t, signal) for (t, signal) in zip(fill(times, length(channel2)), channel2)])

    return wvfs_ch1, wvfs_ch2
end 
wvfs1, wvfs2 = read_file(files_csv[1])

# shift baseline to zero 
bl_window = 0.0u"µs" .. 7.0u"µs"
bl_stats = signalstats.(wvfs1, leftendpoint(bl_window), rightendpoint(bl_window))
wvfs1 = shift_waveform.(wvfs1, -bl_stats.mean)# substract baseline from waveforms
bl_stats2 = signalstats.(wvfs2, leftendpoint(bl_window), rightendpoint(bl_window))
wvfs2 = shift_waveform.(wvfs2, -bl_stats2.mean)# substract baseline from waveforms



# plot 
wvf_norm = maximum(wvfs1[1].signal)
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (µs)", ylabel = "Signal (a.u.)")
lines!(ax, 1e6.*ustrip.(wvfs2[1].time), (wvfs1[1].signal .- wvfs2[1].signal) ./wvf_norm, label = "ASIC diff.", color = :black)
lines!(ax, 1e6.*ustrip.(wvfs1[1].time), wvfs1[1].signal./wvf_norm, label = "ASIC out-pos",  color = :red2, linestyle = :dash)
lines!(ax, 1e6.*ustrip.(wvfs2[1].time), wvfs2[1].signal./wvf_norm, label = "ASIC out-neg.", color = :dodgerblue, linestyle = :dash)
axislegend(ax, merge = true, position = :lt, orientation = :horizontal)
Makie.ylims!(ax, -1.1, 2.5)
fig 

pltdir = "$(@__DIR__)/plots/"
save(pltdir * "ASIC_diff_output.png", fig)