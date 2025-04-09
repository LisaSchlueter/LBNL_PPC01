## analysis of PPC01 data 
using LegendDataManagement
using HDF5, LegendHDF5IO
using CairoMakie, LegendMakie, Makie
using Unitful
using Random, StatsBase
using IntervalSets
using LegendDSP, RadiationDetectorDSP

# inputs and setup 
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1) # germanium channel 
category = DataCategory(:bch)
asic = LegendData(:ppc01)
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekey).default)

function rms(x::Vector{T}) where {T <: Real} 
    sqrt(mean(x.^2))
end

gain = if string(period) == "p01"
    11.9 * 2
elseif string(period) == "p02" && string(run) in ["r001" , "r002"]
    48.0
elseif string(period) == "p02" && string(run) in ["r003" , "r004"]
    100.0
end 

# read waveforms , get 10 random waveforms from each run 
data = read_ldata(asic, DataTier(:raw), filekey, channel)
Random.seed!(1234)
random_indices = rand(1:length(data.waveform), 10)
wvfs = data.waveform[random_indices]
bl_stats = signalstats.(wvfs, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs = shift_waveform.(wvfs, -bl_stats.mean)

result =  (wvfs = wvfs, )

# plot
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :waveform)
for wvfs_idx in 1:5
    y = result.wvfs[wvfs_idx].signal ./ gain .* 1e6 
    x = ustrip.(result.wvfs[wvfs_idx].time)
    fig = Figure()
    ax = Axis(fig[1,1], title = "$category-$period-$run, rms = $(round(rms(y), digits = 1)) µV ", xlabel = "Time (µs)", ylabel = "Signal / gain (µV)")
    lines!(ax, x, y)
    pname = plt_folder * "/waveform_$(wvfs_idx).png"
    save(pname, fig)
    @info "Saved waveform plot to $pname"
    fig 
end