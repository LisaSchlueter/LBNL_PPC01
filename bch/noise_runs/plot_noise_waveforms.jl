## analysis of PPC01 data 
using LegendDataManagement
using HDF5, LegendHDF5IO
using CairoMakie, LegendMakie, Makie
using Unitful
using Random, StatsBase
using IntervalSets
using LegendDSP, RadiationDetectorDSP

# inputs and setup 
period = DataPeriod(1)
run = DataRun(37)
channel = ChannelId(1) # germanium channel 
category = DataCategory(:bch)
asic = LegendData(:ppc01)
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekey).default)

function rms(x::Vector{T}) where {T <: Real} 
    sqrt(mean(x.^2))
end

# read waveforms , get 10 random waveforms from each run 
data = read_ldata(asic, DataTier(:raw), filekey, channel)
Random.seed!(1234)
random_indices = rand(1:length(data.waveform), 10)
wvfs = data.waveform[random_indices]
bl_stats = signalstats.(wvfs, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs = shift_waveform.(wvfs, -bl_stats.mean)

result = if haskey(data, :waveform_ch1)
    wvf_ch1 = data.waveform_ch1[random_indices]
    wvf_ch2 = data.waveform_ch2[random_indices]

    bl_stats = signalstats.(wvf_ch1, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
    wvf_ch1 = shift_waveform.(wvf_ch1, -bl_stats.mean)

    bl_stats = signalstats.(wvf_ch2, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
    wvf_ch2 = shift_waveform.(wvf_ch2, -bl_stats.mean)

    (wvfs = wvfs, wvfs_ch1 = wvf_ch1, wvfs_ch2 = wvf_ch2)
else
    (wvfs = wvfs, )
end

# plot
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :waveform)
wvfs_idx = 3
begin 
    fig = Figure()
    ax = Axis(fig[1,1], title = "$category-$period-$run, rms = $(round(rms(result.wvfs[wvfs_idx].signal), digits = 1)) ADC", xlabel = "Time (µs)", ylabel = "Signal (ADC)")
    if haskey(result, :wvfs_ch1)
        lines!(ax, ustrip.(result.wvfs[wvfs_idx].time), result.wvfs[wvfs_idx].signal , label = "diff.")
        lines!(ax, ustrip.(result.wvfs_ch1[wvfs_idx].time), result.wvfs_ch1[wvfs_idx].signal , linewidth = 3, label = "ch1")
        lines!(ax, ustrip.(result.wvfs_ch2[wvfs_idx].time), result.wvfs_ch2[wvfs_idx].signal ,  linewidth = 3, label = "ch2")
        axislegend(orientation = :horizontal, location = :lt)
    else
        lines!(ax, ustrip.(result.wvfs[wvfs_idx].time), result.wvfs[wvfs_idx].signal)
    end
    pname = plt_folder * "/waveform_$(wvfs_idx).png"
    save(pname, fig)
    @info "Saved waveform plot to $pname"
    fig
end 

