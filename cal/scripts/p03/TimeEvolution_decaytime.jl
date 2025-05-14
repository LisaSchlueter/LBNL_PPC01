
using LegendDataManagement
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using LegendDSP
using LegendPlots
using CairoMakie
using Unitful, Dates
using Measurements: value as mvalue, uncertainty as muncert 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__)
include("$relPath/hpge-ana/utils/utils_aux.jl")
asic = LegendData(:ppc01)
period = DataPeriod(3)
runs = vcat(1:14, 17:26, 29:34)
channel = ChannelId(1)
category = DataCategory(:cal)

# load decay times and timestamps 
decay_times = [read_ldata(asic, :jldsp, category, period, DataRun(run), channel).tail_τ for run in runs]
timestamps = [read_ldata(asic, :jldsp, category, period, DataRun(run), channel).timestamp for run in runs]

# some basic cleaning (no proper QC here)
τ_min = 100u"µs" 
τ_max = 500u"µs"
qc = [findall(τ_min .< decay_times[i] .< τ_max) for i in eachindex(runs)]

# plot time evolution for one run 
i = 1
for i in eachindex(runs)
    fig = Figure()
    filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, DataRun(runs[i])])[1]
    ax = Axis(fig[1, 1], title = get_plottitle(filekey, _channel2detector(asic, channel), "Decay time evolution") * "\n measurement day: $(Dates.format(unix2datetime(timestamps[i][1]), "mm-dd-yyyy"))", 
    xlabel = "Measurement time (s)", ylabel = "Decay time (µs)")
    scatter!(ax, timestamps[i][qc[i]] .-  timestamps[i][1], ustrip.(decay_times[i])[qc[i]], markersize = 6, label = "run $(runs[i])")
    fig
    plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :decay_time)
    plt_name = plt_folder * "/TimeEvolution_decaytime.png"
    save(plt_name, fig)
end 

# plot median decay time (all waveforms)
τ_pz = [asic.par.rpars.pz[period, DataRun(run), channel].τ for run in runs]

# filekey_start = [search_disk(FileKey, asic.tier[DataTier(:raw), category , period, DataRun(run)])[1] for run in runs]
timestamps_day = [Dates.format(unix2datetime(timestamps[i][1]), "mm-dd") for i in eachindex(runs)]
timestamps_unix = [timestamps[i][1].- timestamps[1][1] for i in eachindex(runs)]
unique_idx = [findfirst(x-> x == value, timestamps_day) for value in unique(timestamps_day)]
# # plot 
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Measurement day in 2025", ylabel = "Mean decay time (µs)", 
        title = "$(_channel2detector(asic, channel)), $period, $(DataRun(runs[1])) to $(DataRun(runs[end]))",
        xticks = (timestamps_unix[unique_idx], timestamps_day[unique_idx]))
errorbars!(timestamps_unix, mvalue.(ustrip.(τ_pz)), muncert.(ustrip.(τ_pz)))
fig 
plt_folder = "$(@__DIR__)/plots/"
plt_name = plt_folder * "/TimeEvolution_MeanDecayTime_allruns.png"
save(plt_name, fig)





