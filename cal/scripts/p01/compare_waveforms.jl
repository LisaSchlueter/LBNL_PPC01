using LegendDataManagement
using LegendDataManagement.LDMUtils
using CairoMakie, LegendPlots
using LegendHDF5IO
using Unitful
using TypedTables
using StatsBase
using ColorSchemes
using Measures
using LegendDSP
using RadiationDetectorDSP
using IntervalSets
using Measurements: value as mvalue
asic = LegendData(:ppc01)
period = DataPeriod(1)
category = :cal 
channel = ChannelId(1)
runs = [6, 7, 8, 9, 10]
bias_kV = collect(1.3:0.3:2.5)

# load waveforms from peak files (they are already qc cut )
filekeys = [search_disk(FileKey, asic.tier[DataTier(:raw), category , period, DataRun(r)]) for r in runs]
waveforms = [read_ldata((:Co60a), asic, DataTier(:jlpeaks), filekeys[i], channel).waveform for i in eachindex(runs)]

# do waveform shift by baseline 
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1][1]).default)
bl_stats = [signalstats.(waveforms[i], leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window)) for i in eachindex(runs)]
wvfs_shift = [shift_waveform.(waveforms[i], -bl_stats[i].mean) for i in eachindex(runs)]
wvfs_max = [maximum.(wvfs_shift[i].signal) for i in eachindex(runs)]
wvfs_norm = [multiply_waveform.(wvfs_shift[i], wvfs_max[i]) for i in eachindex(runs)]

palette = ColorSchemes.twelvebitrainbow
colors = [get(palette, i/5) for i in 1:5]

i = 1
wvfs_mean = [vec(mean(hcat(wvfs_norm[i].signal...), dims = 2)) for i in eachindex(runs)]

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Time (µs)", ylabel = "Signal (V)", limits = ((nothing, nothing), (nothing, nothing)),
                title = "LBNL-PPC01 and ASIC L1k65n with buffer:  $period, r00$(runs[1])-r0$(runs[end])", titlesize = 16)
sample_idx = [rand(1:length(wvfs_norm[i])) for i in eachindex(runs)]
for i in eachindex(runs)
    # lines!(ax, ustrip.(wvfs_norm[i][1].time), wvfs_mean[i], color = colors[i], linewidth = 1.0, label = "$(bias_kV[i]) kV")  
    lines!(ax, ustrip.(wvfs_norm[i][sample_idx[i]].time), wvfs_norm[i][sample_idx[i]].signal, color = colors[i], linewidth = 1.0, label = "$(bias_kV[i]) kV")  
end
axislegend()
xlims!(19, 20.2)
fig           
# plt_folder = "$(@__DIR__)/plots/"
# if !isdir(plt_folder)
#     mkdir(plt_folder)
# end
# plt_name = plt_folder * "biasvoltage_waveforms.png"
# save(plt_name, fig)