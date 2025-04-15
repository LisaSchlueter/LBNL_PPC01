# Extract decay time of exponential tail of HPGe waveforms using peak files 
# Perform fit (truncated gaussian) to get average decay time, that is later used for pole-zero correction of all waveforms
# Note: Use peak files instead of all waveforms, because they are "good" waveforms and are quality cuts 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendDSP
using LegendDSP: get_fltpars
using LegendHDF5IO
using LegendSpecFits
using RadiationSpectra
using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using CairoMakie, Makie, LegendMakie
using Plots 
using Measures
using LinearAlgebra

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/processing_funcs/process_decaytime.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(50)
channel = ChannelId(1)
category = DataCategory(:cal)
det_ged = _channel2detector(asic, channel)

# load config: 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
min_τ, max_τ = 200.0u"µs", 400.0u"µs"#pz_config.min_tau, pz_config.max_tau
nbins        = pz_config.nbins
rel_cut_fit  = pz_config.rel_cut_fit
peak = Symbol(pz_config.peak)
bl_window = dsp_config.bl_window
tail_window = dsp_config.tail_window

# start debug
data = asic
data_peak  = read_ldata((peak), data, :jlpeaks, category, period, run, channel)
wvfs = data_peak.waveform
decay_times = dsp_decay_times(wvfs, bl_window, tail_window)
cuts_τ = cut_single_peak(decay_times, min_τ, max_τ,; n_bins=nbins, relative_cut=rel_cut_fit)
result, report = fit_single_trunc_gauss(decay_times, cuts_τ)
# plot 
filekey = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])[1]
fig = Figure()
LegendMakie.lplot!(report, figsize = (600, 430), titlesize = 17, title = get_plottitle(filekey, det_ged, "Decay Time Distribution"), juleana_logo = false, xlabel = "Decay time ($(unit(decay_times[1])))")


if maximum(abs.(extrema(report.gof.residuals_norm))) > 5.0
    ylim = ceil(Int, maximum(abs.(extrema(report.gof.residuals_norm))))
    ax2 = [ax for ax in fig.content if typeof(ax) <: Makie.Axis][2]
    Makie.ylims!(ax2, -ylim, ylim)
    fig
end


result_pz = (τ = result.μ, fit = result)
writelprops(data.par[category].rpars.pz[period], run, PropDict("$channel" => result_pz))