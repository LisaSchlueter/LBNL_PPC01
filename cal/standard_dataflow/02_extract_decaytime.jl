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
run = DataRun(51)
channel = ChannelId(1)
category = DataCategory(:cal)

# load config: 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
min_τ, max_τ = pz_config.min_tau, pz_config.max_tau
nbins        = pz_config.nbins
rel_cut_fit  = pz_config.rel_cut_fit
peak = Symbol(pz_config.peak)
bl_window = dsp_config.bl_window
tail_window = dsp_config.tail_window

# run processor
plt = process_decaytime(asic, period, run, category, channel, min_τ, max_τ, nbins, rel_cut_fit, peak, bl_window, tail_window; reprocess = true)
display(plt)

# sanity check: read decay time pars
pars_pz = asic.par[category].rpars.pz[period, run, channel]

