# Extract decay time of exponential tail of HPGe waveforms using peak files 
# Perform fit (truncated gaussian) to get average decay time, that is later used for pole-zero correction of all waveforms
# Note: Use peak files instead of all waveforms, because they are "good" waveforms 
env_path = "$(@__DIR__)/" * relpath(split(@__DIR__, "LBNL_PPC01")[1], @__DIR__)  * "/LBNL_PPC01/"
import Pkg; Pkg.activate(env_path)
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
using Plots
using Measures
using Distributions
# set data configuration (where to find data; and where to save results)
if gethostname() == "Lisas-MacBook-Pro.local"
    ENV["LEGEND_DATA_CONFIG"] = "/Users/lisa/Documents/Workspace/LEGEND/LBL_ASIC/ASIC_data/ppc01/config.json"
 else # on NERSC 
     ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
 end 

# load functions from hpge-ana
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_decaytime.jl")
# inputs
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = DataCategory(:cal)

# load config: 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
min_τ, max_τ = pz_config.min_tau, pz_config.max_tau
nbins        = 100#pz_config.nbins
rel_cut_fit  = pz_config.rel_cut_fit
peak = Symbol(pz_config.peak)
bl_window = dsp_config.bl_window
tail_window = dsp_config.tail_window

# run processor
plt = process_decaytime(asic, period, run, category, channel, min_τ, max_τ, nbins, rel_cut_fit, peak, bl_window, tail_window; reprocess = true)
display(plt)


# # DEBUG 
# min_τ = pz_config.min_tau
# max_τ = pz_config.max_tau
# nbins =  pz_config.nbins
# rel_cut_fit =  pz_config.rel_cut_fit
# peak = Symbol(pz_config.peak)
# bl_window = dsp_config.bl_window
# tail_window = dsp_config.tail_window

# # check if decaytime pars already exist
# pz_file = joinpath(mkpath(data_path(asic.par.rpars.pz[period])), "$(string(run)).json")
# if isfile(pz_file) && !reprocess
#     @info "Decay time pars (pz) already exist for $category period $period - run $run - channel $channel - you're done!"
#     return
# end

# det = _channel2detector(asic, channel)
# @debug "Create pars db"
# mkpath(joinpath(data_path(asic.par.rpars.pz), string(period)))

# # load all waveforms 
# data = read_ldata(asic, DataTier(:raw), filekeys, channel)
# wvfs = data.waveform
# decay_times = dsp_decay_times(wvfs, bl_window, tail_window)
# filter!(x -> 0u"µs" < x <  2 * max_τ , decay_times)
# # fit_result = fit(Normal, ustrip.(decay_times))

# cuts_τ = cut_single_peak(decay_times, min_τ, max_τ,; n_bins=nbins, relative_cut=0.05)
# result, report = fit_single_trunc_gauss(decay_times, cuts_τ)
 
# # plot 
# p = plot(report, size = (600, 500), legend = :topright, xlabel = "Decay time (µs)", fillcolor = :deepskyblue2, color = :darkorange, dpi = 200)
# plot!(p, xguidefontsize = 16, yguidefontsize = 16, xtickfontsize = 12, ytickfontsize = 12, legendfontsize = 10,
#              legendforegroundcolor = :silver)
# plot!(p, xlabel = " ", xtickfontsize = 1, bottom_margin = -6mm, ylims = (0, ylims()[2]+0.25*ylims()[2]), subplot = 1)
# filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
# title!(p, get_plottitle(filekey, det, "Decay Time Distribution"),  subplot=1, titlefontsize = 12)
# savelfig(savefig, p, asic, filekey, channel, :decay_time)
# @info "Save sanity plot to $(LegendDataManagement.LDMUtils.get_pltfilename(asic, filekey, channel, :decay_time))"

# @info "Found decay time at $(round(u"µs", result.µ, digits=2)) for channel $channel / det $det"
# result_pz = (τ = result.μ, fit = result)
# writelprops(asic.par.rpars.pz[period], run, PropDict("$channel" => result_pz))
# @info "Saved pars to disk"
# display(p)


