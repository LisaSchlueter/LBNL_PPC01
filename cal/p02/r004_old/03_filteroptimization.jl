env_path = "$(@__DIR__)/" * relpath(split(@__DIR__, "LBNL_PPC01")[1], @__DIR__)  * "/LBNL_PPC01/"
import Pkg; Pkg.activate(env_path)
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using LegendSpecFits
using LegendDSP
using LegendDSP: get_fltpars
using RadiationDetectorSignals
using RadiationDetectorDSP
using Measurements: value as mvalue
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using Plots
using Measures
using Optim
using BSplineKit
using Printf

# set data configuration (where to find data; and where to save results)
if gethostname() == "Lisas-MacBook-Pro.local"
  ENV["LEGEND_DATA_CONFIG"] = "/Users/lisa/Documents/Workspace/LEGEND/LBL_ASIC/ASIC_data/ppc01/config.json"
else # on NERSC 
  ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"
end 
# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/filteropt_rt_optimization_blnoise.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_filteropt.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(4)
channel = ChannelId(1)
category = DataCategory(:cal)


# # do optimization 
process_filteropt(asic, period, run, category, channel; reprocess = true, rt_opt_mode = :bl_noise)

# # read filter optimization pars
# fltopt_pars = asic.par.rpars.fltopt[period,run,channel]


# MANUAL filter optimization 
data = asic
dsp_config = DSPConfig(data.metadata.config.dsp.dsp_config.default)
peak =  Symbol(data.metadata.config.dsp.dsp_config.pz.default.peak)
τ_pz = mvalue(get_values(data.par.rpars.pz[period, run, channel]).τ) ## load decay time for pole-zero correction 

fltopt_file = joinpath(mkpath(data_path(data.par.rpars.fltopt[period])), "$(string(run)).json")
if isfile(fltopt_file) && !reprocess
    @info "Filter optimization file already exist for $category period $period - run $run - channel $channel - you're done!"
    return
end

# prepare results dict
mkpath(joinpath(data_path(data.par.rpars.fltopt), string(period)))
result_filteropt_dict = Dict{Symbol, NamedTuple}()
@debug "Created path for filter optimization results"

  # load waveforms from peakfile
data_peak  = read_ldata((peak), data, :jlpeaks, category, period, run, channel)
wvfs = data_peak.waveform
@debug "Loaded waveforms for peak $peak"

filter_type = :trap
rt_opt_mode = :bl_noise
#   function process_filteropt_fltr(filter_type::Symbol)
@info "Optimize filter $filter_type"
_, def_ft = get_fltpars(PropDict(), filter_type, dsp_config)

filekey = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])[1]

# STEP 1: rise-time optimization --> min. baseline noise after filtering 
result_rt, report_rt = filteropt_rt_optimization_blnoise(filter_type, wvfs, dsp_config, τ_pz; ft = def_ft)
Plots_theme()
p = scatter(ustrip.(collect(report_rt.enc_grid_rt)), report_rt.enc, size = (600, 400), dpi = 150, 
            ms = 4, color = :deepskyblue2, markerstrokecolor = :deepskyblue2,
            xlabel = "Rise time ($(unit(report_rt.rt)))", ylabel = "Noise (a.u.)", 
            title = "Noise sweep ($filter_type); fixed ft = $(def_ft) \n $period, $run, $channel, $peak peak",
            label = "Data")
plot!([ustrip(report_rt.rt), ustrip(report_rt.rt)], [ylims()[1], report_rt.min_enc], color = :red2, linewidth = 2, linestyle = :dot,
                label = @sprintf("Optimal rise-time = %.1f %s", ustrip(report_rt.rt), unit(report_rt.rt)))
plot!([xlims()[1], ustrip(report_rt.rt)], [report_rt.min_enc, report_rt.min_enc], color = :red2, linewidth = 2, linestyle = :dot, label = false)
savelfig(savefig,p , data, filekey, channel,  Symbol("noise_sweep_$(filter_type)_blnoise"))
@info "Save sanity plot to $(LegendDataManagement.LDMUtils.get_pltfilename(data, filekey, channel, :fltopt_rt))"

# 2. flat top time optimization 
e_grid_ft   = (0)getproperty(dsp_config, Symbol("e_grid_ft_$(filter_type)"))
e_grid = getfield(Main, Symbol("dsp_$(filter_type)_ft_optimization"))(wvfs, dsp_config, τ_pz, mvalue(result_rt.rt))
e_min, e_max = _quantile_truncfit(e_grid; qmin = 0.02, qmax = 0.98)
result_ft, report_ft = fit_fwhm_ft(e_grid, e_grid_ft, result_rt.rt,  e_min, e_max, 0.1; peak = data_peak.gamma_line[1])
@info "Found optimal flattop-time: $(result_ft.ft) with FWHM $(round(u"keV", result_ft.min_fwhm, digits=2))"

p = plot(report_ft, legendfontsize = 13, xlabel = "Flat-top time (µs)", ylabel = "FWHM $peak (keV)")
ymin = ifelse(mvalue(ustrip(result_ft.min_fwhm)) - 0.2 < 0, 0, mvalue(ustrip(result_ft.min_fwhm)) - 1)
ymax = maximum(mvalue.(ustrip.(report_ft.fwhm))) + 0.2
ylims!(ymin, ymax)
# title!("$filter_type" * @sprintf("filter: flattop-time optimization; rt = %.2f %s", ustrip(result_rt.rt), unit(result_rt.rt)) * "\n $period, $run, $channel, $peak peak")
title!(get_plottitle(filekey, det, "$peak FT Scan"; additiional_type=string(filter_type)))
savelfig(savefig, p, data, filekey, channel, Symbol("fwhm_ft_scan_$(filter_type)"))

# return result for this filter type
merge(result_rt, result_ft)
#   end 

# filter optimization: rise-time, and flat-top times
for filter_type in filter_types
    result_filteropt_dict[filter_type] =  process_filteropt_fltr(filter_type)
end
result = PropDict(Dict("$channel" => result_filteropt_dict))
writelprops(data.par.rpars.fltopt[period], run, result)
@info "Saved pars to disk"