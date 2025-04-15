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
using Makie, LegendMakie, CairoMakie
using Measures
using Optim
using BSplineKit
using Printf

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/filteropt_rt_optimization_blnoise.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_filteropt.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:bch)
filter_types = [:trap]#, :cusp]

# load configs and modify if needed 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
peak =  Symbol(pz_config.peak)
τ_pz = mvalue(get_values(asic.par.rpars.pz[period, run, channel]).τ)

# START DEBUG
data = asic 
fwhm_rel_cut_fit = 0.1
ft_qmin = 0.02
ft_qmax = 0.98
det = _channel2detector(data, channel)
@info "Optimize filter for period $period, run $run, channel $channel /det $det - $filter_types"

# load waveforms from peakfile
filekeys = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])
if peak == :all 
    data_peak = read_ldata(data, DataTier(:raw), filekeys, channel)
    data_peak = merge(data_peak, (gamma_line = [1170*u"keV"],))
else
    data_peak  = read_ldata((peak), data, :jlpeaks, category, period, run, channel)
end 
wvfs = data_peak.waveform
@debug "Loaded waveforms for peak $peak"

filter_type = filter_types[1]
@info "Optimize filter $filter_type"
_, def_ft = get_fltpars(PropDict(), filter_type, dsp_config)
filekey = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])[1]

# STEP 1: rise-time optimization --> min. baseline noise after filtering 
# if rt_opt_mode == :bl_noise 
    result_rt, report_rt = filteropt_rt_optimization_blnoise(filter_type, wvfs, dsp_config, τ_pz; ft = def_ft)   
    rt_inter = range(ustrip.(report_rt.enc_grid_rt[1]), stop = ustrip(maximum(report_rt.enc_grid_rt[findall(isfinite.(report_rt.enc))])), step = 0.05); 
    p = Figure()
    ax = Axis(p[1, 1], 
        xlabel = "Rise time ($(unit(report_rt.rt)))", ylabel = "Noise (a.u.)",
        limits = ((ustrip.(extrema(report_rt.enc_grid_rt))[1] - 0.2, ustrip.(extrema(report_rt.enc_grid_rt))[2] + 0.2), (nothing, nothing)),
        title = "Noise sweep ($filter_type), $period-$run-$channel, $peak peak \n" * @sprintf("fixed ft = %.2f %s, optimal rt = %.1f %s", ustrip(def_ft), unit(def_ft), ustrip(report_rt.rt), unit(report_rt.rt)), )
    lines!(ax, rt_inter, report_rt.f_interp.(rt_inter), color = :deepskyblue2, linewidth = 3, linestyle = :solid, label = "Interpolation")
    Makie.scatter!(ax, ustrip.(collect(report_rt.enc_grid_rt)), report_rt.enc,  color = :black, label = "Data")
    axislegend()
    p

# elseif rt_opt_mode == :pickoff
#     # for now: do 2 versions or rt...the same using LEGEND DSP function 
#     # pro: + compatibility with Juleana; proper fitting of enc instead of rms
#     # con: - doesnt work well for smaller number of waveforms. result seems to be more noisy, slower?
#     enc_grid = getfield(Main, Symbol("dsp_$(filter_type)_rt_optimization"))(wvfs, dsp_config,  τ_pz; ft=def_ft)
#     enc_min, enc_max = _quantile_truncfit(enc_grid; qmin = 0.02, qmax = 0.98)
#     e_grid_rt   = getproperty(dsp_config, Symbol("e_grid_rt_$(filter_type)"))
#     result_rt, report_rt = fit_enc_sigmas(enc_grid, e_grid_rt, enc_min, enc_max, round(Int,size(enc_grid)[2]/5), 0.1)
#     @info "Found optimal rise-time: $(result_rt.rt) at fixed ft = $def_ft" 
#     p = LegendMakie.lplot(report_rt, title = get_plottitle(filekeys[1], det, "Noise Sweep"; additiional_type=string(filter_type)))
#     pname = plt_folder * split(LegendDataManagement.LDMUtils.get_pltfilename(data, filekeys[1], channel, Symbol("noise_sweep_$(filter_type)_pickoff")),"/")[end]
#     d = LegendDataManagement.LDMUtils.get_pltfolder(data, filekeys[1], Symbol("noise_sweep_$(filter_type)_pickoff"))
#     ifelse(isempty(readdir(d)), rm(d), nothing )
# end 

 # 2. flat top time optimixation 
 e_grid_ft   = getproperty(dsp_config, Symbol("e_grid_ft_$(filter_type)"))
 e_grid = getfield(Main, Symbol("dsp_$(filter_type)_ft_optimization"))(wvfs, dsp_config, τ_pz, mvalue(result_rt.rt))
 e_min, e_max = _quantile_truncfit(e_grid; qmin = ft_qmin, qmax = ft_qmax)
 result_ft, report_ft = fit_fwhm_ft(e_grid, e_grid_ft, result_rt.rt,  e_min, e_max, fwhm_rel_cut_fit; peak = data_peak.gamma_line[1])
 @info "Found optimal flattop-time: $(result_ft.ft) with FWHM $(round(u"keV", result_ft.min_fwhm, digits=2))"
 p = LegendMakie.lplot(report_ft, title = get_plottitle(filekey, det, "$peak FT Scan"; additiional_type=string(filter_type)), juleana_logo = false)



# DEBUG fit_fwhm_ft 
# for f in eachindex(e_grid_ft)
rt = result_rt.rt
f = 1
    # get ft
    ft = e_grid_ft[f]
    # if ft > rt filter doesn't make sense, continue
    if ft > rt
        @debug "FT $ft bigger than RT $rt, skipping"
        # continue
    end
    # get e values for this ft
    e_ft = Array{Float64}(LegendSpecFits.flatview(e_grid)[f, :])
    e_ft = e_ft[isfinite.(e_ft)]

    # sanity check
    min_e  = e_min
    max_e = e_max
    if count(min_e .< e_ft .< max_e) < 100
        @debug "Not enough data points for FT $ft, skipping"
        # continue
        @info "Not enough data points for FT $ft, skipping"
    end
    # cut around peak to increase performance
    n_bins = -1
    rel_cut_fit = fwhm_rel_cut_fit
    fit_cut = LegendSpecFits.cut_single_peak(e_ft, min_e, max_e,; n_bins=n_bins, relative_cut=rel_cut_fit)
    fig = Figure()
    ax = Axis(fig[1,1])
    hist!(ax, e_ft)
    fig
    e_peak_cut = fit_cut.max - 15*(fit_cut.max - fit_cut.low) .< e_ft .< fit_cut.max + 15*(fit_cut.max - fit_cut.low)
    e_ft = e_ft[e_peak_cut]
    
    # create histogram from it
    if isempty(e_ft)
        @debug "Invalid energy vector, skipping"
        continue
    end
    bin_width = 2 * (quantile(e_ft, 0.75) - quantile(e_ft, 0.25)) / ∛(length(e_ft))
    h = fit(Histogram, e_ft, minimum(e_ft):bin_width:maximum(e_ft))

    # create peakstats
    ps = estimate_single_peak_stats_th228(h)
    # check if ps guess is valid
    if any(tuple_to_array(ps) .<= 0)
        @debug "Invalid guess for peakstats, skipping"
        continue
    end
    # fit peak 
    result, _ = fit_single_peak_th228(h, ps,; uncertainty=false)
    # get fwhm
    fwhm[f] = result.fwhm
    fts[f] = ft
    modes[f] = fit_cut.max
    fts_success[f] = true
end