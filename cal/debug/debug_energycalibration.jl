# DEBUG script 
using LegendDataManagement
using LegendDataManagement: readlprops, writelprops
using LegendDataManagement.LDMUtils
using LegendSpecFits
using LegendSpecFits: get_friedman_diaconis_bin_width 
using LegendHDF5IO
using RadiationSpectra
using PropDicts
using Unitful
using TypedTables
using Statistics, StatsBase
using IntervalSets
using Makie, LegendMakie, CairoMakie
using Unitful, Measures
using Measurements: value as mvalue

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
# include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_energy_calibration.jl")

# inputs
reprocess = true 
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(44)
channel = ChannelId(1)
category = :cal 
e_types = [:e_trap]#, :e_cusp]

# load configuration for calibration
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
ecal_config = dataprod_config(asic).energy(filekey).default

# DEBUG START 
data = asic
source = :co60
calib_type = :gamma
gamma_lines =  ecal_config[Symbol("$(source)_lines")]
left_window_sizes = ecal_config[Symbol("$(source)_left_window_sizes")]
right_window_sizes = ecal_config[Symbol("$(source)_right_window_sizes")]
gamma_names = ecal_config[Symbol("$(source)_names")]
fit_funcs = Symbol.(ecal_config[Symbol("$(source)_fit_func")])
gamma_lines_dict = Dict(gamma_names .=> gamma_lines)



# read dsp parameters
dsp_pars = Table(read_ldata(data, :jldsp, category, period, run, channel);)

@debug "Loaded CTC parameters"
pars_ctc = get_values(data.par.rpars.ctc[period, run, channel])
filekey = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])[1]
det = _channel2detector(data, channel)

fs = 12
e_type = :e_trap#_ctc 
 
# load uncalibrated energies after qc and apply ctc if needed
if endswith(string(e_type), "_ctc")
    qdrift = e_uncal = getproperty(dsp_pars, :qdrift)[dsp_pars.qc]
    @debug "Apply CT correction for $e_type"
    e_uncal_func = pars_ctc[e_type_name].func
    e_uncal = ljl_propfunc(e_uncal_func).(Table(NamedTuple{(e_type_name, :qdrift)}((e_uncal, qdrift))))
else
    e_type_name = Symbol(split(string(e_type), "_ctc")[1])
    e_uncal = getproperty(dsp_pars, e_type_name)[dsp_pars.qc]
    e_uncal_func = "$e_type"
end

# do simple calibration and plot 
result_simple, report_simple = simple_calibration(e_uncal, gamma_lines , left_window_sizes, right_window_sizes,; 
            calib_type = calib_type, binning_peak_window=ecal_config.binning_peak_window, quantile_perc=NaN, 
            peak_quantile= ecal_config.left_peak_quantile..ecal_config.right_peak_quantile, 
            bin_quantile = ecal_config.left_bin_quantile..ecal_config.right_bin_quantile, 
            peakfinder_threshold = ecal_config.peakfinder_threshold, 
            peakfinder_σ = ecal_config.peakfinder_σ);

fig = LegendMakie.lplot(report_simple, juleana_logo = false)
[delete!(leg) for leg in fig.content if leg isa Legend]
vl = vlines!([report_simple.peak_guess * ustrip(report_simple.c)], color = :red2, label = "Peak Guess", alpha = 0.5, linewidth = 3)
axislegend(Makie.current_axis(), [vl], ["Peak Guess"], position = :lt)
Makie.xlims!(Makie.current_axis(), 300, ustrip(maximum(gamma_lines)) + 100)
fig
m_cal_simple = result_simple.c
filekey = search_disk(FileKey, data.tier[DataTier(:raw), category , period, run])[1]

report_simple2 = (h_calsimple = report_simple.h_calsimple, 
h_uncal = report_simple.h_uncal,
c = report_simple.c,
fep_guess = report_simple.peak_guess,
peakhists = report_simple.peakhists,
peakstats = report_simple.peakstats)
   
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(data, filekey, :energy_calibration) * "/"
det = _channel2detector(data, channel)
p = LegendMakie.lplot(report_simple2, titlesize = 16, title = get_plottitle(filekey, det, "Simple Calibration"; additiional_type=string(e_type)), cal = true, juleana_logo = false)
p

# # fit peaks
@debug "Fit all peaks"
result_fit, report_fit = fit_peaks(result_simple.peakhists, result_simple.peakstats, gamma_names; 
                            e_unit=result_simple.unit, calib_type=:th228, fit_func = fit_funcs, m_cal_simple=m_cal_simple)
fig_fit = LegendMakie.lplot(report_fit, titlesize = 16, figsize = (600, 400*length(report_fit)), title = get_plottitle(filekey, det, "Peak Fits"; additiional_type=string(e_type)), juleana_logo = false)
fig_fit
pname = plt_folder * _get_pltfilename(data, filekey, channel, Symbol("peak_fits_$(e_type)"))
save(pname, fig_fit)
@info "Save peak fits plot to $(pname)"

# # calibration curve 
gamma_names_cal_fit = [p for p in gamma_names if !(p in ecal_config.cal_fit_excluded_peaks)]
μ_fit =  [result_fit[p].centroid for p in gamma_names_cal_fit]
pp_fit = [gamma_lines_dict[p] for p in gamma_names_cal_fit]
result_calib, report_calib = fit_calibration(ecal_config.cal_pol_order, μ_fit, pp_fit; e_expression=e_uncal_func)
@debug "Found $e_type calibration curve: $(result_calib.func)"
# plot calibration curve 
μ_notfit =  [result_fit[p].centroid for p in gamma_names if !(p in gamma_names_cal_fit)]
pp_notfit = [gamma_lines_dict[p] for p in gamma_names if !(p in gamma_names_cal_fit)]
# pname_calib = plt_folder * _get_pltfilename(data, filekey, channel, Symbol("calibration_curve_$(e_type)"))
fig_calib = Figure()
# g = Makie.GridLayout(fig_calib[2,1])
LegendMakie.lplot!(report_calib, xerrscaling=100, additional_pts=(μ = μ_notfit, peaks = pp_notfit), titlesize = 17, title = get_plottitle(filekey, det, "Calibration Curve"; additional_type=string(e_type)), juleana_logo = false)
replace_resplot(fig_calib, report_calib)
# if res_max > 5.0
#     Makie.ylims!(axs[2], -res_max*1.2, res_max*1.2 )
# end 

function replace_resplot(fig, report)
    # replace residual plot with percent plot . 
    # legs = fig.content[findall(map(x -> x isa Legend, fig.content))]
    axs =  fig.content[findall(map(x -> x isa Axis, fig.content))]
    xvalues = mvalue.(report.x)
    yvalues = mvalue.(report.y)
    yfit_values = mvalue.(report.f_fit(xvalues))
    residuals = 100 * (yvalues .- yfit_values) ./ yfit_values
    res_max =  maximum(abs.(residuals))
    res_max = res_max < 0.15 ? 0.15 : res_max
    x_nofit = mvalue.(μ_notfit)
    y_nofit = mvalue.(ustrip.(pp_notfit))
    residuals_nofit = 100 * (y_nofit .- mvalue.(report.f_fit(x_nofit))) ./ mvalue.(report.f_fit(x_nofit))
    xlabel = axs[2].xlabel.val
    delete!(axs[2])
    g = fig.layout
    ax2 = Makie.Axis(g[2,1], ;xlabel = xlabel, ylabel = "Residuals (%)", 
                yticks = -0.1:0.1:0.1,
                limits = ((nothing, nothing),( -1.2 * res_max, 1.2 * res_max) ))

    Makie.hspan!(ax2, [-0.1], [0.1], color = :silver, alpha = 0.7)
    Makie.hlines!(ax2, [0], [1], color = :silver, linestyle = :solid, linewidth = 2)
    if !isempty(x_nofit)
        Makie.scatter!(ax2, x_nofit, residuals_nofit, color = :silver, strokecolor = :black, strokewidth = 1)
    end 
    Makie.scatter!(ax2, xvalues, residuals, color = :black)#, markersize = markersize)
    fig
    Makie.linkxaxes!(axs[1], ax2)
    Makie.rowgap!(g, -100); fig
    Makie.rowsize!(g, 1, Makie.Auto(4)); fig
    # align y labels 
    yspace = maximum(Makie.tight_yticklabel_spacing!, (axs[1], ax2))
    axs[1].yticklabelspace = yspace
    ax2.yticklabelspace = yspace
    fig 
end 


# save(pname_calib, fig_calib)
@info "Save calibration curve plot to $(pname_calib)"

# # resolution curve 
f_cal_widths(x) = report_calib.f_fit(x) .* report_calib.e_unit .- first(report_calib.par)
gamma_names_fwhm_fit = [p for p in gamma_names if !(p in Symbol.(ecal_config.fwhm_fit_excluded_peaks))]
fwhm_fit = f_cal_widths.([result_fit[p].fwhm for p in gamma_names_fwhm_fit])
pp_fit = [gamma_lines_dict[p] for p in gamma_names_fwhm_fit] 
result_fwhm, report_fwhm = fit_fwhm(ecal_config.fwhm_pol_order, pp_fit, fwhm_fit; e_type_cal=Symbol("$(e_type)_cal"), e_expression=e_uncal_func, uncertainty=true)
@debug "Found $e_type FWHM: $(round(u"keV", result_fwhm.qbb, digits=2))"
# plot resolution curve
fwhm_notfit =  f_cal_widths.([result_fit[p].fwhm for p in gamma_names if !(p in gamma_names_fwhm_fit)])
pp_notfit = [gamma_lines_dict[p] for p in gamma_names if !(p in gamma_names_fwhm_fit)]
# pname_fwhm = plt_folder * _get_pltfilename(data, filekey, channel, Symbol("fwhm_$(e_type)"))
fig_fwhm = LegendMakie.lplot(report_fwhm, additional_pts=(peaks = pp_notfit, fwhm = fwhm_notfit), titlesize = 17, title = get_plottitle(filekey, det, "FWHM"; additiional_type=string(e_type)), juleana_logo = false)
# if isempty(fwhm_notfit)
#     # plot(report_fwhm)
#     LegendMakie.lplot(report_fwhm, title = get_plottitle(filekey, det, "FWHM"; additiional_type=string(e_type)), juleana_logo = false)
# else
#     # plot(report_fwhm, additional_pts=(peaks = pp_notfit, fwhm = fwhm_notfit))
#     LegendMakie.lplot(report_fwhm, additional_pts=(peaks = pp_notfit, fwhm = fwhm_notfit), title = get_plottitle(filekey, det, "FWHM"; additiional_type=string(e_type)), juleana_logo = false)
# end 
# plot!(fig_fwhm, thickness_scaling = 1.5, xticks = 0:500:3000, legend = :bottomright, xlabel = "Energy (keV)", right_margin = 3mm)
# plot!(plot_title=get_plottitle(filekey, _channel2detector(data, channel), "FWHM"; additiional_type=string(e_type)), plot_titlelocation=(0.5,0.3), plot_titlefontsize=8)
# savefig(fig_fwhm, pname_fwhm)
save(pname_fwhm, fig_fwhm)
@info "Save FWHM plot to $(pname_fwhm)"

# calibrate best fit values using energy calibration curve 
f_cal_pos(x) = report_calib.f_fit(x) .* report_calib.e_unit
width_pars = [:σ, :fwhm]
position_pars = [:μ, :centroid]
for (peak, result_peak) in result_fit
    result_peak = merge(result_peak, NamedTuple{Tuple(width_pars)}([f_cal_widths(result_peak[k]) for k in width_pars]...))
    result_peak = merge(result_peak, NamedTuple{Tuple(position_pars)}([f_cal_pos(result_peak[k]) for k in position_pars]...))
    result_fit[peak] = result_peak
end
result_fit[gamma_names[1]].fwhm
result_fit[gamma_names[2]].fwhm

result_energy = (
    m_cal_simple = m_cal_simple,
    fwhm = result_fwhm,
    cal = result_calib,
    fit  = result_fit,
)

