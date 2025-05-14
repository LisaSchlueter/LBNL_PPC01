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
using CairoMakie, LegendPlots
using Measures
using Optim
using Interpolations, BSplineKit
using Printf
# set data configuration (where to find data; and where to save results)

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/filteropt_rt_optimization_blnoise.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_filteropt.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(2)
# begin 
channel = ChannelId(1)
category = DataCategory(:cal)
filter_types = [:trap, :cusp]
rt_opt_mode = :bl_noise

# load configs and modify if needed 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
pz_config = dataprod_config(asic).dsp(filekeys[1]).pz.default
peak =  Symbol(pz_config.peak)
τ_pz = mvalue(get_values(asic.par.rpars.pz[period, run, channel]).τ)

# Load DSP config and modify rise-time grid for trap filter
# dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
dsp_config_pd = dataprod_config(asic).dsp(filekeys[1]).default#.e_grid_trap.rt
dsp_config_pd.e_grid_trap.rt = PropDict(:start => 0.1u"µs", :stop => 2.5u"µs", :step => 0.2u"µs")
dsp_config = DSPConfig(dsp_config_pd)

# data = read_ldata(asic, DataTier(:raw), filekeys, channel)
data  = read_ldata((peak), asic, :jlpeaks, category, period, run, channel)
wvfs = data.waveform
def_ft = 0.5u"µs"

filter_type = :trap 
grid_rt   = getproperty(dsp_config, Symbol("e_grid_rt_$(filter_type)"))
result_rt, report_rt = filteropt_rt_optimization_blnoise(filter_type, wvfs, dsp_config, τ_pz; ft = def_ft)

rt_inter = range(ustrip.(report_rt.enc_grid_rt[1]), stop = ustrip(maximum(report_rt.enc_grid_rt[findall(isfinite.(report_rt.enc))])), step = 0.05); 
p = plot(rt_inter, report_rt.f_interp.(rt_inter), color = :deepskyblue2, linewidth = 3, linestyle = :solid, label = "Interpolation")
Plots.scatter!(ustrip.(collect(report_rt.enc_grid_rt)), report_rt.enc, 
                size = (600, 400), dpi = 150, 
                ms = 4, color = :black, markerstrokecolor = :black,
                xlabel = "Rise time ($(unit(report_rt.rt)))", ylabel = "Noise (a.u.)", 
                title = "Noise sweep ($filter_type), $period-$run-$channel, $peak peak \n" * @sprintf("fixed ft = %.2f %s, optimal rt = %.1f %s", ustrip(def_ft), unit(def_ft), ustrip(report_rt.rt), unit(report_rt.rt)),
                label = "Data",
                legend = :top)
Plots.xlims!(ustrip.(extrema(report_rt.enc_grid_rt))[1] - 0.2, ustrip.(extrema(report_rt.enc_grid_rt))[2] + 0.2)

# ## DEBUG 
# ft = def_ft
# bl_window = dsp_config.bl_window

# # waveforms: shift baseline and deconvolute (pole-zero)
# bl_stats = signalstats.(wvfs, leftendpoint(bl_window), rightendpoint(bl_window))
# wvfs_shift = shift_waveform.(wvfs, -bl_stats.mean)
# deconv_flt = InvCRFilter(τ_pz)
# wvfs_pz = deconv_flt.(wvfs_shift)

# # STEP 1: rise-time optimization --> min. baseline noise after filtering 
# noise = zeros(length(grid_rt))
# for (i, rt) in enumerate(collect(grid_rt))
#     if filter_type == :trap
#         wvfs_flt =  TrapezoidalChargeFilter(rt, ft).(wvfs_pz)   # filtered waveforms 
#     elseif filter_type == :cusp
#         wvfs_flt = CUSPChargeFilter(rt, ft, τ_cusp, flt_length_cusp, cusp_scale).(wvfs_pz) 
#     elseif filter_type == :zac
#         wvfs_flt = ZACChargeFilter(rt, ft, τ_zac, flt_length_zac, zac_scale).(wvfs_pz)
#     end
#     valid_bins = findall(leftendpoint(bl_window) .<=  wvfs_flt[1].time .<=rightendpoint(bl_window)) # bins within baseline of filtered waveform 
#     bl_trap = filter.(isfinite, map(x-> x.signal[valid_bins], wvfs_flt)) # baselines - cut bins in beginning and end 
#     noise[i] = rms(vcat(bl_trap...))
# end

# # find optimal shaping time: 
# if  all(.!isfinite.(noise))
#     @error "Error: no finite noise values found for filter type $filter_type - try adjusting grid_rt."
# end
# result_rt, report_rt = let enc = noise[findall(isfinite.(noise))], rt = ustrip.(collect(grid_rt))[findall(isfinite.(noise))]
#     if length(enc) >= 4
#         f_interp = interpolate(rt, enc, BSplineOrder(4))
#     else
#         f_interp = LinearInterpolation(rt, enc)
#     end
#     result = optimize(f_interp, minimum(rt), maximum(rt)) 
#     rt_opt = Optim.minimizer(result)*u"µs" # optimial rise time for trap filter
#     min_enc = Optim.minimum(result)
#     (rt = rt_opt, min_enc = min_enc), (rt = rt_opt, min_enc = min_enc, enc_grid_rt = grid_rt, enc = noise, f_interp = f_interp)
# end

# rt_inter = range(ustrip.(report_rt.enc_grid_rt[1]), stop = ustrip(maximum(report_rt.enc_grid_rt[findall(isfinite.(report_rt.enc))])), step = 0.05); 
# p = plot(rt_inter, report_rt.f_interp.(rt_inter),color = :silver, linewidth = 2, linestyle = :dot, label = false)
# Plots.scatter!(ustrip.(collect(report_rt.enc_grid_rt)), report_rt.enc, size = (600, 400), dpi = 150, 
#     ms = 4, color = :deepskyblue2, markerstrokecolor = :deepskyblue2,
#     xlabel = "Rise time ($(unit(report_rt.rt)))", ylabel = "Noise (a.u.)", 
#     title = "Noise sweep ($filter_type); fixed ft = $(def_ft) \n $period, $run, $channel, $peak peak",
#     label = "Data")
#  ## DEBUG END 

# enc_grid = getfield(Main, Symbol("dsp_$(filter_type)_rt_optimization"))(wvfs, dsp_config,  τ_pz; ft=def_ft)
# enc_min, enc_max = _quantile_truncfit(enc_grid; qmin = 0.02, qmax = 0.98)
# e_grid_rt   = getproperty(dsp_config, Symbol("e_grid_rt_$(filter_type)"))
# # result_rt, report_rt = fit_enc_sigmas(enc_grid, e_grid_rt, enc_min, enc_max, round(Int,size(enc_grid)[2]/5), 0.1)
# # @info "Found optimal rise-time: $(result_rt.rt) at fixed ft = $def_ft"
# # p = Plots.plot(report_rt)

# fig = Figure()
# ax = Axis(fig[1, 1],  xlabel = "Rise time ($(unit(report_rt.rt)))", ylabel = "RMS noise (a.u.)", 
#         title = "Noise sweep ($filter_type) $period-$run-$channel",
#         xgridvisible = true, ygridvisible = true,)
#     p = lines!(ax, ustrip.(collect(report_rt.enc_grid_rt)), report_rt.enc, 
#     label = "trap. filter: ft = $(def_ft)")
#     Makie.scatter!(ax, ustrip.(collect(report_rt.rt)), report_rt.min_enc, color = :orange, 
#     label = @sprintf("min ENC = %.1e (a.u.)",report_rt.min_enc))
# axislegend(position = :lt)
# fig
# plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :noise) * "/"
# if !isdir(plt_folder)
#     mkdir(plt_folder)
# end
# plt_name = plt_folder * _get_pltfilename(asic, filekeys[1], channel, Symbol("noise-sweep_$(filter_type)_ft$(def_ft)"))
# # save(plt_name, fig)
# # end 
# fig