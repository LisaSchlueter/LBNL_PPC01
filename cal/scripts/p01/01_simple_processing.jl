# VERY simple proessing without any filter optimization or classic processing  - for quick tests 
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
using Plots, CairoMakie
using Measures
using Distributions

# load functions from hpge-ana
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/processing_funcs/process_decaytime.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")
Plots_theme()

# setup 
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(2)
channel = ChannelId(1)
category = DataCategory(:cal)
det = _channel2detector(asic, channel)

# 1. load waveforms 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
data = read_ldata(asic, DataTier(:raw), filekeys, channel)
wvfs = data.waveform
Plots.plot(wvfs[rand(1:length(wvfs))], title = "Example waveform", xlabel = "Time", ylabel = "Amplitude [V]", legend = false)

bl_window = 0.0u"µs" .. 18.0u"µs"
tail_window = 22.0u"µs" .. 131.0u"µs"
Plots.vline!([leftendpoint(bl_window), rightendpoint(bl_window)], label = "baseline", color = :red)
Plots.vline!([leftendpoint(tail_window), rightendpoint(tail_window)], label = "tail", color = :violet)

# 2. get decay times for pole-zero correction: 
decay_times = dsp_decay_times(wvfs, bl_window, tail_window)
filter!(x -> 0u"µs" < x < 1000u"µs", decay_times)
Plots.stephist(decay_times, bins = 100, xlabel = "Decay time", ylabel = "Counts", title = "Decay times of waveforms", label = false)
decay_time_pz = nothing
# try 
    min_τ = 250.0u"µs"
    max_τ = 350.0u"µs"
    # if run < DataRun(3)
    rel_cut_fit = 0.25 
    # else  
    #     rel_cut_fit = 0.1
    # end 
    nbins = 100

    cuts_τ = cut_single_peak(decay_times, min_τ, max_τ,; n_bins=nbins, relative_cut=rel_cut_fit)
    result_τ, report = fit_single_trunc_gauss(decay_times, cuts_τ)
    p = Plots.plot(report, legend = :outertop)
    display(p)
    @info "truncated gaus fit - done"
    decay_time_pz = mvalue(result_τ.µ) # decay time for pole-zero 
# catch e
#     try 
#          @info "do simple gaussian fit instead"
#         # filter!(x -> 100u"µs" < x < 400u"µs", decay_times)
#         Plots.stephist(decay_times, bins = 100, xlabel = "Decay time", ylabel = "Counts", title = "Decay times of waveforms", label = false)
#         result_τ   = fit(Normal,ustrip.(decay_times))
#         decay_time_pz = result_τ.µ*u"µs" # decay time for pole-zero 
#     catch e
#         decay_time_pz = median(decay_times)
#     end 
# end 

## 3. do very simple dsp using trap filter 
t0_threshold = 0.01
# shift waveform to baseline
bl_stats = signalstats.(wvfs, leftendpoint(bl_window), rightendpoint(bl_window))
wvfs = shift_waveform.(wvfs, -bl_stats.mean)# substract baseline from waveforms

# pole-zero correction
deconv_flt = InvCRFilter(decay_time_pz)
wvfs_pz = deconv_flt.(wvfs)
i = rand(1:length(wvfs))
Plots.plot(wvfs[i], title = "Example waveform", xlabel = "Time (µs)", ylabel = "Amplitude (V)", label = "raw")
Plots.plot!(wvfs_pz[i], label = "pz - corrected")

# tail_stats = tailstats.(wvfs, leftendpoint(tail_window), rightendpoint(tail_window)) # tail analysis 
# # get raw wvf maximum/minimum
wvf_max = maximum.(wvfs.signal)
wvf_min = minimum.(wvfs.signal)

# get tail mean, std and slope
pz_stats = signalstats.(wvfs, leftendpoint(tail_window), rightendpoint(tail_window))

# characteristic times in waveform 
t0 = get_t0(wvfs, t0_threshold)
t10 = get_threshold(wvfs, wvf_max .* 0.1)
t50 = get_threshold(wvfs, wvf_max .* 0.5)
t90 = get_threshold(wvfs, wvf_max .* 0.9)

# trap-filter: signal estimator for precise energy reconstruction
trap_rt = 3.0u"µs"
trap_ft = 1.0u"µs"
uflt_trap_rtft = TrapezoidalChargeFilter(trap_rt, trap_ft)
signal_estimator = SignalEstimator(PolynomialDNI(3, 100u"ns"))
e_trap = signal_estimator.(uflt_trap_rtft.(wvfs), t50 .+ (trap_rt + trap_ft/2))

dsp_par =   Table(blmean = bl_stats.mean, blslope = bl_stats.slope, 
                 tailmean = pz_stats.mean, tailslope = pz_stats.slope,
                t0 = t0, t10 = t10, t50 = t50, t90 = t90,
                e_max = wvf_max, e_min = wvf_min,
                e_trap = e_trap)
# DSP done. 

# 4. apply some very rough quality cuts  
if run < DataRun(4) 
    dsp_par = filter(x -> 20.0u"µs" .< x.t0 .< 30.0u"µs", dsp_par)
else
    dsp_par = filter(x -> 35.0u"µs" .< x.t0 .< 55.0u"µs", dsp_par)
end
filter!(x ->  x.e_trap .> 0.0, dsp_par)
filter!(x -> abs(x.blslope) .< 0.002 * 1/u"µs", dsp_par)

# energy calibration: very rough. 
ecal_config = dataprod_config(asic).energy(filekeys[1]).default
source = :co60
calib_type = :gamma
gamma_lines =  [ecal_config[Symbol("$(source)_lines")][1]]
left_window_sizes = [ecal_config[Symbol("$(source)_left_window_sizes")][1]]
right_window_sizes = [ecal_config[Symbol("$(source)_right_window_sizes")][1]]
gamma_names = [ecal_config[Symbol("$(source)_names")][1]]
fit_funcs = [Symbol.(ecal_config[Symbol("$(source)_fit_func")])[1]]
gamma_lines_dict = Dict(gamma_names .=> gamma_lines)
e_uncal = dsp_par.e_trap
p = Plots.stephist(e_uncal, bins = 200)
if run < DataRun(3)
    q = 0.5
else
    q = 0.25
end
vline!([quantile(e_uncal,q)])
cal_simple = gamma_lines[1] / quantile(e_uncal,q)

# roughly calibrated spectrum 
e_cal = e_uncal .* cal_simple
Plots.stephist(e_cal, xlabel = "Energy", bins = 0:10:3000)
if run < DataRun(4)
    emin = 800
    emax = 1400
    bins = 0:20:3000
else
    emin = 1070
    emax = 1300
    bins = 0:10:3000
end
Plots.vline!([emin])
Plots.vline!([emax])

e_cal_cut = filter(x-> emin*u"keV" <= x < emax*u"keV", e_cal)
result_fit = fit(Normal, ustrip.(e_cal_cut))
fwhm = round(result_fit.σ * 2.355, digits = 1) *u"keV"
µ = round(mvalue(result_fit.μ) , digits = 1) * u"keV"


h = fit(Histogram, ustrip(e_cal), bins)
ymax = ceil(maximum(h.weights)/10)*10*1.2

Makie_theme(; fs = 18, xgridvisible = false, ygridvisible = false)
fig = Figure(size = (600, 400))
ax = Axis(fig[1, 1], 
        xticks = 0:500:maximum(bins),
        title = "VERY simple calibrated energy specrum (e_trap): $period - $run - $det", 
        xlabel = "Energy (keV)",
        ylabel = "Counts",
        titlesize = 14)
hist!(fig[1, 1], ustrip.(e_cal), bins = bins, color = :dodgerblue)
Makie.ylims!(ax, 0, ymax)
Makie.text!(ustrip(µ), 0.9*ymax, text = "$(µ)\nFWHM = $(fwhm)"; align = (:center, :center), fontsize = 18)
fig

plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :spectrum_approx) * "/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * _get_pltfilename(asic, filekeys[1], channel, Symbol("spectrum_approx_e_trap"))
save(plt_name, fig)