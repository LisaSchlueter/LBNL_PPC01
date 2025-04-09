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
using BSplineKit, Interpolations
using Printf

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/src/filteropt_rt_optimization_blnoise.jl")
include("$(@__DIR__)/$relPath/utils/utils_physics.jl")

# init data management 
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(3)
channel = ChannelId(1)
category = DataCategory(:bch)
det_ged = _channel2detector(asic, channel)

# load configs and modify if needed 
filter_type = :trap
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])#[1:4]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
# rt = (4.0:1.0:40.5).*u"µs"
rt = (1.0:0.5:12.5).*u"µs"
# rt = (0.5:1.0:50.5).*u"µs"
def_ft = 0.1u"µs"
pd_default = dataprod_config(asic).dsp(filekeys[1]).default
dsp_config_mod = DSPConfig(merge(pd_default, PropDict( 
                    "e_grid_$(filter_type)" => PropDict("rt" => PropDict(:start => minimum(rt), :stop => maximum(rt), :step => step(rt))),
                    "flt_defaults" => PropDict("$(filter_type)" => PropDict(:ft => def_ft, :rt => 1.0u"µs")),
                    "flt_length_cusp" =>  1.0u"µs",
                    "flt_length_cusp" =>  1.0u"µs")))

# settings 
filter_type = :trap
waveform_type = :waveform
diff_output = true 
reprocess = true

# daq settings
if parse(Int, string(period)[2:end]) == 1
    gain = if parse(Int, "$run"[2:end]) <= 35
        1.0
    else
        11.9
    end
    C_f = 500.0*1e-15
    C_inj = 500.0*1e-15
    n_evts = 100
elseif parse(Int, string(period)[2:end]) == 2
    if parse(Int, "$run"[2:end]) <= 2
        gain =  48.0
        n_evts = 100
    else
        gain = 100.0
        n_evts = 10 
    end
    C_inj = 3000.0*1e-15
    C_f = 500.0*1e-15
end
set = (gain = gain, C_f = C_f, C_inj = C_inj)

# load waveforms and do noise sweep 
begin 
    pars_pd = if isfile(joinpath(data_path(asic.par[category].rpars.noise[period]), "$run.json" )) && !reprocess
        @info "Noise sweep already done for $category-$period-$run-$channel $waveform_type - load results!"
        asic.par[category].rpars.noise[period,run, channel]
    else 
        PropDict()
    end 

    if haskey(pars_pd, Symbol(waveform_type))
        @info "Noise sweep already done for $category-$period-$run-$channel $waveform_type - you're done!"
    else 
        # load data and do noise sweep 
        data = read_ldata(asic, DataTier(:raw), filekeys, channel)
        wvfs = data[waveform_type][1:n_evts]
        result_rt, report_rt = noise_sweep(filter_type, wvfs, dsp_config_mod)
    
        # merge with previous results and save. 
        pars_pd = merge(pars_pd, Dict(Symbol("$(waveform_type)") => result_rt))
        writelprops(asic.par[category].rpars.noise[period], run, PropDict("$channel" => pars_pd))
    end 
    # read result 
    result_rt = asic.par[category].rpars.noise[period, run, channel][waveform_type]
end 

#  plot 
f_interp = let enc = result_rt.noise, rt = ustrip.(result_rt.rt)
    if length(enc) >= 4
        f_interp = BSplineKit.interpolate(rt, enc, BSplineOrder(4))
    else
        f_interp = LinearInterpolation(rt, enc)
    end
end
result_rt[:f_interp] = f_interp

fig = plot_noise_sweep_osci(result_rt;  yunit = :e, gain = set.gain, C_f = set.C_f, ) # V only for osci. 
fig = plot_noise_sweep_osci(result_rt;  yunit = :µV, gain = set.gain, C_f = set.C_f, ) # V only for osci.
function plot_noise_sweep_osci(report; yunit = :µV, gain = set.gain, C_f = set.C_f, C_inj = set.C_inj)
    gain = gain * C_inj / C_f
    y_scale = if yunit == :µV
        1e6 / gain
    elseif yunit ==:e
        V_to_electrons(1.0, C_inj; gain = gain) 
    end
    x = report.rt
    x_unit = unit(x[1])
    x = ustrip.(x)
    y = report.noise .* y_scale
    x_inter = range(x[1], stop = maximum(x[findall(isfinite.(y))]), step = 0.05); 
    y_inter = report.f_interp.(x_inter) .* y_scale
 
    # plot  
    fig = Figure()
    ax = Axis(fig[1, 1], 
        xlabel = "Rise time ($x_unit)", ylabel = "Noise / gain ($yunit)",
        limits = ((extrema(x)[1] - 0.2, extrema(x)[2] + 0.2), (nothing, nothing)),
        title = get_plottitle(filekeys[1], det_ged, "Noise sweep (gain $(round(gain, digits = 1)))") * @sprintf("\nft = %.2f %s, rt opt. = %.1f %s, noise min = %.2f %s", ustrip(def_ft), unit(def_ft), ustrip(report.rt_opt), unit(report.rt_opt), report.min_noise * y_scale, yunit)) 
    lines!(ax, x_inter, y_inter, color = :deepskyblue2, linewidth = 3, linestyle = :solid, label = "Interpolation")
    Makie.scatter!(ax, x, y,  color = :black, label = "Data")
    axislegend(position = :rt)

    plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :noise_sweep) * "/"
    pname = plt_folder *  _get_pltfilename(asic, filekeys[1], channel, Symbol("noisesweep_$(filter_type)_$(waveform_type)_osci"))
    save(pname, fig)
    @info "Save plot to $pname"
    fig 
end
