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
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")
include("$(@__DIR__)/$relPath/src/filteropt_rt_optimization_blnoise.jl")
include("$(@__DIR__)/$relPath/utils/utils_physics.jl")

# daq settings
set = (
    csa_gain = 1.0,
    csa_F = 500*1e-15,
    dynamicrange_V = 2.0,
    bits = 14
)

# init data management 
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(33)
channel = ChannelId(1)
category = DataCategory(:bch)

# settnigs 
filter_type = :trap
waveform_type = :waveform_ch2
n_evts = 5000
reprocess = false 

# load configs and modify if needed 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1:4]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
rt = (0.9:0.5:15.0).*u"µs"
#  _, def_ft = get_fltpars(PropDict(), filter_type, dsp_config)
def_ft = 0.1u"µs"
pd_default = dataprod_config(asic).dsp(filekeys[1]).default
dsp_config_mod = DSPConfig(merge(pd_default, PropDict( 
                    "e_grid_$(filter_type)" => PropDict("rt" => PropDict(:start => minimum(rt), :stop => maximum(rt), :step => step(rt))),
                    "flt_defaults" => PropDict("$(filter_type)" => PropDict(:ft => def_ft, :rt => 1.0u"µs")),
                    "flt_length_cusp" =>  1.0u"µs",
                    "flt_length_cusp" =>  1.0u"µs")))


# load waveforms and do noise sweep 
pars_pd = if isfile(joinpath(data_path(asic.par[category].rpars.noise[period]), "$run.json" )) && !reprocess
    # @info "Noise sweep already done for $category-$period-$run-$channel $waveform_type - you're done!"
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

   

# read result and plot 
result_rt = asic.par[category].rpars.noise[period, run, channel][waveform_type]
f_interp = let enc = result_rt.enc[findall(isfinite.(result_rt.enc))], rt = ustrip.(collect(result_rt.enc_grid_rt))[findall(isfinite.(result_rt.enc))]
    if length(enc) >= 4
        f_interp = BSplineKit.interpolate(rt, enc, BSplineOrder(4))
    else
        f_interp = LinearInterpolation(rt, enc)
    end
end
result_rt[:f_interp] = f_interp
fig = plot_noise_sweep(result_rt; yunit = :ADC)
fig = plot_noise_sweep(result_rt; yunit = :keV)


function plot_noise_sweep(report; yunit = :ADC)
    y_scale = if yunit == :ADC
        1.0 
    elseif yunit == :keV
        pulser_ADC_to_keV(1.0, set.csa_F; bits = set.bits, dynamicrange_V = set.dynamicrange_V, gain = set.csa_gain) 
    elseif yunit == :e
        pulser_ADC_to_electrons(1.0, set.csa_F; bits = set.bits, dynamicrange_V = set.dynamicrange_V, gain = set.csa_gain )
    else
        error("Invalid yunit")
    end 

    x = collect(report.enc_grid_rt)
    x_unit = unit(x[1])
    x = ustrip.(x)
    y = report.enc .* y_scale
    x_inter = range(x[1], stop = maximum(x[findall(isfinite.(y))]), step = 0.05); 
    y_inter = report.f_interp.(x_inter) .* y_scale
 
     # plot  
     p = Figure()
     ax = Axis(p[1, 1], 
         xlabel = "Rise time ($x_unit)", ylabel = "Noise ($yunit)",
         limits = ((extrema(x)[1] - 0.2, extrema(x)[2] + 0.2), (nothing, nothing)),
         title = "Noise sweep ($filter_type), $period-$run-$channel" * @sprintf("fixed ft = %.2f %s, optimal rt = %.1f %s", ustrip(def_ft), unit(def_ft), ustrip(report.rt), unit(report.rt)), )
     lines!(ax, x_inter, y_inter, color = :deepskyblue2, linewidth = 3, linestyle = :solid, label = "Interpolation")
     Makie.scatter!(ax, x, y,  color = :black, label = "Data")
     axislegend()
    #  pname = plt_folder * split(LegendDataManagement.LDMUtils.get_pltfilename(data, filekeys[1], channel, Symbol("noise_sweep_$(filter_type)_$(waveform_type)")),"/")[end]
    #  d = LegendDataManagement.LDMUtils.get_pltfolder(data, filekeys[1], Symbol("noise_sweep_$(filter_type)_$(waveform_type)"))
    #  ifelse(isempty(readdir(d)), rm(d), NamedTuple())
    #  save(pname, p)
    #  @info "Save sanity plot to $pname"
    p
end




