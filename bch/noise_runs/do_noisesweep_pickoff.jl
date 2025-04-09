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
period = DataPeriod(1)
run = DataRun(38)
channel = ChannelId(1)
category = DataCategory(:bch)
det_ged = _channel2detector(asic, channel)

# load configs and modify if needed 
filter_type = :trap
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])#[1:4]
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
# rt = (0.5:0.5:25.0).*u"µs"
rt = (0.5:1.0:35.0).*u"µs"
def_ft = 0.1u"µs"
pd_default = dataprod_config(asic).dsp(filekeys[1]).default
dsp_config_mod = DSPConfig(merge(pd_default, PropDict( 
                    "e_grid_$(filter_type)" => PropDict("rt" => PropDict(:start => minimum(rt), :stop => maximum(rt), :step => step(rt))),
                    "flt_defaults" => PropDict("$(filter_type)" => PropDict(:ft => def_ft, :rt => 1.0u"µs")),
                    "flt_length_cusp" =>  1.0u"µs",
                    "flt_length_cusp" =>  1.0u"µs"),
                    ))

filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])#[1:10]
data_raw = read_ldata(asic, DataTier(:raw), filekeys, channel)
wvfs = data_raw.waveform

# do noise sweep 
enc_grid = getfield(Main, Symbol("dsp_$(filter_type)_rt_optimization"))(wvfs, dsp_config_mod,  1000000.0u"µs"; ft=def_ft)
e_grid_rt  = getproperty(dsp_config_mod, Symbol("e_grid_rt_$(filter_type)"))
enc_min, enc_max = _quantile_truncfit(enc_grid; qmin = 0.02, qmax = 0.98)
result_rt, report_rt = fit_enc_sigmas(enc_grid, e_grid_rt, enc_min, enc_max, round(Int,size(enc_grid)[2]/5), 0.1)
@info "Found optimal rise-time: $(result_rt.rt) at fixed ft = $def_ft" 

# plot results
p = LegendMakie.lplot(report_rt, title = get_plottitle(filekeys[1], det_ged, "Noise sweep pickoff"; additiional_type=string(filter_type)))
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :noise_sweep) * "/"
pname = plt_folder * split(LegendDataManagement.LDMUtils.get_pltfilename(asic, filekeys[1], channel, Symbol("noise_sweep_$(filter_type)_pickoff")),"/")[end]
d = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], Symbol("noise_sweep_$(filter_type)_pickoff"))
ifelse(isempty(readdir(d)), rm(d), nothing )
save(pname, p)
@info "Save plot to $pname"
