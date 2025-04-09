using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendHDF5IO

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/simple_dsp.jl")
include("$(@__DIR__)/$relPath/processing_funcs/process_dsp.jl")

# inputs 
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
τ_pz = mvalue(get_values(asic.par[category].rpars.pz[period, run, channel]).τ)
pars_filter = asic.par[category].rpars.fltopt[period,run,channel]
# pars_filter = PropDict() # if empty, use default filter parameters 
# do dsp
process_dsp(asic, period, run, category, channel, dsp_config, τ_pz, pars_filter; reprocess = true)

# read dsp pars and plot 
dsp_pars = read_ldata(asic, :jldsp, category, period, run, channel);
Table(dsp_pars)
columnnames(Table(dsp_pars))

e_ADC = filter!(isfinite, dsp_pars.e_trap)
filter!(x-> 4200 < x < 4300, e_ADC)
using Makie, LegendMakie, CairoMakie
fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Energy (ADC)", ylabel = "Counts / bin", limits = ((nothing, nothing), (0, nothing)))
hist!(ax, e_ADC, bins = 100)
fig

