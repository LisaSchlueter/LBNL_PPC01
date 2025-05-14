using LegendDataManagement
using Makie, CairoMakie, LegendMakie
using Measurements: value as mvalue, uncertainty as muncert
using Unitful 
using StatsBase

# setup 
asic = LegendData(:ppc01)
period = DataPeriod(3)
channel = ChannelId(1)
category = :cal 
e_types = [:e_trap]
runs = 42:44 

# load pars 
cable_length = sort([185.0, 405.0, 645.0 ]; rev = true)
fwhm_a = ustrip.([asic.par.rpars.ecal[period, DataRun(r), channel].e_trap.fit[:Co60a].fwhm for r in runs])
fwhm_b = ustrip.([asic.par.rpars.ecal[period, DataRun(r), channel].e_trap.fit[:Co60b].fwhm for r in runs])

# plot 
fig = Figure()
ax = Axis(fig[1,1], xticks = cable_length,  
                    xlabel = "Cable length (cm)", ylabel = "FWHM (keV)")
hlines!(ax, [mvalue(mean(fwhm_a))], linestyle = :dash, alpha = 0.5)
errorbars!(ax, cable_length, mvalue.(fwhm_a), muncert.(fwhm_a), label = "Co60a (1.17 MeV)", whiskerwidth = 0) 
scatter!(ax, cable_length, mvalue.(fwhm_a), label = "Co60a (1.17 MeV)", markersize = 10) 

hlines!(ax, [mvalue(mean(fwhm_b))], linestyle = :dash, alpha = 0.5)
errorbars!(ax, cable_length, ustrip.(mvalue.(fwhm_b)), muncert.(fwhm_b), label = "Co60b (1.33 MeV)", whiskerwidth = 0) 
scatter!(ax, cable_length, mvalue.(fwhm_b), label = "Co60b (1.33 MeV)", markersize = 10) 

axislegend(merge = true, orientation = :horizontal, position = :lt)
# Makie.ylims!(ax, 2.5 ,2.99 )
Makie.ylims!(ax, 2.44 ,2.9 )
fig
save("$(@__DIR__)/plots/Co60_resolution_vs_cablelength.png", fig)

using LegendHDF5IO, Dates
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), :cal , period, DataRun(42)])[1]
d = read_ldata(:timestamp, asic, DataTier(:raw), filekey, channel)
unix2datetime(d[1])