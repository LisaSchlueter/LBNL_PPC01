using LegendDataManagement
using LegendDataManagement.LDMUtils
using CairoMakie, LegendPlots
using LegendHDF5IO
using Unitful
using TypedTables
using StatsBase
using ColorSchemes
asic = LegendData(:ppc01)
period = DataPeriod(1)
category = :cal 
channel = ChannelId(1)
runs = [6, 7, 8, 9, 10, 11, 12, 13]
bias_kV = [collect(1.3:0.3:2.5)...,2.7, 2.9, 3.0]

# load fwhm from energy calibraiton pars
fwhm_Co60a = [asic.par.rpars.ecal[period, DataRun(r), channel].e_trap.fit.Co60a.fwhm for r in runs]
fwhm_Co60b = [asic.par.rpars.ecal[period, DataRun(r), channel].e_trap.fit.Co60b.fwhm for r in runs]
peaks =  round.(ustrip.(asic.par.rpars.ecal[period, DataRun(6), channel].e_trap.cal.peaks), digits = 1 )
palette = ColorSchemes.twelvebitrainbow
colors = [get(palette, i/length(runs)) for i in eachindex(runs)]

# xlabel_str = "Rise time t$(Makie.UnicodeFun.to_subscript(80)) - t$(Makie.UnicodeFun.to_subscript(20)) (ns)"
# risetimes = risetimes_t8020
# xlims = (30, 180)

fig = Figure()
ax = Axis(fig[1, 1], xlabel = "Detector bias voltage (kV)", ylabel = "FWHM (keV)", limits = ((nothing, nothing), (nothing,nothing)), #(3.4, 5.3)
                title = "LBNL-PPC01 and ASIC L1k65n with buffer:  $period, r00$(runs[1])-r0$(runs[end])", titlesize = 16)
for i in eachindex(runs)
    scatter!(ax, bias_kV[i], ustrip.(fwhm_Co60a[i]), color = colors[1], label = "$(Makie.UnicodeFun.to_superscript(60))Co a) $(peaks[1]) keV")
    scatter!(ax, bias_kV[i], ustrip.(fwhm_Co60b[i]), color = colors[end-1], label = "$(Makie.UnicodeFun.to_superscript(60))Co b) $(peaks[2]) keV")
end 
axislegend("Gamma peak", position = :rt, merge = true)
fig

plt_folder = "$(@__DIR__)/plots/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * "biasvoltage_fwhm.png"
save(plt_name, fig)