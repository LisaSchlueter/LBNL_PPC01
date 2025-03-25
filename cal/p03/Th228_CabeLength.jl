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
runs = 50:52

# load pars 
cable_length = sort([185.0, 405.0, 645.0 ]; rev = false)

filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, DataRun(runs[1])])[1]
gamma_names = Symbol.(dataprod_config(asic).energy(filekey).default.th228_names)
gamma_lines = round.(Int, ustrip.(dataprod_config(asic).energy(filekey).default.th228_lines))

peak = :Tl208FEP
fwhm = Vector{Vector{Measurements.Measurement{Float64}}}(undef, length(gamma_lines))
for (p, peak) in enumerate(gamma_names)
    fwhm[p] = ustrip.([asic.par[category].rpars.ecal[period, DataRun(r), channel].e_trap.fit[peak].fwhm for r in runs])
end

# plot 1 peak (FEP)
i = 7
fig = Figure()
ax = Axis(fig[1,1], xticks = cable_length,  
                    xlabel = "Cable length (cm)", ylabel = "FWHM (keV)")
errorbars!(ax, cable_length, mvalue.(fwhm[i]), muncert.(fwhm[i]), label = "Data: $(gamma_names[i]) ($(gamma_lines[i]) keV)", whiskerwidth = 0) 
scatter!(ax, cable_length, mvalue.(fwhm[i]), label = "Data: $(gamma_names[i]) ($(gamma_lines[i]) keV)", markersize = 10) 
hlines!(ax, [mvalue(mean(fwhm[i]))], linestyle = :dash, alpha = 1.0, label = "mean")
axislegend(merge = true, orientation = :horizontal, position = :lt)
Makie.ylims!(ax, 0.99 * (mvalue(minimum(fwhm[i]))  - muncert(minimum(fwhm[i]))) ,  1.01 * (mvalue(maximum(fwhm[i]))  + muncert(maximum(fwhm[i]))) )
fig
save("$(@__DIR__)/plots/Th228_resolution_vs_cablelength_$(gamma_names[i]).png", fig)

