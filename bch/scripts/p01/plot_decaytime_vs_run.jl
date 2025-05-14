# Extract decay time of exponential tail of HPGe waveforms using peak files 
# Perform fit (truncated gaussian) to get average decay time, that is later used for pole-zero correction of all waveforms
# Note: Use peak files instead of all waveforms, because they are "good" waveforms and are quality cuts 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils
using LegendDSP
using LegendDSP: get_fltpars
using LegendHDF5IO
using LegendSpecFits
using RadiationSpectra
using RadiationDetectorDSP
using Measurements: value as mvalue, uncertainty as muncert
using PropDicts
using StatsBase, IntervalSets
using Unitful
using TypedTables
using CairoMakie, Makie, LegendMakie
using Plots 
using Measures
using LinearAlgebra


# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/processing_funcs/process_decaytime.jl")
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:bch)


# plot decay time over time 
runs_warm1 = 1:10
runs_cold1 = 13:22
runs_cold2 = 23:32
τ_w1 = ustrip.([asic.par.rpars.pz[period, DataRun(r), channel].τ for r in runs_warm1])
τ_c1 = ustrip.([asic.par.rpars.pz[period, DataRun(r), channel].τ for r in runs_cold1])
τ_c2 = ustrip.([asic.par.rpars.pz[period, DataRun(r), channel].τ for r in runs_cold2])

fig = Figure()
ax = Axis(fig[1,1], xlabel = "Run number", ylabel = "Decay time (µs)")

Makie.errorbars!(ax, runs_warm1, mvalue.(τ_w1), 100 .*muncert.(τ_w1), label = "03/17: 295 K (errorbars x 100)", whiskerwidth = 0)
Makie.scatter!(ax, runs_warm1,  mvalue.(τ_w1), markersize = 10, label = "03/17: 295 K (errorbars x 100)")

Makie.errorbars!(ax, runs_cold1, mvalue.(τ_c1), 100 .*muncert.(τ_c1), label = "03/18: 90 K (errorbars x 100)", whiskerwidth = 0)
Makie.scatter!(ax, runs_cold1, mvalue.(τ_c1), markersize = 10, label = "03/18: 90 K (errorbars x 100)")

Makie.errorbars!(ax,runs_cold2, mvalue.(τ_c2), 100 .* muncert.(τ_c1), label = "03/20: 90 K (errorbars x 100)", whiskerwidth = 0)
Makie.scatter!(ax, runs_cold2, mvalue.(τ_c2), markersize = 10, label = "03/20: 90 K (errorbars x 100)")

axislegend(merge = true, position = :lb)
fig 
plt_dir = "$(@__DIR__)/plots/"
if !isdir(plt_dir)
    mkpath(plt_dir)
end
save("$(plt_dir)/decaytime_vs_run.png", fig)