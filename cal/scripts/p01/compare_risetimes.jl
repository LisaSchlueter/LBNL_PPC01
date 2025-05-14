using LegendDataManagement
using LegendDataManagement.LDMUtils
using CairoMakie, LegendMakie, Makie
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
# load dsp parameters and apply qc cuts
qc_cuts = [asic.par.rpars.qc[period][DataRun(r), channel].wvf_keep.all for r in runs]
dsp_pars = [Table(read_ldata(asic, :jldsp, category, period, DataRun(runs[i]), channel))[qc_cuts[i]] for i in eachindex(runs)];
risetimes_t9010 = [dsp_pars[i].t90 .- dsp_pars[i].t10 for i in eachindex(runs)]
risetimes_t8020 = [dsp_pars[i].t80 .- dsp_pars[i].t20 for i in eachindex(runs)]

palette = ColorSchemes.twelvebitrainbow
colors = [get(palette, i/length(runs)) for i in eachindex(runs)]

mode = :t9010 #:t8020
if mode == :t8020
    xlabel_str = "Rise time t$(Makie.UnicodeFun.to_subscript(80)) - t$(Makie.UnicodeFun.to_subscript(20)) (ns)"
    risetimes = risetimes_t8020
    xlims = (30, 180)
else
    xlabel_str = "Rise time t$(Makie.UnicodeFun.to_subscript(90)) - t$(Makie.UnicodeFun.to_subscript(10)) (ns)"
    risetimes = risetimes_t9010
    xlims = (100, 330)
end

fig = Figure()
ax = Axis(fig[1, 1], xlabel = xlabel_str, ylabel = "Counts",  limits = (xlims, (0, nothing)),
    title = "$period, r00$(runs[1])-r0$(runs[end])", titlesize = 16)
for i in eachindex(runs)
    local x = ustrip.(risetimes[i]) .*1e3
    local xmax = quantile(x, 0.8)
    filter!(x -> x <= xmax, x)  
     stephist!(ax, x, bins = 75,   color = colors[i])
    hist!(ax, x, bins = 75, color = (colors[i], 0.3), label = "$(bias_kV[i]) kV",) 
end 
axislegend("Bias voltage", position = :lt)
fig
plt_folder = "$(@__DIR__)/plots/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * "risetimes_$(mode)_2.png"
save(plt_name, fig)


# look only at highest voltages 

fig = Figure()
ax = Axis(fig[1, 1], xlabel = xlabel_str, ylabel = "Counts",  limits = ((130, 190), (0, nothing)),
    title = "$period, r00$(runs[1])-r0$(runs[end])", titlesize = 16)
for i in 5:8#eachindex(runs)
    local x = ustrip.(risetimes[i]) .*1e3
    local xmax = quantile(x, 0.8)
    filter!(x -> x <= xmax, x)  
    stephist!(ax, x, bins = 200,   color = colors[i])
    hist!(ax, x, bins = 200, color = (colors[i], 0.4), label = "$(bias_kV[i]) kV",) 
end 
axislegend("Bias voltage", position = :lt)
fig
plt_folder = "$(@__DIR__)/plots/"
if !isdir(plt_folder)
    mkdir(plt_folder)
end
plt_name = plt_folder * "risetimes_$(mode)_highV.png"
save(plt_name, fig)