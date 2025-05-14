
# make histogram of some dsp parameters 
using LegendDataManagement
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using Makie, LegendMakie, CairoMakie
using ColorSchemes
using Unitful
using TypedTables
using StatsBase
using KernelDensity
colors = LegendMakie.LegendTheme.palette.color[]

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# settings 
asic = LegendData(:ppc01)
period = DataPeriod(3)
category = :cal 
channel = ChannelId(1)
run = DataRun(2) 
apply_qc = true 

# load dsp pars 
dsp_pars = Table(read_ldata(asic, :jldsp, category, period, run, channel))
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :dsp) * "/"

# load quality cuts: 
if apply_qc == true
    qc_flag = asic.par.rpars.qc[period][run, channel].wvf_keep.all
else
    qc_flag = fill(true, length(dsp_pars))
end

# load timestamps 
timesstamps = read_ldata(asic, :jldsp, category, period, run, channel).timestamp[qc_flag]
time_rel = timesstamps .- timesstamps[1]

# plot time descriptions
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :decay_time)

function plot_TimeEvolution(X::Vector{<:Real}; y_lbl::String = "", qmin = 0.0, qmax = 1.0)
    Xmin, Xmax = quantile(X, qmin), quantile(X, qmax)
    idx = findall(x -> Xmin <= x <= Xmax, X)  
    X = X[idx]
    t = time_rel[idx]

    fig = Figure(size = (550, 1000))
    g = Makie.GridLayout(fig[2,1],)
    # ax1 = Axis(g[1, 1], ylabel = y_lbl, xlabel = "Time (s)",  limits = ((nothing, nothing), (nothing, nothing)), title = "$(_channel2detector(asic, channel))-$period-$run", 
    # titlesize = 16 )
    # barplot!(ax1, t, X, label = "", color = :silver)
    ax = Axis(g[1, 1], xlabel = y_lbl, ylabel = "Counts",  limits = ((nothing, nothing), (0, nothing)), title = "$(_channel2detector(asic, channel))-$period-$run", titlesize = 16)
    stephist!(ax, X, bins = 75, color = :grey, label = "median = $(round(median(X), digits = 1))")
    hist!(ax, X, bins = 75, color = (:grey, 0.3), label = "median = $(round(median(X), digits = 1))") 
    axislegend(ax, merge = true)

    ax2= Axis(g[2, 1], ylabel = y_lbl, xlabel = "Time (s)",  limits = ((0, maximum(t)), (nothing, nothing)) )
    k = KernelDensity.kde((t, X))
    Makie.contour!(ax2, k.x, k.y, k.density, levels = 15, colormap = :plasma, label = "kernel density")
    # hlines!(ax2, [median(X)], color = :black, label = "median")
    axislegend(ax2, merge = true)
    fig 
end

# rise time 
risetime = ustrip.(dsp_pars.t90 .- dsp_pars.t10)
plot_TimeEvolution(1e3 .* risetime[qc_flag]; qmin = 0.05, qmax = 0.65, y_lbl = "Rise time t$(Makie.UnicodeFun.to_subscript(90)) - t$(Makie.UnicodeFun.to_subscript(10)) (ns)")
save(plt_folder * "risetime9010_qc$(apply_qc)_timeevolution.png", fig_rt)

risetime_8020 = ustrip.(dsp_pars.t80 .- dsp_pars.t20)
plot_TimeEvolution(1e3 .* risetime_8020[qc_flag]; qmin = 0.05, qmax = 0.65, y_lbl = "Rise time t$(Makie.UnicodeFun.to_subscript(80)) - t$(Makie.UnicodeFun.to_subscript(20)) (ns)")
save(plt_folder * "risetime8020_qc$(apply_qc)_timeevolution.png", fig_rt)








