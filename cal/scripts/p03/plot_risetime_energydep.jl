
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

# plot time descriptions
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :rise_time) * "/"
e_unit = "keV"
function plot_kde(X::Vector{<:Real}, Y::Vector{<:Real};  y_lbl::String = "", qmin = 0.0, qmax = 1.0)
    xt = 400:300:1500
    Ymin, Ymax =  quantile(Y, qmin), quantile(Y, qmax)
    idx = findall(y -> Ymin <= y <= Ymax, Y)  
    X = X[idx]
    Y = Y[idx]
    xlims = (300, 1500)

    fig = Figure()

    g = Makie.GridLayout(fig[1,1])

    ax = Makie.Axis(g[1,1], limits = (xlims...,0,nothing), ylabel = "")
    # Makie.stephist!(ax, X, color = :darkgrey, bins = 400:10:1500)
    Makie.hist!(ax, X, color = (:darkgrey, 0.5), bins = 400:10:1500)

    ax2 = Makie.Axis(g[2,1], xlabel = "Energy ($e_unit)", ylabel = y_lbl, xticks = xt, limits = (xlims..., 0.95*Ymin, 1.05*Ymax))#, xticks = 2400:10:2630, yticks = 0:0.2:1, 
    k = KernelDensity.kde((X, Y))
    Makie.contourf!(ax2, k.x, k.y, k.density, levels = 15, colormap = :binary)
    Makie.contour!(ax2, k.x, k.y, k.density, levels = 15 - 1, colormap = :plasma)
    Makie.lines!(ax2, [0], [0], label = "", color = :darkgrey)


    ax3 = Makie.Axis(g[2,2], limits = (0,nothing, 0.95*Ymin, 1.05*Ymax), xlabel = "")
    Makie.hist!(ax3, Y, color = (:darkgrey, 0.5), bins = 100, direction = :x)
    ax3.xticks = Makie.WilkinsonTicks(3, k_min = 3, k_max=4)

    #  # Formatting
     Makie.linkxaxes!(ax,ax2)
     Makie.hidexdecorations!(ax)
     Makie.rowgap!(g, 0)
     Makie.rowsize!(g, 1, Makie.Auto(0.5))
     Makie.linkyaxes!(ax2,ax3)
     Makie.hideydecorations!(ax3)
     Makie.colgap!(g, 0)
     Makie.colsize!(g, 2, Makie.Auto(0.5))
     xspace = maximum(Makie.tight_xticklabel_spacing!, (ax2, ax3))
     ax2.xticklabelspace = xspace
     ax3.xticklabelspace = xspace
     yspace = maximum(Makie.tight_yticklabel_spacing!, (ax, ax2))
     ax.yticklabelspace = yspace
     ax2.yticklabelspace = yspace
    fig
end

# get cal. energy and rise times 
cal_func_str = asic.par.rpars.ecal[period, run, channel].e_trap.cal.func
cal_func = ljl_propfunc(cal_func_str)
e_cal = ustrip.(cal_func.(dsp_pars))
risetime = ustrip.(dsp_pars.t90 .- dsp_pars.t10)

# plot 
fig_kde = plot_kde(e_cal[qc_flag], 1e3 .* risetime[qc_flag]; qmin = 0.05, qmax = 0.62, y_lbl = "Rise time t$(Makie.UnicodeFun.to_subscript(90)) - t$(Makie.UnicodeFun.to_subscript(10)) (ns)")
save(plt_folder * "risetime_EnergyDep_qc$(apply_qc).png", fig_kde)


# # even simpler plot without kde 
# fig = Figure()
# ax = Axis(fig[1, 1], xlabel = "Energy (keV)", ylabel = "Rise time t$(Makie.UnicodeFun.to_subscript(90)) - t$(Makie.UnicodeFun.to_subscript(10)) (ns)",  limits = ((300, 1500), (100, 400)), title = "$(_channel2detector(asic, channel))-$period-$run", titlesize = 16)
# scatter!(ax, e_cal[qc_flag], 1e3 .* risetime[qc_flag])
# fig