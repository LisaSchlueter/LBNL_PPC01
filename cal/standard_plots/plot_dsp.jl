# make histogram of some dsp parameters 
using LegendDataManagement
using LegendDataManagement.LDMUtils
using LegendHDF5IO
using Makie, LegendMakie, CairoMakie
using ColorSchemes
using Unitful
using TypedTables
using StatsBase
colors = LegendMakie.LegendTheme.palette.color[]

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# settings 
asic = LegendData(:ppc01)
period = DataPeriod(3)
category = :cal 
channel = ChannelId(1)
run = DataRun(31) 
apply_qc = true 

for r in 50:52
    run = DataRun(r)
   

# load dsp pars 
dsp_pars = Table(read_ldata(asic, :jldsp, category, period, run, channel))
filekey = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])[1]
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekey, :dsp) * "/"

# load quality cuts: 
if apply_qc == true
    qc_flag = asic.par[category].rpars.qc[period][run, channel].wvf_keep.all
else
    qc_flag = fill(true, length(dsp_pars))
end


function plot_dsppar(X::Vector{<:Real}; x_lbl::String = "", qmin = 0.0, qmax = 1.0, color = colors[1])
    Xmin, Xmax = quantile(X, qmin), quantile(X, qmax)
    filter!(x -> Xmin <= x <= Xmax, X)  
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = x_lbl, ylabel = "Counts",  limits = ((nothing, nothing), (0, nothing)),
         title = "$(_channel2detector(asic, channel))-$period-$run", titlesize = 16)
    Makie.stephist!(ax, X, bins = 75, color = color, label = "median = $(round(median(X), digits = 1))")
    Makie.hist!(ax, X, bins = 75, color = (color, 0.3), label = "median = $(round(median(X), digits = 1))") 
    axislegend(ax, merge = true)
    fig 
end

# rise time
risetime_9010 = dsp_pars.t90 .- dsp_pars.t10
fig_rt = plot_dsppar(ustrip.(risetime_9010[qc_flag])*1e3; qmax = 0.8, qmin = 0.01, color = colors[1], x_lbl = "Rise time t$(Makie.UnicodeFun.to_subscript(90)) - t$(Makie.UnicodeFun.to_subscript(10)) (ns)")
save(plt_folder * "risetime9010_qc$apply_qc.png", fig_rt)

risetime_8020 = dsp_pars.t80 .- dsp_pars.t20
fig_rt = plot_dsppar(ustrip.(risetime_8020[qc_flag])*1e3; qmax = 0.8, qmin = 0.01, color = colors[1], x_lbl = "Rise time t$(Makie.UnicodeFun.to_subscript(80)) - t$(Makie.UnicodeFun.to_subscript(20)) (ns)")
save(plt_folder * "risetime8020_qc$apply_qc.png", fig_rt)

# baseline slope 
blslope = dsp_pars.blslope
fig_bl = plot_dsppar(1e2*ustrip.(blslope)[qc_flag]; qmax = 0.99, qmin = 0.01, color = colors[1], x_lbl = "Baseline slope  (10$(Makie.UnicodeFun.to_superscript(-2)) µs$(Makie.UnicodeFun.to_superscript(-1)))")
save(plt_folder * "blslope_qc$apply_qc.png", fig_bl)

# t0
t0 = dsp_pars.t0
fig_t0 = plot_dsppar(ustrip.(t0)[qc_flag]; qmax = 0.999, qmin = 0.08, color = colors[1], x_lbl = "t$(Makie.UnicodeFun.to_subscript(0)) (µs)")
save(plt_folder * "t0_qc$apply_qc.png", fig_t0)

fig_t02 = plot_dsppar(ustrip.(t0); qmax = 0.999, qmin = 0.0, color = colors[1], x_lbl = "t$(Makie.UnicodeFun.to_subscript(0)) (µs)")
save(plt_folder * "t0_qcfalse.png", fig_t02)

# qdrift
qdrift = dsp_pars.qdrift
fig_qdrift = plot_dsppar(ustrip.(qdrift)[qc_flag];  qmin = 0.0,  qmax = 0.999, color = colors[1], x_lbl = "qdrift (a.u.)")
save(plt_folder * "qdrift_qc$(apply_qc)_all.png", fig_qdrift)

end




