env_path = "$(@__DIR__)/" * relpath(split(@__DIR__, "LBNL_PPC01")[1], @__DIR__)  * "/LBNL_PPC01/"
import Pkg; Pkg.activate(env_path)
# to dsp par and their accepted qc cuts 
using LegendDataManagement
using LegendDataManagement: readlprops
using LegendDataManagement.LDMUtils: get_pltfolder
using LegendHDF5IO
using LegendDSP
using RadiationSpectra
using RadiationDetectorDSP
using Plots, Measures
using PropDicts, TypedTables
using Statistics, StatsBase
using Measurements: value as mvalue
using Unitful, Measures, Printf
using IntervalSets
ENV["LEGEND_DATA_CONFIG"] = "/global/cfs/projectdirs/m2676/data/teststands/lbnl/ppc01/config.json"

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/src/apply_qc.jl")
include("$(@__DIR__)/$relPath/utils/utils_plot.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(2)
run = DataRun(1)
channel = ChannelId(1)
category = :cal 

# plot path
pltpath = get_pltfolder(asic, period, run, category, :qc) * "/"
if !ispath(pltpath)
    mkpath(pltpath)
end

# load dsp parameters and qc parameter 
qc_config = asic.metadata.config.qc.qc_config.default
dsp_par = Table(read_ldata(asic, :jldsp, category, period, run, channel))
qc_cuts, _  = apply_qc(dsp_par, qc_config)
qc_pars = Symbol.(keys(qc_config))

# plot histograms of dsp parameters with and without qc cuts
Plots_theme()
p = Vector{Plots.Plot}(undef, length(qc_pars))
for i in eachindex(p)
    par = ustrip.(getproperty(dsp_par, qc_pars[i])[qc_cuts.wvf_keep.finite])
    if qc_pars[i] == :blslope || qc_pars[i] == :tailslope
        filter!(x -> quantile(par, 0.025) < x < quantile(par, 0.975), par)
    end

    h = fit(Histogram, par, nbins = 200)
    bin_centers = collect(h.edges[1]) #.+ step(h.edges[1])/2
    p[i] = histogram(getproperty(dsp_par, qc_pars[i]), 
            fill = true, color = :silver,
            xlabel = string(qc_pars[i]), 
            ylabel = "Counts", 
            bins = bin_centers, 
            label = "all waveforms",
            linecolor = :transparent)

    qc_rejected = round(100 - prod(qc_cuts.qc_surv[qc_pars[i]][k] for k in keys(qc_cuts.qc_surv[qc_pars[i]]))*100, digits = 1)
    histogram!(getproperty(dsp_par, qc_pars[i])[.!qc_cuts.wvf_keep[qc_pars[i]]], 
                fill = true, color = :orange, 
                alpha = 0.8, linecolor = :orange,
                bins = bin_centers,
                label = "quality cut rejected $qc_rejected %"
                ) 
    
end
ptot = plot(p..., layout = (2,2), size = (1000, 700), legend = :topleft, margin = 5mm)
savefig(ptot, pltpath * "dsp_pars_qc_cuts.png")

# plot some NOT cut waveforms 
nplts = 25
wvf = read_ldata(asic, DataTier(:raw), category, period, run, channel)
waveforms_keep = wvf.waveform[findall(qc_cuts.wvf_keep.all)][1:nplts*10]
bl_stats = signalstats.(waveforms_keep, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
waveforms_keep = shift_waveform.(waveforms_keep, -bl_stats.mean)
deconv_flt = InvCRFilter(τ_pz)
wvfs_pz = deconv_flt.(waveforms_keep)
pltpath_kept = pltpath * "wvfs/kept/"
if !ispath(pltpath_kept)
    mkpath(pltpath_kept)
end
for j = 1:nplts
    pw = Vector{Plots.Plot}(undef, 10)
    for i= 1:10
        wvf_idx = 10*(j-1)+i
        @info wvf_idx
        pw[i] = plot(waveforms_keep[wvf_idx], label = false, legend = false)
        plot!(wvfs_pz[wvf_idx], label = false, color = :red2)
    end
    plot(pw..., layout = (5,2), plot_title = "Waveforms: QC passed ",
            size = (600, 900), 
            left_margin = 5mm, 
            margins = 2mm)
    savefig(pltpath_kept * "waveforms_notcut_$j.png")
end

# plot some cut waveforms 
nplts = 25
wvf = read_ldata(asic, DataTier(:raw), category, period, run, channel)
waveforms_cut = wvf.waveform[findall(.!qc_cuts.wvf_keep.all)][1:nplts*10]
bl_stats = signalstats.(waveforms_cut, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
waveforms_cut = shift_waveform.(waveforms_cut, -bl_stats.mean)
deconv_flt = InvCRFilter(τ_pz)
wvfs_pz = deconv_flt.(waveforms_cut)
pltpath_cut = pltpath * "wvfs/cut/"
if !ispath(pltpath_cut)
    mkpath(pltpath_cut)
end
for j = 1:nplts
    pw = Vector{Plots.Plot}(undef, 10)
    for i= 1:10
        wvf_idx = 10*(j-1)+i
        @info wvf_idx
        pw[i] = plot(waveforms_keep[wvf_idx], label = false, legend = false)
        plot!(wvfs_pz[wvf_idx], label = false, color = :red2)
    end
    plot(pw..., layout = (5,2), plot_title = "Waveforms QC cut ",
            size = (600, 900), 
            left_margin = 5mm, 
            margins = 2mm)
    savefig(pltpath_cut * "waveforms_cut_$j.png")
end


