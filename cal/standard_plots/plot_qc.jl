# to dsp par and their accepted qc cuts 
using LegendDataManagement
using LegendDataManagement.LDMUtils: get_pltfolder
using LegendHDF5IO
using CairoMakie, LegendMakie, Makie
using Measures
using PropDicts, TypedTables
using StatsBase
using Measurements: value as mvalue
using LegendDSP, RadiationDetectorDSP, IntervalSets
using Unitful, Measures, Printf

# include relevant functions 
relPath = relpath(split(@__DIR__, "hpge-ana")[1], @__DIR__) * "/hpge-ana/"
include("$(@__DIR__)/$relPath/utils/utils_aux.jl")

# inputs
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(50)
channel = ChannelId(1)
category = :cal 
det = _channel2detector(asic, channel)

# load config 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# plot path
pltpath = get_pltfolder(asic, period, run, category, :qc) * "/"
if !ispath(pltpath)
    mkpath(pltpath)
end

# load dsp parameters and qc parameter 
dsp_par = Table(read_ldata(asic, :jldsp, category, period, run, channel))

# load quality cuts 
qc_cuts = asic.par[category].rpars.qc[period][run][channel]
qc_pars = filter!(x -> (x != :all) && (x != :finite), Symbol.(keys(qc_cuts.qc_surv)))

# plot histograms of dsp parameters with and without qc cuts
begin 
    fig = Figure(size = (750, 550))
    grid = fig[1,1] = GridLayout()
    for i in eachindex(qc_pars)
        row = div(i - 1, 2) + 1
        col = mod(i - 1, 2) + 1
        par = ustrip.(getproperty(dsp_par, qc_pars[i])[qc_cuts.wvf_keep.finite])
        ustr = unit(getproperty(dsp_par, qc_pars[i])[1])
        if ustr == NoUnits ustr = "a.u." end
        if qc_pars[i] == :blslope || qc_pars[i] == :tailslope
            # filter!(x -> quantile(par, 0.025) < x < quantile(par, 0.975), par)
            par = par[quantile(par, 0.025) .< par .< quantile(par, 0.975)]
            exp_i = floor(Int64, log10(quantile(par, 0.975)) )
            par = par .* 10^-exp_i
            xl = string(qc_pars[i]) * " (10$(Makie.UnicodeFun.to_superscript(exp_i)) $ustr)"#L" (10^$(round(Int, exp_i)) $ustr)"
        elseif qc_pars[i] == :e_trap
            if period >= DataPeriod(3)
                ustr = "ADC"
                # filter!(x -> x < quantile(par, 0.975), par)
                par = par[par .< quantile(par, 0.975)]
            end
            xl = string(qc_pars[i]) * " ($ustr)"
            exp_i = 0.0
        else
            xl = string(qc_pars[i]) * " ($ustr)"
            exp_i = 0.0
        end
        h = fit(Histogram, par, nbins = 200)
        bin_centers = collect(h.edges[1]) #.+ step(h.edges[1])/2
        ax = Axis(grid[row, col], xlabel = xl, limits = ((nothing, nothing), (0, nothing)))
        hist!(ax, par, bins = bin_centers, label = "all")

        qc_frac_rejected = round(100 - prod(qc_cuts.qc_surv[qc_pars[i]][k] for k in keys(qc_cuts.qc_surv[qc_pars[i]]))*100, digits = 1)
        hist!(ax, 10^-exp_i.*ustrip.(getproperty(dsp_par, qc_pars[i])[.!qc_cuts.wvf_keep[qc_pars[i]]]), 
                color = :orange, 
                bins = bin_centers, 
                label = "rejected")#rejected $qc_frac_rejected %" )
        axislegend(ax, position = :lt)
    end
    t = LegendDataManagement.LDMUtils.get_plottitle(filekeys[1], det, "Quality cut parameters distribution")
    Label(grid[1, 1:2, Top()], t, valign = :bottom,
        font = :bold,
        padding = (0, 0, 5, 0))
    fig
end 
pname = pltpath * _get_pltfilename(asic, filekeys[1], channel, :qc_dist)
save(pname, fig)


# PLOTS waveforms cut and not cut 
function _plot_qc_waveform_grid(waveforms, waveforms_pz; ncol = 2, nwaveforms = 10, nplots = 1, save_plot = false, mode::Symbol = :pass)
    if mode == :pass
        save_str = "waveform_pass"  
        title_str = LegendDataManagement.LDMUtils.get_plottitle(filekeys[1], det, "Waveforms QC pass (not cut)")  
        pltpath_local = pltpath * "wvfs/pass/"
    elseif mode == :cut
        save_str = "waveform_cut" 
        title_str = LegendDataManagement.LDMUtils.get_plottitle(filekeys[1], det, "Waveforms QC cut")  
        pltpath_local = pltpath * "wvfs/cut/"
    end 
    if !ispath(pltpath_local)
        mkpath(pltpath_local)
    end

    colors = LegendMakie.LegendTheme.palette.color[][1:2]
    figs = Vector{Figure}(undef, nplots)
    for j = 1:nplots
        nrow = ceil(Int, nwaveforms/ncol)
        figs[j] = Figure(size = (425.0*ncol, nrow*220.0))
        grid = figs[j][1,1] = GridLayout()
    
        for i= 1:nwaveforms
            row = div(i - 1, ncol) + 1
            col = mod(i - 1, ncol) + 1
            ax = Axis(grid[row, col], xlabel = "Time (µs)", ylabel = "Signal (V)")
            wvf_idx = 10*(j-1)+i
            @debug wvf_idx
            lines!(ax, ustrip.(waveforms[wvf_idx].time), ustrip.(waveforms[wvf_idx].signal), linewidth = 2.0, color = colors[1])
            lines!(ax, ustrip.(waveforms_pz[wvf_idx].time), ustrip.(waveforms_pz[wvf_idx].signal), linewidth = 2.0, color = colors[2])
            text!(ax, 0.95, 0.05; text = "$(wvf_idx)", space = :relative, align = (:right, :bottom), color = colors[1])
        end
    
        Label(grid[1, 1:2, Top()], title_str, valign = :bottom, font = :bold, padding = (0, 0, 0, 0), )
        figs[j] 
        if save_plot == true 
            pname = pltpath_local * _get_pltfilename(asic, filekeys[1], channel, Symbol("$(save_str)_$j"))
            save(pname, figs[j])
        end 
    end 
    return figs
end 

# plot some NOT cut waveforms 
nplts = 25
waveforms_keep = read_ldata(asic, DataTier(:raw), category, period, run, channel).waveform[findall(qc_cuts.wvf_keep.all)][1:nplts*10]
bl_stats = signalstats.(waveforms_keep, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
waveforms_keep = shift_waveform.(waveforms_keep, -bl_stats.mean)
deconv_flt = InvCRFilter(mvalue(asic.par[category].rpars.pz[period, run, channel].τ))
waveforms_keep_pz = deconv_flt.(waveforms_keep)
figs = _plot_qc_waveform_grid(waveforms_keep, waveforms_keep_pz; ncol = 2, nwaveforms = 10, nplots = 10, save_plot = true, mode = :pass)
figs[10]

# plot some cut waveforms 
waveforms_cut = waveforms_keep = read_ldata(asic, DataTier(:raw), category, period, run, channel).waveform[findall(.!qc_cuts.wvf_keep.all)][1:nplts*10]
bl_stats = signalstats.(waveforms_cut, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
waveforms_cut = shift_waveform.(waveforms_cut, -bl_stats.mean)
deconv_flt = InvCRFilter(mvalue(asic.par[category].rpars.pz[period, run, channel].τ))
waveforms_cut_pz = deconv_flt.(waveforms_cut)
figs_cut = _plot_qc_waveform_grid(waveforms_cut, waveforms_cut_pz; ncol = 2, nwaveforms = 10, nplots = 10, save_plot = true, mode = :cut)
figs_cut[1]


