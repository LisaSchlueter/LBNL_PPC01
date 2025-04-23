using HDF5
using Distances
using Statistics, LinearAlgebra
using MLJ, MLJClusteringInterface
using CairoMakie, LegendMakie, Makie
using LegendHDF5IO 
using LegendDataManagement
using RadiationDetectorDSP, RadiationDetectorSignals
using LegendDSP
using IntervalSets
using TypedTables: Table as TTable 
using PropDicts
using ColorSchemes
using Printf
using StatsBase 
using Juleanita
using Unitful
include("$(@__DIR__)/utils_ml.jl")

# settings and configs
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)
reprocess = false 

# ml settings: 
preference_quantile = 0.3
damp = 0.5
maxiter = 200
tol = 1.0e-6

# load configs 
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# check if ML pars already exist
ml_file = joinpath(data_path(asic.par[category].rpars.qc_ml[period]), "$run.json" )
pars_ml = if isfile(ml_file) && !reprocess
    @info "Load ML for $category-$period-$run-$channel"
    asic.par[category].rpars.qc_ml[period,run, channel]
else 
    PropDict()
end 

if haskey(pars_ml, :ap)
     @info "Load ML-AP clustering for $category-$period-$run-$channel - You are done!"
     result_ap = pars_ml.ap
else
    @info "Run ML-AP clustering for $category-$period-$run-$channel (run code below!)"
end  

plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :ml_qualitycuts) * "/"

# load waveforms 
data_raw = TTable(read_ldata(asic, DataTier(:raw), filekeys, channel))
wvfs_train_raw = data_raw.waveform[1:1000]
wvfs_train_eventnumber = data_raw.eventnumber[1:1000]

# baseline-shift and normalize waveforms 
wvfs_train = normalize_waveforms(wvfs_train_raw, dsp_config.bl_window)
# sanity plot 5 random waveforms 
for _ in 1:5
    let i = rand(1:length(wvfs_train)) 
        fig = Figure()
        ax = Axis(fig[1, 1], title = "Waveform $i", xlabel =  "Time ($(unit(wvfs_train_raw[i].time[1])))", ylabel = "Signal (a. u.)")
        lines!(ax, ustrip.(wvfs_train_raw[i].time), wvfs_train_raw[i].signal, label = "original")
        lines!(ax, ustrip.(wvfs_train[i].time), wvfs_train[i].signal, label = "normalized")
        axislegend(position = :lt)
        
        _plot_path = joinpath(plt_folder, "waveforms/")
        if !isdir(_plot_path)
            mkpath(_plot_path)
        end
        _pname = "$(_plot_path)AP_waveforms_normalized_$i.png"
        save(_pname, fig)
        @info "Save waveform plot to $(_pname)"
        fig
    end
end 

# run affinity propagation
result_ap, report_ap = trainAP(wvfs_train; 
    preference_quantile = preference_quantile, 
    damp = damp, 
    maxiter = maxiter, 
    tol = tol);
# add waveform eventnumbers to result; 
result_ap = merge(result_ap, (waveforms = merge(result_ap.waveforms, (train_eventnumber = wvfs_train_eventnumber,)),))

# plot all cluster centers: exemplars with ap labels 
plot_col = get(ColorSchemes.tol_muted, range(0.0, 1.0, length=report_ap.ap.ncluster))
fig_ex_ap = plot_APexemplars(report_ap.exemplars.centers, string.(report_ap.exemplars.labels), plot_col)
_plot_path = joinpath(plt_folder, "AP_exemplars/")
if !isdir(_plot_path)
    mkpath(_plot_path)
end
_pname = "$(_plot_path)/AP_exemplars_damp$(report_ap.ap.damp)_qpref$(report_ap.ap.preference_quantile).png"
save(_pname, fig_ex_ap)
@info "Save exemplars plot to $(_pname)"
fig_ex_ap

# save intermediate results from AP 
pars_ml = PropDict(merge(pars_ml, PropDict(:ap => result_ap)))
writelprops(asic.par[category].rpars.qc_ml[period], run, PropDict("$channel" => pars_ml))
@info "Save ML-AP results to pars (intermediate,  labelling not done yet)"

@info "read results"
pars_ml = asic.par[category].rpars.qc_ml[period, run, channel]
result_ap = pars_ml.ap

# rename clusters into LEGEND qc-labels. 1 QC-label can consit of more than one AP cluster
# re-label the all waveforms by hand. several cluster can belong to the the same QC label. 
@info "Renaming clusters into LEGEND qc-labels. THIS HAS TO BE DONE MANUALLY!!!! "
qc_labels = dataprod_config(asic).qc_ml(filekeys[1]).qc_labels
relabel_nt = (normal = [6, 27, 35, 36, 50],
    neg_go = [],
    up_slo = [],
    down_slo = [],
    spike = [],
    x_talk = [],
    slo_ri = [],
    early_tr = [14, 18, 23, 34],
    late_tr = [],
    sat = [],
    soft_pi = [2, 5, 8, 13, 17, 22, 24, 28, 32, 44, 48, 56],
    hard_pi = [1, 3, 4, 7, 9, 10, 11, 12, 15, 16, 19, 20, 21, 25,26,29, 30, 31, 33, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 51, 52, 53, 54, 55, 57],
    bump = [],
    noise = []
)
# sanity check for relabeling 
begin
    @assert length(vcat(values(relabel_nt)...)) == report_ap.ap.ncluster  "Relabelling does not match number of clusters"
    @assert length(unique(vcat(values(relabel_nt)...))) == length(vcat(values(relabel_nt)...)) "Relabelling has duplicates" 
    if length(unique(vcat(values(relabel_nt)...))) !== report_ap.ap.ncluster
        findall(map(x -> x ∉ vcat(values(relabel_nt)...), range(report_ap.ap.ncluster) ) )
    end
end 

function assign_qc_labels(ap_labels, relabel_nt::NamedTuple)
    qc_labels = fill(0, length(ap_labels))
    for (i, ap_label) in enumerate(ap_labels)
        if ap_label in relabel_nt.normal 
            qc_labels[i] = 0
        elseif ap_label in relabel_nt.neg_go
            qc_labels[i] = 1
        elseif ap_label in relabel_nt.up_slo
            qc_labels[i] = 2
        elseif ap_label in relabel_nt.down_slo
            qc_labels[i] = 3
        elseif ap_label in relabel_nt.spike
            qc_labels[i] = 4
        elseif ap_label in relabel_nt.x_talk
            qc_labels[i] = 5
        elseif ap_label in relabel_nt.slo_ri
            qc_labels[i] = 6
        elseif ap_label in relabel_nt.early_tr
            qc_labels[i] = 7
        elseif ap_label in relabel_nt.late_tr
            qc_labels[i] = 8
        elseif ap_label in relabel_nt.sat
            qc_labels[i] = 9
        elseif ap_label in relabel_nt.soft_pi
            qc_labels[i] = 10
        elseif ap_label in relabel_nt.hard_pi
            qc_labels[i] = 11
        elseif ap_label in relabel_nt.bump
            qc_labels[i] = 12
        elseif ap_label in relabel_nt.noise
            qc_labels[i] = 13
        else
            error("Unknown AP label: $ap_label")
        end
    end 
    return qc_labels
end

# apply relabel-logic to each waveform and exemplars
waveform_qc_labels = assign_qc_labels(result_ap.waveforms.ap_labels, relabel_nt)
exemplar_qc_labels = assign_qc_labels(report_ap.exemplars.labels, relabel_nt)
# add qc_labels in result_ap
waveforms = merge(result_ap.waveforms, PropDict(:qc_labels => waveform_qc_labels))
exemplars = merge(result_ap.exemplars, PropDict(:qc_labels => exemplar_qc_labels))
result_ap = PropDict(:ap => result_ap.ap,
                    :waveforms => merge(result_ap.waveforms, PropDict(:qc_labels => waveform_qc_labels)), 
                    :exemplars => merge(result_ap.exemplars, PropDict(:qc_labels => exemplar_qc_labels)),
                    :legend => PropDict(:qc_labels => qc_labels, :relabel_nt => relabel_nt))
report_ap = PropDict(:ap => report_ap.ap,
                    :waveforms => merge(report_ap.waveforms, PropDict(:qc_labels => waveform_qc_labels)), 
                    :exemplars => merge(report_ap.exemplars, PropDict(:qc_labels => exemplar_qc_labels)),
                    :legend => PropDict(:qc_labels => qc_labels, :relabel_nt => relabel_nt))


# plot all cluster centers: exemplars with new qc labels
begin 
    plot_qclabel = report_ap.exemplars.qc_labels
    plot_qclabel_str = map(x -> "$(report_ap.legend.qc_labels[x]) ($x)", report_ap.exemplars.qc_labels) 
    colors = get(ColorSchemes.tol_muted, range(0.0, 1.0, length=14))
    plot_col = colors[report_ap.exemplars.qc_labels .+ 1 ]
    fig_ex_qc = plot_APexemplars(report_ap.exemplars.centers, plot_qclabel_str, plot_col)
    _plot_path = joinpath(plt_folder, "AP_exemplars/")
    if !isdir(_plot_path)
        mkpath(_plot_path)
    end
    _pname = "$(_plot_path)/AP_exemplars_QClabels_damp$(report_ap.ap.damp)_qpref$(report_ap.ap.preference_quantile).png"
    save(_pname, fig_ex_qc)
    @info "Save exemplars with qc labels plot to $(_pname)"
    fig_ex_qc
end

# save results 
pars_ml = PropDict(merge(pars_ml, PropDict(:ap => result_ap)))
writelprops(asic.par[category].rpars.qc_ml[period], run, PropDict("$channel" => pars_ml))
@info "Save ML-AP results to pars"


# EFFICIENCY plot figure 
ap_pars = asic.par[category].rpars.qc_ml[period, run, channel].ap
ap_pars.qc_labels[Symbol.(string.(x_ticklbl))[1]]

_qc_cat =  parse.(Int, string.(keys(countmap(ap_pars.wvfs_qc_labels))))
qc_cat = [ap_pars.qc_labels[_qc_cat[i]] for i in eachindex(_qc_cat)]
x  = collect(1:length(qc_cat))
y  = parse.(Int, string.(values(countmap(ap_pars.wvfs_qc_labels)))) ./ length(ap_pars.wvfs_qc_labels)

fig = Figure()
ax = Axis(fig[1, 1], 
    limits = ((nothing, nothing), (0, 1)),
    xlabel = "QC category", 
    ylabel = "Fraction of training waveforms", 
    title = title = Juleanita.get_plottitle(filekeys[1], _channel2detector(asic, channel), "Affinity Propagation Clustering"))
ax.xticks = x
ax.xtickformat = x -> qc_cat
barplot!(ax, x, y, bar_labels = :y, label = "$(length(ap_pars.wvfs_ap_labels)) waveforms")
axislegend()
fig
save("$(@__DIR__)/plots/efficiency/AP_fractions_damp$(damp)_pref$(round(preference, digits = 2)).png", fig)
fig
