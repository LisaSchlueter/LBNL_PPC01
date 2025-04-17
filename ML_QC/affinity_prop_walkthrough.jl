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

# settings and configs
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)
reprocess = false 

# check if decaytime pars already exist
ml_file = joinpath(data_path(asic.par[category].rpars.qc_ml[period]), "$run.json" )
pars_ml = if isfile(ml_file) && !reprocess
    @info "Load ML for $category-$period-$run-$channel"
    asic.par[category].rpars.qc_ml[period,run, channel]
else 
    PropDict()
end 

if haskey(pars_ml, :ap)
     @info "Load ML-AP clustering for $category-$period-$run-$channel"
     result_ap = pars_ml.ap
else
    @info "Run ML-AP clustering for $category-$period-$run-$channel (run code below!)"
end  

# START of ML-AP clustering
# load configs
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# load waveforms 
data_raw = TTable(read_ldata(asic, DataTier(:raw), filekeys, channel))
wvfs_train = data_raw.waveform[1:1000]
wvfs_train_eventnumber = data_raw.eventnumber[1:1000]

# baseline-shift and normalize waveforms 
bl_stats = signalstats.(wvfs_train, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs_shift = shift_waveform.(wvfs_train, -bl_stats.mean)
wvf_absmax = maximum.(map(x-> abs.(x), wvfs_shift.signal))
wfs = multiply_waveform.(wvfs_shift, 1 ./ wvf_absmax)
# sanity plot 
let i = rand(1:length(wfs)) 
    fig = Figure()
    ax = Axis(fig[1, 1], title = "Waveform $i")
    lines!(ax, wvfs_train[i].time, wvfs_train[i].signal, label = "original")
    lines!(ax, wfs[i].time, wfs[i].signal, label = "normalized")
    axislegend(position = :lt)
    save("$(@__DIR__)/plots/waveforms/AP_waveforms_normalized_$i.png", fig)
    fig
end

# transform waveforms to a matrix of size (n_waveforms, n_samples) for ML clustering
wvfs_matrix = transpose(hcat(wfs.signal...))
X = MLJ.table(wvfs_matrix)

# calcualte similary matrix to get an idea what the preference could look like: 
S =  - pairwise(Cityblock(), wvfs_matrix, dims = 1)
Su = S[triu(trues(size(S)))]
preference = quantile(Su, 0.3)
f = Figure()
ax = Axis(f[1, 1])
hist!(ax, Su , bins = 100)
f

# prepare model
damp = 0.5 
model = AffinityPropagation(
            damp = damp, 
            maxiter = 200, 
            tol = 1.0e-6, 
            preference = preference, 
            metric = Cityblock())

# apply model to data X
begin
    _machine = machine(model)
    waveform_ap_labels = collect(predict(_machine, X)) # these are the labels of each waveform (assigned to a cluster)
end        

# evaluate the model 
_report = report(_machine)

# look whats inside report
@info "Report keys: $(keys(_report))"
@info "model converged =  $(_report.converged)"
ap_iterations = _report.iterations
ap_ncluster = length(_report.exemplars) # waveform index which is cluster exemplar
ap_centers = _report.centers # copy of waveforms which are cluster exemplars
exemplar_label = _report.cluster_labels # labels of clusters (default just 1:nclusters)
all(wvfs_matrix[_report.exemplars[1], :] .== _report.centers[:, 1]) # confirm that this is actually the same
@info "$(length(_report.cluster_labels)) clusters found: $(_report.cluster_labels[1]) - $(_report.cluster_labels[end])"

# plot all cluster centers: exemplars with ap labels 
ncol = round(Int,sqrt(length(_report.cluster_labels)))
nrow = ceil(Int,length(_report.cluster_labels)/ncol)
let fig = Figure(size=(ncol*100,nrow*110), figure_padding = 5)
	for i in 1:length(_report.cluster_labels)
        _row = div(i-1,ncol)+1
        _col = mod(i-1,ncol)+1
        _ax =  Axis(fig[_row,_col], xticklabelsize = 5, yticklabelsize = 5)
		lines!(_ax, _report.centers[:,i], linewidth = 1.5)
        Label(fig[_row, _col], "$i", fontsize = 10, padding = (10, 10),  tellwidth = false)
        hidedecorations!(_ax)
	end
    Label(fig[0, :], "Cluster Center Exemplar Waveforms", fontsize = 20, tellwidth = false)
	rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    save("$(@__DIR__)/plots/exemplars/AP_exemplars_damp$(damp)_pref$(round(preference, digits = 2)).png", fig)
    fig
end

# # ### do an mini-optimization: (this has to be done in a better way; also optimizde damp over grid )
# q = [0.25, 0.5, 0.75]
# preferences = quantile(Su, q)
# reports = Vector{Any}(undef, length(preferences))
# nclusters = Vector{Int}(undef, length(preferences))
# for i in eachindex(preferences)
#     p = preferences[i]
#     model = AffinityPropagation(damp = damp,maxiter = 200, tol = 1.0e-6, preference = p,  metric = Cityblock())
#     # apply model to data X
#     begin
#         _machine = machine(model)
#         y_hat = predict(_machine, X)
#     end        
#     # evaluate the model 
#     reports[i] = report(_machine)
#     nclusters[i] = length(reports[i].cluster_labels)
#     @info "Preference: $p, converged =  $(reports[i].converged)"
#     @info "$(length(reports[i].cluster_labels)) clusters found"
# end 
# f = Figure()
# ax = Axis(f[1, 1], xlabel = "Preference (quantile)", ylabel = "Number of clusters", limits = ((nothing, nothing), (0, nothing)))
# barplot!(ax, q, nclusters)
# f
# save("$(@__DIR__)/plots/optimization/AP_optimize_perference_damp$damp.png", f)

# rename clusters into LEGEND qc-labels. 1 QC-label can consit of more than one AP cluster
# re-label the all waveforms by hand. several cluster can belong to the the same QC label. 
qc_labels = Dict(
    "0" => "normal" ,
    "1" => "neg. going",
    "2" => "up slope",
    "3" => "down slope",
    "4" => "spike",
    "5" => "x-talk",
    "6" => "slope rising",
    "7" => "early trigger",
    "8" => "late trigger",
    "9" => "saturated",
    "10" => "soft_pileup",
    "11" => "hard pileup",
    "12" => "bump",
    "13" => "noise"
)

pd_relabel = PropDict(
    :normal => [6, 27, 35, 36, 50],
    :neg_go => [],
    :up_slo => [],
    :down_slo => [],
    :spike => [],
    :x_talk => [],
    :slo_ri => [],
    :early_tr => [14, 18, 23, 34],
    :late_tr => [],
    :sat => [],
    :soft_pi => [2, 5, 8, 13, 17, 22, 24, 28, 32, 44, 48, 56],
    :hard_pi => [1, 3, 4, 7, 9, 10, 11, 12, 15, 16, 19, 20, 21, 25,26,29, 30, 31, 33, 37, 38, 39, 40, 41, 42, 43, 45, 46, 47, 49, 51, 52, 53, 54, 55, 57],
    :bump => [],
    :noise => []
)

# sanity check for relabeling 
begin
    @printf("Correct number of labels? %s \n", length(vcat(values(pd_relabel)...)) == ap_ncluster) # check if all fields are present
    @printf("All labels unique? %s", length(unique(vcat(values(pd_relabel)...))) == ap_ncluster) # check if all fields are present
    if length(unique(vcat(values(pd_relabel)...))) !== ap_ncluster
        findall(map(x -> x ∉ vcat(values(pd_relabel)...), range(ap_ncluster) ) )
    end
end 

waveform_qc_labels = fill(0, length(waveform_ap_labels))

for (i, ap_label) in enumerate(waveform_ap_labels)
    if ap_label in pd_relabel.normal 
        waveform_qc_labels[i] = 0
    elseif ap_label in pd_relabel.neg_go
        waveform_qc_labels[i] = 1
    elseif ap_label in pd_relabel.up_slo
        waveform_qc_labels[i] = 2
    elseif ap_label in pd_relabel.down_slo
        waveform_qc_labels[i] = 3
    elseif ap_label in pd_relabel.spike
        waveform_qc_labels[i] = 4
    elseif ap_label in pd_relabel.x_talk
        waveform_qc_labels[i] = 5
    elseif ap_label in pd_relabel.slo_ri
        waveform_qc_labels[i] = 6
    elseif ap_label in pd_relabel.early_tr
        waveform_qc_labels[i] = 7
    elseif ap_label in pd_relabel.late_tr
        waveform_qc_labels[i] = 8
    elseif ap_label in pd_relabel.sat
        waveform_qc_labels[i] = 9
    elseif ap_label in pd_relabel.soft_pi
        waveform_qc_labels[i] = 10
    elseif ap_label in pd_relabel.hard_pi
        waveform_qc_labels[i] = 11
    elseif ap_label in pd_relabel.bump
        waveform_qc_labels[i] = 12
    elseif ap_label in pd_relabel.noise
        waveform_qc_labels[i] = 13
    else
        error("Unknown AP label: $ap_label")
    end
end 

colors = get(ColorSchemes.tol_muted, range(0.0, 1.0, length=14))
# plot all cluster centers: exemplars with new qc labels 
ncol = round(Int,sqrt(length(_report.cluster_labels)))
nrow = ceil(Int,length(_report.cluster_labels)/ncol)
let fig = Figure(size=(ncol*100,nrow*110), figure_padding = 5)
	for i in 1:length(_report.cluster_labels)
        wf_idx = _report.exemplars[i]
        wf = wfs[wf_idx]
        qc_label = waveform_qc_labels[wf_idx]
        _row = div(i-1,ncol)+1
        _col = mod(i-1,ncol)+1
        _ax =  Axis(fig[_row,_col], xticklabelsize = 5, yticklabelsize = 5)
		lines!(_ax, wf.time, wf.signal, linewidth = 1.5, color = colors[qc_label+1])
        Label(fig[_row, _col], "$(qc_labels["$(qc_label)"]) ($qc_label)", fontsize = 10, padding = (10, 10),  tellwidth = false)
        hidedecorations!(_ax)
	end
    Label(fig[0, :], "Cluster Center Exemplar Waveforms with QC labels", fontsize = 20, tellwidth = false)
	rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    save("$(@__DIR__)/plots/exemplars/AP_exemplars_qclabel_damp$(damp)_pref$(round(preference, digits = 2)).png", fig)
    fig
end

result_ap = PropDict(
    :qc_labels => qc_labels,
    :ap_labels => exemplar_label,
    :ap_qc_relabels => pd_relabel,
    :wvfs_train_eventnumber => wvfs_train_eventnumber,
    :wvfs_ap_labels => waveform_ap_labels,
    :wvfs_qc_labels => waveform_qc_labels,
    :preference => preference,
    :damp => damp,
    :iterations => ap_iterations,
)
pars_ml = PropDict(merge(pars_ml, PropDict(:ap => result_ap)))
writelprops(asic.par[category].rpars.qc_ml[period], run, PropDict("$channel" => pars_ml))
@info "Save ML-AP results to pars"

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
