using HDF5
using Distances
using Statistics, LinearAlgebra
using MLJ, MLJClusteringInterface
using LIBSVM
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
using Unitful 
using Juleanita

# settings and configs
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)
reprocess = false 

# check if qc_ml file already exists.
ml_file = joinpath(data_path(asic.par[category].rpars.qc_ml[period]), "$run.json" )
pars_ml = if isfile(ml_file)
    @info "Load ML for $category-$period-$run-$channel"
    asic.par[category].rpars.qc_ml[period,run, channel]
else 
    PropDict()
end 

# load AffinityPropagation results 
if !haskey(pars_ml, :ap)
    error("Run ML-AP clustering for $category-$period-$run-$channel do not exist. Run AP processor first")
end 

# START of ML-SVM
# load waveforms that were also used during AP training
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
eventnumber = read_ldata((:eventnumber), asic, DataTier(:raw), filekeys, channel)
data_raw = TTable(read_ldata(asic, DataTier(:raw), filekeys, channel))[findall(map(x -> x in pars_ml.ap.wvfs_train_eventnumber, eventnumber))]
wvfs_train = data_raw.waveform

# 2. baseline-shift and normalize waveforms 
bl_stats = signalstats.(wvfs_train, leftendpoint(dsp_config.bl_window), rightendpoint(dsp_config.bl_window))
wvfs_shift = shift_waveform.(wvfs_train, -bl_stats.mean)
wvf_absmax = maximum.(map(x-> abs.(x), wvfs_shift.signal))
wfs = multiply_waveform.(wvfs_shift, 1 ./ wvf_absmax)

# 3. create discrete wavelet transform  (DWT)
function dwt(waveforms::ArrayOfRDWaveforms; nlevels::Int = 2)
    # create Haar filter 
    haar_flt = HaarAveragingFilter(2)  

    # wvfs_flt = waveforms .|> haar_flt .|> haar_flt
    # Apply Haar filter nlevels times
    wvfs_flt = copy(waveforms)
    for _ in 1:nlevels
        wvfs_flt = haar_flt.(wvfs_flt)
    end

    # normalize with max of absolute extrema values
    norm_fact = map(x -> max(abs(first(x)), abs(last(x))), extrema.(wvfs_flt.signal))
    replace!(norm_fact, 0.0 => one(first(norm_fact)))
    wvfs_flt = multiply_waveform.(wvfs_flt, 1 ./ norm_fact)
end 
haar_levels = 6
wvfs_dwt = dwt(wfs; nlevels = haar_levels)
let fig = Figure()
    i = rand(1:length(wvfs_dwt))
    ax = Axis(fig[1, 1], title = "Discrete wavelet transform ($i): \ndownsampled + noise reduced", xlabel = "Time (µs)", ylabel = "Norm. Signal", limits = ((nothing, nothing), (nothing, 1.35)))
    lines!(ax, ustrip.(wfs[i].time), wfs[i].signal, label = "Original ($(round(Int, length(wfs[i].time)/1e3))k samples)")
    lines!(ax, ustrip.(wvfs_dwt[i].time), wvfs_dwt[i].signal, label = "DWT level $(haar_levels) ($(round(Int, length(wvfs_dwt[i].time)/1e3))k samples)", linestyle = :dash)
    axislegend(position = :lt)
    save("$(@__DIR__)/plots/waveforms/Waveform_DWT_$i.png", fig)
	fig
end

ntrain = round(Int, length(wvfs_dwt)*0.8)
labels_train = pars_ml.ap.wvfs_qc_labels[1:ntrain]
labels_test = pars_ml.ap.wvfs_qc_labels[ntrain+1:end]
X_train = hcat(wvfs_dwt.signal...)[:, 1:ntrain]
X_test = hcat(wvfs_dwt.signal...)[:, ntrain+1:end]
@assert size(X_train, 2) == length(labels_train) "Number of samples in X must match the length of label"

# Train an SVM model with a radial basis function (RBF) kernel
model = svmtrain(X_train, labels_train; kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=0.5)

# Predict labels for the training data
pred_labels_train = svmpredict(model, X_train)
accuracy_train = sum(pred_labels_train[1] .== labels_train) / length(labels_train)
println("Accuracy train: $accuracy_train")

# Predict labels for the test data
pred_labels_test = svmpredict(model, X_test)
accuracy_test = sum(pred_labels_test[1] .== labels_test) / length(labels_test)
println("Accuracy test: $accuracy_test")

plot_SVM_QCeff(pred_labels_train[1], pars_ml.ap; mode = "train", accuracy = accuracy_train)
plot_SVM_QCeff(pred_labels_test[1], pars_ml.ap; mode = "test",  accuracy = accuracy_test)

function plot_SVM_QCeff(label_pred::Vector, pars_mlap::PropDict; mode = "", accuracy = NaN)
    _qc_cat =  parse.(Int, string.(keys(countmap(label_pred))))
    qc_cat = [pars_mlap.qc_labels[_qc_cat[i]] for i in eachindex(_qc_cat)]
    x  = collect(1:length(qc_cat))
    y  = parse.(Int, string.(values(countmap(label_pred)))) ./ length(label_pred)
    label_plot = if !isfinite(accuracy)
       "$(length(label_pred)) waveforms ($mode)"
    else 
       "$(length(label_pred)) waveforms ($mode) \naccuracy: $(round(accuracy, digits = 3))"
    end

    fig = Figure()
    ax = Axis(fig[1, 1], 
        limits = ((nothing, nothing), (0, 1)),
        xlabel = "QC category", 
        ylabel = "Fraction of training waveforms", 
        title = title = Juleanita.get_plottitle(filekeys[1], _channel2detector(asic, channel), "AP-SVM Quality cuts"),
        titlesize = 16)
    ax.xticks = x
    ax.xtickformat = x -> qc_cat
    barplot!(ax, x, y, bar_labels = :y, label = label_plot)
    axislegend()
    fig
    save("$(@__DIR__)/plots/efficiency/AP_fractions_damp$(damp)_pref$(round(preference, digits = 2))_$mode.png", fig)
    fig
end 


## that's what's in dataflow  SVM: 
# find optimal hyper parameters 
# # SVM with optimized hyper parameters 
# model = svmtrain(wfs_dwt, qc_labels, 
#                 cost=hyperparams.cost, 
#                 kernel=LIBSVM.Kernel.RadialBasis, 
#                 gamma=hyperparams.gamma,
#                 weights = Dict(parse.(Float64, string.(keys(hyperparams.weights))) .=> values(hyperparams.weights)),
#                 probability=hyperparams.probability,
#                 cachesize=Float64(hyperparams.cache_size),
#                 coef0=hyperparams.coef0,
#                 shrinking=hyperparams.shrinking,
#                 tolerance=hyperparams.tolerance
#                 )

