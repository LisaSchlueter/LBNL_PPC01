# # Goal: find optimal hyper parameters. 
# using HDF5
# using Distances
# using Statistics, LinearAlgebra
# using MLJ, MLJClusteringInterface
# using CairoMakie, LegendMakie, Makie
# using LegendHDF5IO 
# using LegendDataManagement
# using RadiationDetectorDSP, RadiationDetectorSignals
# using LegendDSP
# using IntervalSets
# using TypedTables: Table as TTable 
# using PropDicts
# using ColorSchemes
# using Printf
# using StatsBase 
# using Juleanita

# # settings and configs
# asic = LegendData(:ppc01)
# period = DataPeriod(1)
# run = DataRun(1)
# channel = ChannelId(1)
# category = DataCategory(:cal)
# reprocess = false 

# # check if decaytime pars already exist
# ml_file = joinpath(data_path(asic.par[category].rpars.qc_ml[period]), "$run.json" )
# pars_ml = if isfile(ml_file) && !reprocess
#     @info "Load ML for $category-$period-$run-$channel"
#     asic.par[category].rpars.qc_ml[period,run, channel]
# else 
#     PropDict()
# end 

# if haskey(pars_ml, :svm)
#     haskey(pars_ml.svm, hyperpars)
#     @info "Load ML-SVM hyperpars for $category-$period-$run-$channel"
#     hyperpars = pars_ml.svm.hyperpars
# else
#     @info "ML-SVM hyperpars for $category-$period-$run-$channel not found. (run code below)"
# end 

# """
# **cost (C)**:
# Description: The regularization parameter that controls the trade-off between achieving a low error on the training data and minimizing model complexity.
# Effect:
# A small C makes the SVM more tolerant to misclassifications (simpler model, higher bias).
# A large C tries to classify all training points correctly (complex model, lower bias but higher variance).
# Example Values: 0.1, 1, 10, 100
# """
# cost = 1.0  # Default value, adjust based on cross-validation

# """
# 3. gamma:
# Description: Controls the influence of a single training example in the RBF kernel. It defines how far the influence of a single training example reaches.
# Effect:
# A small gamma means a large influence (smooth decision boundary).
# A large gamma means a small influence (complex decision boundary).
# Example Values: 0.001, 0.01, 0.1, 1
# """
# gamma = 0.1 

# SVM_hyperpars = PropDict(
#     cost = cost,
#     gamma = gamma,
#     weights = Dict("0" => 1.0, "1" => 1.0),
#     probability = true,
#     cache_size = 1000,
#     coef0 = 0.0,
#     shrinking = true,
#     tolerance = 0.001
# )

using LegendHDF5IO 
using LegendDataManagement
using RadiationDetectorDSP, RadiationDetectorSignals
using LegendDSP
using IntervalSets
using TypedTables: Table as TTable 
using PropDicts
using ColorSchemes
using Printf
using Random
using Statistics
using LIBSVM
using StatsBase
using Distributions
using MLJ
using MLJBase
using MLJIteration
using MLJModels
using MLJLIBSVMInterface
# settings and configs
asic = LegendData(:ppc01)
period = DataPeriod(1)
run = DataRun(1)
channel = ChannelId(1)
category = DataCategory(:cal)
reprocess = true 

# check if decaytime pars already exist
ml_file = joinpath(data_path(asic.par[category].rpars.qc_ml[period]), "$run.json" )
pars_ml = if isfile(ml_file) 
    @info "Load ML for $category-$period-$run-$channel"
    asic.par[category].rpars.qc_ml[period,run, channel]
else 
    PropDict()
end 

if haskey(pars_ml, :svm)
    if haskey(pars_ml.svm, hyperpars)
        # @info "Load ML-SVM hyperpars for $category-$period-$run-$channel"
        # hyperpars = pars_ml.svm.hyperpars
        @info "you are done"
    else
        @info "ML-SVM hyperpars for $category-$period-$run-$channel not found. (run code below)"
    end
else
    @info "ML-SVM hyperpars for $category-$period-$run-$channel not found. (run code below)"
end 



"""
START CHATGPT + adapt
"""


# Load training data
# load waveforms that were alsoed used during AP training
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)
_eventnumber_all = read_ldata((:eventnumber),asic, DataTier(:raw), filekeys, channel)
data_raw = TTable(read_ldata(asic, DataTier(:raw), filekeys, channel))[findall(map(x -> x in pars_ml.ap.wvfs_train_eventnumber, _eventnumber_all))]
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

# load labels and create matrix for SVM input  (n samples, n features)
labels = pars_ml.ap.wvfs_qc_labels
wvfs_dwt = dwt(wfs; nlevels = haar_levels)
wvfs_matrix = transpose(hcat(wvfs_dwt.signal...))


# Define hyperparameter distributions
C_dist = LogUniform(1e-2, 1e10)
gamma_dist = LogUniform(1e-9, 1e3)

# Define parameter grid
param_grid = Dict(
    :gamma => gamma_dist,
    :cost => C_dist
)

# Define cross-validation
cv = Holdout(fraction_train=0.8, shuffle=true, rng=Random.GLOBAL_RNG)

# Define SVM model
model = LIBSVMClassifier(
    kernel = LIBSVM.Kernel.RadialBasis,
    class_weights = Dict(0 => 1.0, 1 => 1.0),
    probability = true
)

# Define random search
tuned_model = TunedModel(
    model = model,
    tuning = RandomSearch(n_iter=100, rng=Random.GLOBAL_RNG),
    resampling = cv,
    range = param_grid,
    measure = MLJ.accuracy
)

# Train the model
mach = machine(tuned_model, dwts_norm, labels)
println("Starting random hyperparameter search")
start_time = time()
fit!(mach)
println("--- $(round((time() - start_time) / 60, digits=2)) minutes elapsed ---")

# Print the best parameters and score
best_params = fitted_params(mach).best_model
best_score = fitted_params(mach).best_measurement
println("The best parameters for the SVM are $(best_params) with a score of $(round(best_score, digits=2))")
