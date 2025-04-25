# to do: optimize hyperparameters for the AffinityPropagation
# - preference 
# - damp
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
using Random
using Base.Threads
include("$(@__DIR__)/utils_ml.jl")

# data settings and config 
asic = LegendData(:ppc01)
period = DataPeriod(3)
run = DataRun(52)
channel = ChannelId(1)
category = DataCategory(:cal)
filekeys = search_disk(FileKey, asic.tier[DataTier(:raw), category , period, run])
dsp_config = DSPConfig(dataprod_config(asic).dsp(filekeys[1]).default)

# ml settings: 
maxiter = 200
tol = 1.0e-6
npreference = 5
ndamp = 3
preference_quantiles = range(0.1, 0.95, npreference)
damps = range(0.5, 0.99, ndamp) # values from Esteban (0.85..0.99)

# plot settings 
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :ml_qualitycuts) * "/"

# load waveforms and prepare for AP
wvf_max = Int(1e6)
nsamples = 10000
rng = MersenneTwister(1234)
_idx = Int.(rand(rng, 1:1e5, nsamples))

data_raw = TTable(read_ldata(asic, DataTier(:raw), filekeys, channel))[_idx]
wvfs_train_raw = data_raw.waveform
wvfs_train_eventnumber = data_raw.eventnumber

# baseline-shift and normalize waveforms 
wvfs_train = normalize_waveforms(wvfs_train_raw, dsp_config.bl_window)

# transform waveforms to a matrix of size (n_waveforms, n_samples) for ML clustering
wvfs_matrix = transpose(hcat(wvfs_train.signal...))
if size(wvfs_matrix) !== (length(wvfs_train), length(wvfs_train[1].signal)) 
    error("Waveform matrix size is not correct - it should be (n_waveforms, n_samples), but is $(size(wvfs_matrix))")
end
X_train = MLJ.table(wvfs_matrix)

# Similary matrix
S =  - pairwise(Cityblock(), wvfs_matrix, dims = 1)
Su = S[triu(trues(size(S)))]
preferences = quantile.(Ref(Su), preference_quantiles)

# get number of cluster over grid
grid_clusters = fill(0, length(preferences), length(damps))
grid_converged = fill(false, length(preferences), length(damps))
grid_preferences =fill(0.0, length(preferences), length(damps))
grid_damps = fill(0.0, length(preferences), length(damps))

Threads.@threads for idx in 1:(length(preferences) * length(damps))
    p = div(idx - 1, length(damps)) + 1
    d = mod(idx - 1, length(damps)) + 1
    _damp = damps[d]
    _preference = preferences[p]

    model = AffinityPropagation(
            damp = _damp, 
            maxiter = maxiter, 
            tol = tol, 
            preference =  _preference, 
            metric = Cityblock())
    _machine = machine(model)
    MLJ.predict(_machine, X_train)
    _report = report(_machine)
    grid_clusters[p, d] = length(_report.cluster_labels)
    grid_converged[p, d] = _report.converged
    grid_preferences[p, d] = _preference
    grid_damps[p, d] = _damp
end

ap_opt = (
    grid_clusters = grid_clusters,
    grid_converged = grid_converged,
    grid_preferences = grid_preferences,
    grid_damps = grid_damps,
)
# save intermediate results from AP 
pars_ml = PropDict(:ap_opt => ap_opt)
writelprops(asic.par[category].rpars.qc_ml[period], run, PropDict("$channel" => pars_ml))
