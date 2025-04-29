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
maxiter = 1000
tol = 1.0e-6
npreference = 6
ndamp = 3
preference_quantiles = vec(hcat(range(0.01, 0.05, 3)...,range(0.1, 0.5, npreference-3)...))
damps = range(0.85, 0.99, ndamp) # values from Esteban (0.85..0.99). otherwise high probability that AP does not converge. 

# plot settings 
plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :ml_qualitycuts) * "/"

# load waveforms and prepare for AP
wvf_max = Int(1e5)
nsamples = 10000
rng = MersenneTwister(1234)
_idx = randperm(rng, wvf_max)[1:nsamples]

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
_, time_grid, _, _, _ = @timed Threads.@threads for idx in 1:(length(preferences) * length(damps))
    p = div(idx - 1, length(damps)) + 1
    d = mod(idx - 1, length(damps)) + 1
    _damp = damps[d]
    _preference = preferences[p]

    local model = AffinityPropagation(
            damp = _damp, 
            maxiter = maxiter, 
            tol = tol, 
            preference =  _preference, 
            metric = Cityblock())
    local _machine = machine(model)
    MLJ.predict(_machine, X_train)
    local _report = report(_machine)
    grid_clusters[p, d] = length(_report.cluster_labels)
    grid_converged[p, d] = _report.converged
end

ap_opt = (
    grid_clusters = grid_clusters,
    grid_converged = grid_converged,
    time_grid = time_grid,
    preferences = preferences,
    preference_quantiles = preference_quantiles,
    dampings = damps,
    wvfs_train_eventnumber = wvfs_train_eventnumber,
)

# save intermediate results from AP 
pars_ml = PropDict(:ap_opt => ap_opt)
writelprops(asic.par[category].rpars.qc_ml[period], run, PropDict("$channel" => pars_ml))

preference_quantiles = pars_ml.ap_opt.preference_quantiles
damps = pars_ml.ap_opt.dampings
clusters = hcat(pars_ml.ap_opt.grid_clusters...)
converged = hcat(pars_ml.ap_opt.grid_converged...)
clusters = Float64.(clusters)
clusters[.!converged] .= NaN

begin 
    x = 1:length(preference_quantiles)
    fig = Figure()
    ax = Axis(fig[1,1],  title =Juleanita.get_plottitle(filekeys[1], _channel2detector(asic, channel), "AP hyperpar optimization"),
                        xlabel = "Preferences quantile", ylabel = "Damping",  
                        xticks = (x, string.(round.(preference_quantiles, digits=2))),
                        yticks = (damps, string.(round.(damps, digits=2))),
                        titlesize = 14)
    heatmap!(ax, x, damps, clusters, colormap = :isoluminant_cm_70_c39_n256)
    # Overlay text annotations for each cell
    for i in eachindex(preference_quantiles)
        for j in eachindex(damps)
            text!(
                ax,
                x[i],  # x-coordinate (preference value)
                damps[j],        # y-coordinate (damp value)
                text = string(clusters[i, j]),  # Value in the cell
                align = (:center, :center),     # Center the text in the cell
                fontsize = 12,                  # Adjust font size as needed
                color = :white                  # Text color
            )
        end
    end
    fig

    # plot settings 
    plt_folder = LegendDataManagement.LDMUtils.get_pltfolder(asic, filekeys[1], :ml_qualitycuts) * "/AP_hyperpars/"
    if !isdir(plt_folder)
        mkpath(plt_folder)
    end
    pname = "$(plt_folder)AP_hyperpars_opt_$(length(damps))damps$(minimum(damps))-$(maximum(damps))_$(length(preference_quantiles))qprefs$(minimum(preference_quantiles))-$(maximum(preference_quantiles)).png"
    save(pname, fig)
    @info "Figure saved to $pname"
    fig
end 

# report 
begin 
    column_names = vcat([Symbol("pref$(round(p, digits=2))") for p in preference_quantiles]...)
    tab_clusters = TTable(damp = damps; (column_names .=> Vector(eachrow(clusters)))...)
    tab_converged = TTable(damp = damps; (column_names .=> Vector(eachrow(converged)))...)

    report = lreport()
    lreport!(report, "# Affinity Propagation hyperparameters optimization")
    lreport!(report, "We perfrom AP on data over a grid of different dampings and preferences quantiles. The following tables shows the number of resulting cluster (exemplars) for different combinations of damping and preference. The goal is to have about 100 exemplars. ")
    lreport!(report, fig )
    lreport!(report, "Computing time : $(round(pars_ml.ap_opt.time_grid/3600, digits = 2)) hours")
    lreport!(report, "## Number of clusters/exemplars:")
    lreport!(report, tab_clusters)
    lreport!(report, "## AP Converged:")
    lreport!(report, tab_converged)
    writelreport(replace(pname, ".png" => ".md"), report)
    @info "Report saved to $(replace(pname, ".png" => ".md"))"
end 

