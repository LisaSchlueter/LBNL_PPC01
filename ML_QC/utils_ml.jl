function normalize_waveforms(wvfs::ArrayOfRDWaveforms, bl_window::ClosedInterval{<:Quantity})
    bl_stats = signalstats.(wvfs, leftendpoint(bl_window), rightendpoint(bl_window))
    wvfs_shift = shift_waveform.(wvfs, -bl_stats.mean)
    wvf_absmax = maximum.(map(x-> abs.(x), wvfs_shift.signal))
    return multiply_waveform.(wvfs_shift, 1 ./ wvf_absmax)
end

function trainAP(wvfs_train::ArrayOfRDWaveforms; preference_quantile::T = 0.5, damp::T = 0.1, maxiter::Int = 200, tol::T = 1.0e-6) where T<:Real
    # transform waveforms to a matrix of size (n_waveforms, n_samples) for ML clustering
    wvfs_matrix = transpose(hcat(wvfs_train.signal...))
    if size(wvfs_matrix) !== (length(wvfs_train), length(wvfs_train[1].signal)) 
        error("Waveform matrix size is not correct - it should be (n_waveforms, n_samples), but is $(size(wvfs_matrix))")
    end
    X = MLJ.table(wvfs_matrix)

    # Similary matrix
    S =  - pairwise(Cityblock(), wvfs_matrix, dims = 1)
    Su = S[triu(trues(size(S)))]
    preference = quantile(Su, preference_quantile)

    # prepare model
    model = AffinityPropagation(
                damp = damp, 
                maxiter = maxiter, 
                tol = tol, 
                preference = preference, 
                metric = Cityblock())

    # apply model to data
    begin
        _machine = machine(model)
        waveform_ap_labels = collect(MLJ.predict(_machine, X)) # these are the labels of each waveform (assigned to a cluster)
    end        

    # evaluate the model 
    _report = report(_machine)

    # check convergence 
    if !_report.converged
        @warn "AffinityPropagation did not converge. Check the model parameters."
    end
    @info "$(length(_report.cluster_labels)) clusters found"

    # summarize results
    result_ap = (
        waveforms = (ap_labels = waveform_ap_labels, ),
        ap = (damp = damp,  preference = preference, preference_quantile = preference_quantile, 
            iterations = _report.iterations, ncluster = length(_report.exemplars), converged = _report.converged),
        exemplars = (idx = _report.exemplars, eventnumber =  wvfs_train_eventnumber[_report.exemplars],
                    labels = _report.cluster_labels),
    )

    report_ap = (waveforms = result_ap.waveforms,
                    ap = result_ap.ap,
                    exemplars = merge(result_ap.exemplars, (centers = wvfs_train[_report.exemplars],) ))

    return result_ap, report_ap 
end 


function plot_APexemplars(centers, labels::Vector, colors::Vector)
    ncluster = length(centers)
    ncol = round(Int, sqrt(ncluster))
    nrow = ceil(Int, ncluster/ncol)
    fig = Figure(size=(ncol*100,nrow*110), figure_padding = 5)
	for i in 1:ncluster
        # wf_idx = result_ap.exemplars.idx[i]
        # qc_label = result_ap.waveforms.qc_labels[wf_idx]
        _row = div(i-1,ncol)+1
        _col = mod(i-1,ncol)+1
        _ax =  Axis(fig[_row,_col], xticklabelsize = 5, yticklabelsize = 5)
		lines!(_ax, ustrip.(centers[i].time), centers[i].signal, linewidth = 1.5, color = colors[i])
        Label(fig[_row, _col], "$(labels[i])", fontsize = 10, padding = (10, 10),  tellwidth = false)
        hidedecorations!(_ax)
	end
    Label(fig[0, :], "Cluster Center Exemplar Waveforms", fontsize = 20, tellwidth = false)
	rowgap!(fig.layout, 5)
    colgap!(fig.layout, 5)
    return fig
end