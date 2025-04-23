# to do: optimize hyperparameters for the AffinityPropagation
# - preference 
# - damp


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
