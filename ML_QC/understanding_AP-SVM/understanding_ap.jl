
using MLJ
using MLJClusteringInterface
using StatsBase, Statistics
using LinearAlgebra
using Distances
using Random
function create_dataset(;seed::Int = 42, n_samples = 100, n_features::Int = 2)
    Random.seed!(seed)  # For reproducibility
    # Create random data points for two classes, # Combine data and create labels
    X_class1 = randn(n_features, n_samples) .+ 1.0  # Class 1 centered at (1, 1)
    X_class2 = randn(n_features, n_samples) .- 1.0  # Class 2 centered at (-1, -1)
    X = hcat(X_class1, X_class2)
    labels = vcat(ones(Int, n_samples), -ones(Int, n_samples))  # Labels: 1 and -1
    # Ensure X has the correct shape (samples, features)
    @assert size(X, 2) == length(labels) "Number of samples in X must match the length of y"
    return transpose(X), labels
end

# create training data set: 
n_samples, n_features = 100, 2
X_train, labels_train = create_dataset(seed=42, n_samples=100, n_features=2)
MLJ.table(X_train) # check the shape of the data
# calcualte similary matrix to get an idea what the preference could look like: 
S =  - pairwise(Cityblock(), X_train, dims = 1)
Su = S[triu(trues(size(S)))]
preference = -100.0
damp = 0.8

# prepare model
model = AffinityPropagation(
            damp = damp, 
            maxiter = 200, 
            tol = 1.0e-6, 
            preference = preference, 
            metric = Cityblock())

# train model 
_machine = machine(model)

# evaluate train data 
labels_pred_train = collect(MLJ.predict(_machine, X_train)) # these are the labels of each waveform (assigned to a cluster)
# _report_train = report(_machine)
# _report_train.exemplars # that's a lot of clusters 
labels_predre_train =  labels_pred_train  
labels_predre_train[labels_pred_train .== 1] .= 1
labels_predre_train[labels_pred_train .== 2] .= -1
accuracy_train = sum(labels_predre_train .== labels_train) / length(labels_train)
println("Accuracy train: $accuracy_train")

