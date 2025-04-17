using LIBSVM
using Random

function create_dataset(;seed::Int = 42, n_samples = 100, n_features::Int = 2)
    Random.seed!(seed)  # For reproducibility
    # Create random data points for two classes, # Combine data and create labels
    X_class1 = randn(n_features, n_samples) .+ 1.0  # Class 1 centered at (1, 1)
    X_class2 = randn(n_features, n_samples) .- 1.0  # Class 2 centered at (-1, -1)
    X = hcat(X_class1, X_class2)
    labels = vcat(ones(Int, n_samples), -ones(Int, n_samples))  # Labels: 1 and -1
    # Ensure X has the correct shape (200 samples, 2 features)
    @assert size(X, 2) == length(labels) "Number of samples in X must match the length of y"
    # The shape of X needs to be (nfeatures, nsamples)
    return X, labels
end

# create training data set: 
n_samples, n_features = 100, 2
X_train, labels_train = create_dataset(seed=42, n_samples=100, n_features=2)

# Train an SVM model with a radial basis function (RBF) kernel
model = svmtrain(X_train, labels_train; kernel=LIBSVM.Kernel.RadialBasis, cost=1.0, gamma=0.5)

# Predict labels for the training data
pred_labels_train = svmpredict(model, X_train)
accuracy_train = sum(pred_labels_train[1] .== labels_train) / length(labels_train)
println("Accuracy train: $accuracy_train")

# Predict labels for a test data set
X_test, labels_test = create_dataset(seed=22, n_samples = 500, n_features = n_features)
pred_labels_test = svmpredict(model, X_test)
accuracy_test = sum(pred_labels_test[1] .== labels_test) / length(labels_test)
println("Accuracy train: $accuracy_test")


## optimize cost and gamma hyperparameters
# Define the number of random samples to evaluate
n_random_samples = 50

# Define distributions for gamma and cost
gamma_dist = () -> 10.0^(rand(Uniform(-3, 3)))  # Samples gamma from 10^[-3, 3]
cost_dist = () -> 10.0^(rand(Uniform(-2, 2)))   # Samples cost from 10^[-2, 2]

# Placeholder for best parameters and accuracy
best_gamma = nothing
best_cost = nothing
best_accuracy = 0.0

# Perform random search
for _ in 1:n_random_samples
    gamma = gamma_dist()
    cost = cost_dist()
    
    # Train the model
    model = svmtrain(X_train, labels_train; kernel=LIBSVM.Kernel.RadialBasis, cost=cost, gamma=gamma)
    
    # Perform cross-validation (or use a validation set)
    pred_labels_test, _ = svmpredict(model, X_test)
    accuracy = sum(pred_labels_test .== labels_test) / length(labels_test)
    
    # Update best parameters if accuracy improves
    if accuracy > best_accuracy
        best_accuracy = accuracy
        best_gamma = gamma
        best_cost = cost
    end
end

println("Best gamma: $best_gamma, Best cost: $best_cost, Best accuracy: $best_accuracy")


