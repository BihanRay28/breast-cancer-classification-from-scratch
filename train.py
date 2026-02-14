from nn_core import initialize_parameters, forward_prop, compute_cost, back_prop, parameter_update, layer_size

def train(X, Y, n_h, iterations=8000, print_cost=False, learning_rate=0.01):
    n_x, n_y, _ = layer_size(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)

    costs = []

    for i in range(iterations):
        cache = forward_prop(parameters, X)
        cost = compute_cost(cache, Y)
        gradients = back_prop(parameters, cache, X, Y)
        parameters = parameter_update(parameters, gradients, learning_rate)

        if i % 100 == 0:          # store cost every 100 iters for a smooth graph
            costs.append(cost)

        if print_cost and i % 1000 == 0:
            print(f"cost after {i}: {cost}")

    return parameters, costs
