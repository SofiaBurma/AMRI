import numpy as np

def differential_evolution(func, bounds, pop_size=50, generations=100, F=0.8, P=0.7):

    dim = len(bounds)

    population = np.array([
        [np.random.uniform(low, high) for low, high in bounds]
        for _ in range(pop_size)
    ])

    fitness = np.array([func(ind) for ind in population])

    history = []

    for gen in range(generations):

        for i in range(pop_size):

            idxs = list(range(pop_size))
            idxs.remove(i)

            a, b, c = population[np.random.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)

            mutant = np.clip(
                mutant,
                [b[0] for b in bounds],
                [b[1] for b in bounds]
            )

            trial = population[i].copy()

            j_rand = np.random.randint(dim)

            for j in range(dim):
                if np.random.rand() < P or j == j_rand:
                    trial[j] = mutant[j]

            trial_fitness = func(trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness

        history.append(np.min(fitness))

    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()

    return best_solution, history

import numpy as np

def pso(func, bounds, K=30, N=100):

    M = len(bounds)

    X = np.random.uniform(
        low=[b[0] for b in bounds],
        high=[b[1] for b in bounds],
        size=(K, M)
    )

    V = np.zeros((K, M))

    fitness = np.array([func(x) for x in X])

    X_best = X.copy()
    best_fitness = fitness.copy()

    best_idx = np.argmin(fitness)
    X_global = X[best_idx].copy()
    global_best_fitness = fitness[best_idx]

    history = []

    for _ in range(N):

        for k in range(K):

            r1, r2 = np.random.rand(M), np.random.rand(M)

            V[k] = (
                0.7 * V[k]
                + 2 * r1 * (X_best[k] - X[k])
                + 2 * r2 * (X_global - X[k])
            )

            X[k] = X[k] + V[k]

            current_fitness = func(X[k])

            if current_fitness < best_fitness[k]:
                X_best[k] = X[k].copy()
                best_fitness[k] = current_fitness

        best_idx = np.argmin(best_fitness)

        if best_fitness[best_idx] < global_best_fitness:
            X_global = X_best[best_idx].copy()
            global_best_fitness = best_fitness[best_idx]

        history.append(global_best_fitness)

    return X_global, history