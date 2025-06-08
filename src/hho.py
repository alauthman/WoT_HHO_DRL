import numpy as np
import random

class HarrisHawksOptimizer:
    def __init__(self, obj_fn, lb, ub, dim, population_size=10, max_iter=20):
        self.obj_fn = obj_fn
        self.lb = np.array(lb)
        self.ub = np.array(ub)
        self.dim = dim
        self.population_size = population_size
        self.max_iter = max_iter

    def optimize(self):
        X = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(self.obj_fn, 1, X)
        best_idx = np.argmin(fitness)
        X_best = X[best_idx].copy()
        fit_best = fitness[best_idx]

        for t in range(self.max_iter):
            E1 = 2 * (1 - t / self.max_iter)
            for i in range(self.population_size):
                E0 = 2 * random.random() - 1
                E = E1 * E0
                if abs(E) >= 1:
                    q = random.random()
                    rand_idx = np.random.randint(self.population_size)
                    X_rand = X[rand_idx]
                    if q < 0.5:
                        X[i] = X_rand - random.random() * abs(X_rand - 2 * random.random() * X[i])
                    else:
                        X[i] = (X_best - X.mean(axis=0)) - random.random() * (self.ub - self.lb) * random.random() + self.lb
                else:
                    r = random.random()
                    if r >= 0.5:
                        X[i] = X_best - E * abs(X_best - X[i])
                    else:
                        X[i] = X_best - E * abs(X_best - X.mean(axis=0))

                X[i] = np.clip(X[i], self.lb, self.ub)
                fitness_i = self.obj_fn(X[i])
                if fitness_i < fit_best:
                    X_best = X[i].copy()
                    fit_best = fitness_i

        return X_best, fit_best
