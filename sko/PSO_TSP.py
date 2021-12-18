#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/19/12
# @Author  : github.com/t3bol90

# %%
import random
import matplotlib.pyplot as plt
import math

from sko.base import SkoBase


class Particle:
    @staticmethod
    def _distance(c1, c2):
        return math.hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def _path_cost(_route):
        return sum([Particle._distance(city, _route[index - 1]) for index, city in enumerate(_route)])

    def __init__(self, route, cost=None):
        self.route = route
        self.pbest = route
        self.current_cost = cost if cost else self.path_cost()
        self.pbest_cost = cost if cost else self.path_cost()
        self.velocity = []

    def clear_velocity(self):
        self.velocity.clear()

    def update_costs_and_pbest(self):
        self.current_cost = self.path_cost()
        if self.current_cost < self.pbest_cost:
            self.pbest = self.route
            self.pbest_cost = self.current_cost

    def path_cost(self):
        return Particle._path_cost(self.route)


class PSO(SkoBase):

    def __init__(self, iterations, population_size, gbest_probability=1.0, pbest_probability=1.0, points=None):
        self.points = points
        self.gbest = None
        self.gcost_iter = []
        self.iterations = iterations
        self.population_size = population_size
        self.particles = []
        self.gbest_probability = gbest_probability
        self.pbest_probability = pbest_probability

        solutions = self.initial_population()
        self.particles = [Particle(route=solution) for solution in solutions]

    def random_route(self):
        return random.sample(self.points, len(self.points))

    def initial_population(self):
        random_population = [self.random_route()
                             for _ in range(self.population_size - 1)]
        greedy_population = [self.greedy_route(0)]
        return [*random_population, *greedy_population]

    def greedy_route(self, start_index):
        unvisited = self.points[:]
        del unvisited[start_index]
        route = [self.points[start_index]]
        while len(unvisited):
            index, nearest_city = min(
                enumerate(unvisited), key=lambda item: Particle._distance(item[1], route[-1]))
            route.append(nearest_city)
            del unvisited[index]
        return route

    def run(self):
        self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
        plt.ion()
        plt.draw()
        for t in range(self.iterations):
            self.gbest = min(self.particles, key=lambda p: p.pbest_cost)
            self.gcost_iter.append(self.gbest.pbest_cost)

            for particle in self.particles:
                particle.clear_velocity()
                temp_velocity = []
                gbest = self.gbest.pbest[:]
                new_route = particle.route[:]

                for i in range(len(self.points)):
                    if new_route[i] != particle.pbest[i]:
                        swap = (i, particle.pbest.index(
                            new_route[i]), self.pbest_probability)
                        temp_velocity.append(swap)
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                for i in range(len(self.points)):
                    if new_route[i] != gbest[i]:
                        swap = (i, gbest.index(
                            new_route[i]), self.gbest_probability)
                        temp_velocity.append(swap)
                        gbest[swap[0]], gbest[swap[1]
                                              ] = gbest[swap[1]], gbest[swap[0]]

                particle.velocity = temp_velocity

                for swap in temp_velocity:
                    if random.random() <= swap[2]:
                        new_route[swap[0]], new_route[swap[1]] = \
                            new_route[swap[1]], new_route[swap[0]]

                particle.route = new_route
                particle.update_costs_and_pbest()
