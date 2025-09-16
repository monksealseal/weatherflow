
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import unittest

import torch

from weatherflow.solvers.ode_solver import WeatherODESolver

class TestODESolver(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.solver = WeatherODESolver()
        
    def test_solve_simple_system_2d(self):
        def velocity_fn(x, t):
            return -x  # Simple decay

        x0 = torch.ones(10, 3)
        t = torch.linspace(0, 1, 10)

        solution, stats = self.solver.solve(velocity_fn, x0, t)
        self.assertEqual(solution.shape, (10, 10, 3))
        self.assertTrue(stats["success"])
        self.assertEqual(stats["constraint_violations"], 0.0)
        self.assertTrue(math.isfinite(stats["energy_conservation"]))

    def test_solve_tensor_4d(self):
        def velocity_fn(x, t):
            return -x

        solver = WeatherODESolver(physics_constraints=True)
        x0 = torch.ones(2, 3, 4, 5)
        t = torch.linspace(0, 1, 5)

        solution, stats = solver.solve(velocity_fn, x0, t)
        self.assertEqual(solution.shape, (5, 2, 3, 4, 5))
        self.assertTrue(stats["success"])
        self.assertGreaterEqual(stats["constraint_violations"], 0.0)
        self.assertTrue(math.isfinite(stats["energy_conservation"]))
