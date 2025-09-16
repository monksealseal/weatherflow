
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

try:
    from weatherflow.solvers.ode_solver import WeatherODESolver
except ModuleNotFoundError:
    module_path = Path(__file__).resolve().parents[1] / "weatherflow" / "solvers" / "ode_solver.py"
    spec = importlib.util.spec_from_file_location("weatherflow.solvers.ode_solver", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("weatherflow.solvers", ModuleType("weatherflow.solvers"))
    sys.modules["weatherflow.solvers.ode_solver"] = module
    spec.loader.exec_module(module)
    WeatherODESolver = module.WeatherODESolver

class TestODESolver(unittest.TestCase):
    def setUp(self):
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
