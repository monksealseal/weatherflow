
import unittest
import torch
from weatherflow.solvers.ode_solver import WeatherODESolver

class TestODESolver(unittest.TestCase):
    def setUp(self):
        self.solver = WeatherODESolver()
        
    def test_solve_simple_system(self):
        def velocity_fn(x, t):
            return -x  # Simple decay
            
        x0 = torch.ones(10, 3)
        t = torch.linspace(0, 1, 10)
        
        solution, stats = self.solver.solve(velocity_fn, x0, t)
        self.assertEqual(solution.shape, (10, 10, 3))
        self.assertTrue(stats["success"])
        
    def test_physics_constraints(self):
        solver = WeatherODESolver(physics_constraints=True)
        # Test conservation properties
