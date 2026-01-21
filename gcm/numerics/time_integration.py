"""
Time integration schemes for GCM

Implements various time-stepping methods:
- Forward Euler
- Leapfrog
- Runge-Kutta 3rd order
- Adams-Bashforth
"""

import numpy as np


class TimeIntegrator:
    """
    Time integration for GCM equations

    Handles time-stepping of prognostic variables
    """

    def __init__(self, method='rk3'):
        """
        Initialize time integrator

        Parameters
        ----------
        method : str
            Integration method: 'euler', 'leapfrog', 'rk3', 'ab2'
        """
        self.method = method

        # Storage for multi-step methods
        self.previous_tendencies = None

    def step(self, state, dt, tendency_function):
        """
        Advance state by one time step

        Parameters
        ----------
        state : ModelState
            Current model state
        dt : float
            Time step (s)
        tendency_function : callable
            Function that computes tendencies: tendency_function(state)
            Should modify state.du_dt, state.dv_dt, etc.

        Returns
        -------
        state : ModelState
            Updated state
        """
        if self.method == 'euler':
            return self._euler_step(state, dt, tendency_function)
        elif self.method == 'rk3':
            return self._rk3_step(state, dt, tendency_function)
        elif self.method == 'leapfrog':
            return self._leapfrog_step(state, dt, tendency_function)
        elif self.method == 'ab2':
            return self._ab2_step(state, dt, tendency_function)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")

    def _euler_step(self, state, dt, tendency_function):
        """Forward Euler time step"""
        # Compute tendencies
        state.reset_tendencies()
        tendency_function(state)

        # Update prognostic variables
        state.u += state.du_dt * dt
        state.v += state.dv_dt * dt
        state.T += state.dT_dt * dt
        state.q += state.dq_dt * dt
        state.qc += state.dqc_dt * dt
        state.qi += state.dqi_dt * dt
        state.ps += state.dps_dt * dt

        # Ensure physical constraints
        self._apply_constraints(state)

        # Update diagnostics
        state.update_diagnostics()

        # Update time
        state.time += dt

        return state

    def _rk3_step(self, state, dt, tendency_function):
        """
        3rd order Runge-Kutta time step

        More accurate and stable than forward Euler
        """
        # Save initial state
        u0 = state.u.copy()
        v0 = state.v.copy()
        T0 = state.T.copy()
        q0 = state.q.copy()
        qc0 = state.qc.copy()
        qi0 = state.qi.copy()
        ps0 = state.ps.copy()

        # Stage 1
        state.reset_tendencies()
        tendency_function(state)

        k1_u = state.du_dt.copy()
        k1_v = state.dv_dt.copy()
        k1_T = state.dT_dt.copy()
        k1_q = state.dq_dt.copy()
        k1_qc = state.dqc_dt.copy()
        k1_qi = state.dqi_dt.copy()
        k1_ps = state.dps_dt.copy()

        state.u = u0 + 0.5 * dt * k1_u
        state.v = v0 + 0.5 * dt * k1_v
        state.T = T0 + 0.5 * dt * k1_T
        state.q = q0 + 0.5 * dt * k1_q
        state.qc = qc0 + 0.5 * dt * k1_qc
        state.qi = qi0 + 0.5 * dt * k1_qi
        state.ps = ps0 + 0.5 * dt * k1_ps

        self._apply_constraints(state)
        state.update_diagnostics()

        # Stage 2
        state.reset_tendencies()
        tendency_function(state)

        k2_u = state.du_dt.copy()
        k2_v = state.dv_dt.copy()
        k2_T = state.dT_dt.copy()
        k2_q = state.dq_dt.copy()
        k2_qc = state.dqc_dt.copy()
        k2_qi = state.dqi_dt.copy()
        k2_ps = state.dps_dt.copy()

        state.u = u0 + 0.75 * dt * k2_u
        state.v = v0 + 0.75 * dt * k2_v
        state.T = T0 + 0.75 * dt * k2_T
        state.q = q0 + 0.75 * dt * k2_q
        state.qc = qc0 + 0.75 * dt * k2_qc
        state.qi = qi0 + 0.75 * dt * k2_qi
        state.ps = ps0 + 0.75 * dt * k2_ps

        self._apply_constraints(state)
        state.update_diagnostics()

        # Stage 3
        state.reset_tendencies()
        tendency_function(state)

        k3_u = state.du_dt.copy()
        k3_v = state.dv_dt.copy()
        k3_T = state.dT_dt.copy()
        k3_q = state.dq_dt.copy()
        k3_qc = state.dqc_dt.copy()
        k3_qi = state.dqi_dt.copy()
        k3_ps = state.dps_dt.copy()

        # Final update (3rd order accurate combination)
        state.u = u0 + dt * (k1_u/6.0 + k2_u/6.0 + 2.0*k3_u/3.0)
        state.v = v0 + dt * (k1_v/6.0 + k2_v/6.0 + 2.0*k3_v/3.0)
        state.T = T0 + dt * (k1_T/6.0 + k2_T/6.0 + 2.0*k3_T/3.0)
        state.q = q0 + dt * (k1_q/6.0 + k2_q/6.0 + 2.0*k3_q/3.0)
        state.qc = qc0 + dt * (k1_qc/6.0 + k2_qc/6.0 + 2.0*k3_qc/3.0)
        state.qi = qi0 + dt * (k1_qi/6.0 + k2_qi/6.0 + 2.0*k3_qi/3.0)
        state.ps = ps0 + dt * (k1_ps/6.0 + k2_ps/6.0 + 2.0*k3_ps/3.0)

        self._apply_constraints(state)
        state.update_diagnostics()

        state.time += dt

        return state

    def _leapfrog_step(self, state, dt, tendency_function):
        """
        Leapfrog time step (centered differences)

        Uses previous state for time-stepping
        Requires storage of n-1 state
        """
        # For first step, use Euler
        if not hasattr(state, 'u_prev'):
            return self._euler_step(state, dt, tendency_function)

        # Compute tendencies
        state.reset_tendencies()
        tendency_function(state)

        # Leapfrog step: u(n+1) = u(n-1) + 2*dt*dudt(n)
        u_new = state.u_prev + 2 * dt * state.du_dt
        v_new = state.v_prev + 2 * dt * state.dv_dt
        T_new = state.T_prev + 2 * dt * state.dT_dt
        q_new = state.q_prev + 2 * dt * state.dq_dt
        qc_new = state.qc_prev + 2 * dt * state.dqc_dt
        qi_new = state.qi_prev + 2 * dt * state.dqi_dt
        ps_new = state.ps_prev + 2 * dt * state.dps_dt

        # Save current state as previous
        state.u_prev = state.u.copy()
        state.v_prev = state.v.copy()
        state.T_prev = state.T.copy()
        state.q_prev = state.q.copy()
        state.qc_prev = state.qc.copy()
        state.qi_prev = state.qi.copy()
        state.ps_prev = state.ps.copy()

        # Update to new state
        state.u = u_new
        state.v = v_new
        state.T = T_new
        state.q = q_new
        state.qc = qc_new
        state.qi = qi_new
        state.ps = ps_new

        self._apply_constraints(state)
        state.update_diagnostics()

        state.time += dt

        return state

    def _ab2_step(self, state, dt, tendency_function):
        """Adams-Bashforth 2nd order"""
        # Compute current tendencies
        state.reset_tendencies()
        tendency_function(state)

        current_tendencies = {
            'u': state.du_dt.copy(),
            'v': state.dv_dt.copy(),
            'T': state.dT_dt.copy(),
            'q': state.dq_dt.copy(),
            'qc': state.dqc_dt.copy(),
            'qi': state.dqi_dt.copy(),
            'ps': state.dps_dt.copy()
        }

        if self.previous_tendencies is None:
            # First step: use Euler
            state.u += dt * current_tendencies['u']
            state.v += dt * current_tendencies['v']
            state.T += dt * current_tendencies['T']
            state.q += dt * current_tendencies['q']
            state.qc += dt * current_tendencies['qc']
            state.qi += dt * current_tendencies['qi']
            state.ps += dt * current_tendencies['ps']
        else:
            # AB2: u(n+1) = u(n) + dt * (1.5*f(n) - 0.5*f(n-1))
            state.u += dt * (1.5 * current_tendencies['u'] - 0.5 * self.previous_tendencies['u'])
            state.v += dt * (1.5 * current_tendencies['v'] - 0.5 * self.previous_tendencies['v'])
            state.T += dt * (1.5 * current_tendencies['T'] - 0.5 * self.previous_tendencies['T'])
            state.q += dt * (1.5 * current_tendencies['q'] - 0.5 * self.previous_tendencies['q'])
            state.qc += dt * (1.5 * current_tendencies['qc'] - 0.5 * self.previous_tendencies['qc'])
            state.qi += dt * (1.5 * current_tendencies['qi'] - 0.5 * self.previous_tendencies['qi'])
            state.ps += dt * (1.5 * current_tendencies['ps'] - 0.5 * self.previous_tendencies['ps'])

        # Save current tendencies for next step
        self.previous_tendencies = current_tendencies

        self._apply_constraints(state)
        state.update_diagnostics()

        state.time += dt

        return state

    def _apply_constraints(self, state, max_wind=200.0):
        """
        Apply physical constraints to prognostic variables

        - Humidity must be non-negative
        - Cloud water/ice must be non-negative
        - Pressure must be positive
        - Temperature must be reasonable
        - Wind speed must be reasonable
        """
        # Non-negative moisture
        state.q = np.maximum(0.0, state.q)
        state.qc = np.maximum(0.0, state.qc)
        state.qi = np.maximum(0.0, state.qi)

        # Positive pressure
        state.ps = np.maximum(1000.0, state.ps)

        # Reasonable temperature bounds
        state.T = np.clip(state.T, 150.0, 400.0)

        # Limit wind speed to prevent blowup
        wind_speed = np.sqrt(state.u**2 + state.v**2)
        mask = wind_speed > max_wind
        if np.any(mask):
            scale = np.where(mask, max_wind / np.maximum(wind_speed, 1e-10), 1.0)
            state.u *= scale
            state.v *= scale
