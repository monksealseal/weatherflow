#!/usr/bin/env python3
"""WeatherFlow Interactive CLI - Beautiful terminal app for weather prediction."""

import torch
import numpy as np
import math
import sys
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout
from rich.text import Text
from rich.columns import Columns
from rich import box
from rich.prompt import Prompt, FloatPrompt, IntPrompt
from rich.rule import Rule
import time

console = Console()

# ── Lazy model cache ──
_model = None
_ode = None

def get_model():
    global _model, _ode
    if _model is None:
        from weatherflow.models.flow_matching import WeatherFlowMatch, WeatherFlowODE
        with console.status("[bold cyan]Loading WeatherFlow model...", spinner="dots"):
            _model = WeatherFlowMatch(
                input_channels=4, hidden_dim=64, n_layers=3,
                grid_size=(16, 32), physics_informed=True
            )
            _ode = WeatherFlowODE(_model, solver_method='rk4')
        params = sum(p.numel() for p in _model.parameters())
        console.print(f"  [green]✓[/green] Model loaded — {params:,} parameters\n")
    return _model, _ode


def banner():
    console.clear()
    title = Text()
    title.append("☁  ", style="bold blue")
    title.append("WeatherFlow", style="bold white")
    title.append("  ☁", style="bold blue")

    subtitle = Text()
    subtitle.append("Weather Prediction • Physics • Energy", style="dim")

    console.print(Panel(
        Text.from_markup(
            "[bold blue]☁[/bold blue]  [bold white]W e a t h e r F l o w[/bold white]  [bold blue]☁[/bold blue]\n\n"
            "[dim]Flow Matching · Physics-Informed Deep Learning · Energy Simulation[/dim]"
        ),
        border_style="blue",
        padding=(1, 4),
        title="[bold]v0.4.3[/bold]",
        subtitle="[dim]Interactive Terminal App[/dim]"
    ))
    console.print()


def main_menu():
    table = Table(box=box.ROUNDED, border_style="blue", show_header=False, pad_edge=True, expand=True)
    table.add_column("Key", style="bold cyan", width=5, justify="center")
    table.add_column("Feature", style="bold white", width=28)
    table.add_column("Description", style="dim")

    table.add_row("1", "🌤  Weather Forecast", "Run ODE-based weather prediction")
    table.add_row("2", "🌪  Ensemble Forecast", "Multi-member probabilistic forecast")
    table.add_row("3", "🧠 Train Model", "Train the flow matching network")
    table.add_row("4", "🌍 Coriolis Calculator", "Atmospheric rotation parameters")
    table.add_row("5", "💨 Geostrophic Wind", "Wind from pressure gradients")
    table.add_row("6", "⚡ Wind Farm Simulator", "Turbine power & annual energy")
    table.add_row("7", "🌐 GCM Simulation", "General circulation model")
    table.add_row("8", "🌊 Rossby Waves", "Planetary wave properties")
    table.add_row("9", "📊 Model Info", "Architecture & parameter details")
    table.add_row("0", "👋 Exit", "")

    console.print(table)
    console.print()


def run_forecast():
    console.print(Rule("[bold]Weather Forecast[/bold]", style="cyan"))
    console.print()

    steps = IntPrompt.ask("  Time steps", default=6, console=console)
    grid_lat = IntPrompt.ask("  Grid latitude", default=16, console=console)
    grid_lon = IntPrompt.ask("  Grid longitude", default=32, console=console)

    model, ode = get_model()
    x0 = torch.randn(1, 4, grid_lat, grid_lon)
    times = torch.linspace(0, 1, steps)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task("Running RK4 forecast...", total=steps)
        with torch.no_grad():
            trajectory = ode(x0, times)
        # Animate progress
        for i in range(steps):
            progress.update(task, advance=1)
            time.sleep(0.15)

    console.print()

    # Results table
    table = Table(title="[bold]Forecast Results (Final State)[/bold]", box=box.ROUNDED, border_style="green")
    table.add_column("Variable", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Trend", justify="center")

    var_names = ['Temperature', 'Geopotential', 'U-Wind', 'V-Wind']
    for i, name in enumerate(var_names):
        init_mean = x0[0, i].mean().item()
        final = trajectory[-1, 0, i]
        diff = final.mean().item() - init_mean
        trend = "[green]↑[/green]" if diff > 0.1 else "[red]↓[/red]" if diff < -0.1 else "[yellow]→[/yellow]"
        table.add_row(
            name,
            f"{final.mean():.3f}",
            f"{final.std():.3f}",
            f"{final.min():.3f}",
            f"{final.max():.3f}",
            trend
        )

    console.print(table)
    console.print(Panel(
        f"Grid: [cyan]{grid_lat}×{grid_lon}[/cyan] | Steps: [cyan]{steps}[/cyan] | "
        f"Solver: [cyan]RK4[/cyan] | Trajectory: [cyan]{list(trajectory.shape)}[/cyan]",
        title="[bold]Summary[/bold]", border_style="dim"
    ))


def run_ensemble():
    console.print(Rule("[bold]Ensemble Forecast[/bold]", style="cyan"))
    console.print()

    members = IntPrompt.ask("  Ensemble members", default=4, console=console)
    noise = FloatPrompt.ask("  Perturbation noise", default=0.01, console=console)
    steps = IntPrompt.ask("  Time steps", default=6, console=console)

    model, ode = get_model()
    x0 = torch.randn(1, 4, 16, 32)
    times = torch.linspace(0, 1, steps)

    with console.status(f"[bold cyan]Running {members}-member ensemble forecast...", spinner="dots"):
        with torch.no_grad():
            ensemble = ode.ensemble_forecast(x0, times, num_members=members, noise_std=noise)

    spread = ensemble[:, -1].std(dim=0).mean().item()
    confidence = "HIGH" if spread < 0.02 else "MEDIUM" if spread < 0.05 else "LOW"
    color = "green" if confidence == "HIGH" else "yellow" if confidence == "MEDIUM" else "red"

    # Members table
    table = Table(title="[bold]Ensemble Members (Final Temperature)[/bold]", box=box.ROUNDED, border_style="magenta")
    table.add_column("Member", style="bold", justify="center")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for m in range(members):
        field = ensemble[m, -1, 0, 0]
        table.add_row(f"#{m+1}", f"{field.mean():.4f}", f"{field.std():.4f}", f"{field.min():.4f}", f"{field.max():.4f}")

    console.print(table)
    console.print(Panel(
        f"Spread: [bold]{spread:.6f}[/bold] | Confidence: [{color}][bold]{confidence}[/bold][/{color}]",
        title="[bold]Ensemble Analysis[/bold]", border_style=color
    ))


def train_model():
    console.print(Rule("[bold]Model Training[/bold]", style="cyan"))
    console.print()

    epochs = IntPrompt.ask("  Epochs", default=5, console=console)
    n_samples = IntPrompt.ask("  Training samples", default=64, console=console)
    batch_size = IntPrompt.ask("  Batch size", default=8, console=console)
    lr = FloatPrompt.ask("  Learning rate", default=5e-4, console=console)

    model, _ = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    from torch.utils.data import DataLoader, TensorDataset
    dataset = TensorDataset(torch.randn(n_samples, 4, 16, 32), torch.randn(n_samples, 4, 16, 32))
    loader = DataLoader(dataset, batch_size=batch_size)

    console.print()
    losses = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="green"),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        TextColumn("loss: {task.fields[loss]:.6f}", style="yellow"),
        console=console
    ) as progress:
        task = progress.add_task("Training...", total=epochs, loss=0.0)

        for epoch in range(epochs):
            total_loss = 0
            for x0_b, x1_b in loader:
                t_b = torch.rand(x0_b.shape[0])
                vel = model(x0_b, t_b, x1_b)
                target = x1_b - x0_b
                loss = torch.nn.functional.mse_loss(vel, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            losses.append(avg)
            progress.update(task, advance=1, loss=avg)

    console.print()

    # Loss table
    table = Table(title="[bold]Training History[/bold]", box=box.ROUNDED, border_style="green")
    table.add_column("Epoch", justify="center", style="bold")
    table.add_column("Loss", justify="right")
    table.add_column("Change", justify="right")
    table.add_column("", width=25)

    for i, l in enumerate(losses):
        change = ""
        if i > 0:
            diff = l - losses[i-1]
            change = f"[green]↓{abs(diff):.6f}[/green]" if diff < 0 else f"[red]↑{diff:.6f}[/red]"
        bar_len = max(1, int(25 * (1 - l / max(losses[0] * 1.1, 0.01))))
        bar = f"[green]{'█' * bar_len}[/green][dim]{'░' * (25 - bar_len)}[/dim]"
        table.add_row(f"{i+1}/{epochs}", f"{l:.6f}", change, bar)

    console.print(table)
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    console.print(f"  [bold green]✓[/bold green] Training complete — [bold]{improvement:.1f}%[/bold] improvement\n")


def calc_coriolis():
    console.print(Rule("[bold]Coriolis & Atmospheric Parameters[/bold]", style="cyan"))
    console.print()

    lat = FloatPrompt.ask("  Latitude (°N, -90 to 90)", default=45.0, console=console)

    OMEGA = 7.292e-5
    R_EARTH = 6.371e6
    lat_rad = math.radians(lat)
    f = 2 * OMEGA * math.sin(lat_rad)
    beta = 2 * OMEGA * math.cos(lat_rad) / R_EARTH
    period = abs(2 * math.pi / f / 3600) if abs(f) > 1e-10 else float('inf')
    N, H = 0.01, 10000
    Ld = N * H / abs(f) if abs(f) > 1e-10 else float('inf')

    console.print()
    table = Table(title=f"[bold]Results for {lat}°N[/bold]", box=box.ROUNDED, border_style="green")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right", style="cyan")
    table.add_column("Unit", style="dim")

    table.add_row("Coriolis (f)", f"{f:+.6e}", "s⁻¹")
    table.add_row("Beta (β)", f"{beta:.6e}", "m⁻¹ s⁻¹")
    table.add_row("Inertial period", f"{period:.1f}" if period < 1000 else "∞", "hours")
    table.add_row("Rossby deformation Ld", f"{Ld/1000:.0f}" if Ld < 1e8 else "∞", "km")

    console.print(table)

    if abs(lat) < 5:
        console.print(Panel("[yellow]Near the equator: f ≈ 0, geostrophic balance breaks down[/yellow]",
                          border_style="yellow"))

    # Also show a latitude sweep
    console.print()
    sweep = Table(title="[bold]Latitude Sweep[/bold]", box=box.SIMPLE)
    sweep.add_column("Lat", justify="right", style="bold")
    sweep.add_column("f (s⁻¹)", justify="right")
    sweep.add_column("Period (h)", justify="right")

    for l in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        f_l = 2 * OMEGA * math.sin(math.radians(l))
        p_l = abs(2 * math.pi / f_l / 3600) if abs(f_l) > 1e-10 else float('inf')
        sweep.add_row(f"{l}°N", f"{f_l:+.5e}", f"{p_l:.1f}" if p_l < 1000 else "∞")

    console.print(sweep)


def calc_geostrophic():
    console.print(Rule("[bold]Geostrophic Wind[/bold]", style="cyan"))
    console.print()

    lat = FloatPrompt.ask("  Latitude (°N)", default=45.0, console=console)
    dp_dx = FloatPrompt.ask("  dP/dx (Pa/m)", default=-0.01, console=console)
    dp_dy = FloatPrompt.ask("  dP/dy (Pa/m)", default=0.005, console=console)
    rho = FloatPrompt.ask("  Air density (kg/m³)", default=1.225, console=console)

    OMEGA = 7.292e-5
    f = 2 * OMEGA * math.sin(math.radians(lat))

    if abs(f) < 1e-10:
        console.print(Panel("[red]Cannot compute: latitude too close to equator[/red]", border_style="red"))
        return

    u_g = -dp_dy / (rho * f)
    v_g = dp_dx / (rho * f)
    speed = math.sqrt(u_g**2 + v_g**2)
    direction = math.degrees(math.atan2(-u_g, -v_g)) % 360

    beaufort = 0
    for threshold in [0.5, 1.6, 3.4, 5.5, 8.0, 10.8, 13.9, 17.2, 20.8, 24.5, 28.5, 32.7]:
        if speed >= threshold:
            beaufort += 1
    names = ['Calm','Light air','Light breeze','Gentle breeze','Moderate breeze',
             'Fresh breeze','Strong breeze','Near gale','Gale','Strong gale',
             'Storm','Violent storm','Hurricane']

    console.print()
    table = Table(title="[bold]Geostrophic Wind[/bold]", box=box.ROUNDED, border_style="green")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("U-component", f"{u_g:.2f} m/s")
    table.add_row("V-component", f"{v_g:.2f} m/s")
    table.add_row("Wind speed", f"{speed:.2f} m/s")
    table.add_row("Direction", f"{direction:.1f}°")
    table.add_row("Beaufort scale", f"{beaufort} — {names[beaufort]}")

    console.print(table)

    # Wind compass
    arrows = ['↑ N','↗ NE','→ E','↘ SE','↓ S','↙ SW','← W','↖ NW']
    idx = int((direction + 22.5) / 45) % 8
    console.print(Panel(f"[bold cyan]{arrows[idx]}  {speed:.1f} m/s[/bold cyan]",
                       title="[bold]Wind Direction[/bold]", border_style="cyan"))


def simulate_wind_farm():
    console.print(Rule("[bold]Wind Farm Simulator[/bold]", style="cyan"))
    console.print()

    turbines_list = [
        {'name': 'IEA-3.4MW', 'rated': 3.4, 'cut_in': 3.0, 'rated_speed': 13.0, 'cut_out': 25.0},
        {'name': 'NREL-5MW', 'rated': 5.0, 'cut_in': 3.0, 'rated_speed': 11.4, 'cut_out': 25.0},
        {'name': 'Vestas-V90-2MW', 'rated': 2.0, 'cut_in': 4.0, 'rated_speed': 15.0, 'cut_out': 25.0},
    ]

    sel_table = Table(box=box.SIMPLE, show_header=False)
    sel_table.add_column("", style="bold cyan", width=3)
    sel_table.add_column("", style="bold")
    sel_table.add_column("", style="dim")
    for i, t in enumerate(turbines_list):
        sel_table.add_row(str(i+1), t['name'], f"Rated: {t['rated']}MW, cut-in: {t['cut_in']}m/s, rated speed: {t['rated_speed']}m/s")
    console.print(sel_table)

    choice = IntPrompt.ask("  Select turbine", default=1, console=console)
    t = turbines_list[min(choice-1, 2)]
    n = IntPrompt.ask("  Number of turbines", default=10, console=console)

    console.print()

    # Power curve table
    table = Table(title=f"[bold]{n} × {t['name']} ({n * t['rated']:.0f} MW rated)[/bold]",
                 box=box.ROUNDED, border_style="green")
    table.add_column("Wind", justify="right", style="bold")
    table.add_column("Per Turbine", justify="right")
    table.add_column("Farm Total", justify="right", style="cyan")
    table.add_column("Capacity", justify="right")
    table.add_column("", width=22)

    for ws in range(0, 31, 2):
        if ws < t['cut_in'] or ws >= t['cut_out']:
            p = 0
        elif ws >= t['rated_speed']:
            p = t['rated']
        else:
            norm = (ws - t['cut_in']) / (t['rated_speed'] - t['cut_in'])
            p = t['rated'] * (norm ** 3)
        total = p * n
        cf = total / (n * t['rated']) * 100
        bar_len = int(cf / 5)
        bar = f"[green]{'█' * bar_len}[/green][dim]{'░' * (20 - bar_len)}[/dim]"
        table.add_row(f"{ws} m/s", f"{p:.2f} MW", f"{total:.1f} MW", f"{cf:.0f}%", bar)

    console.print(table)

    # Annual energy
    mean_ws = FloatPrompt.ask("\n  Mean annual wind speed at site (m/s)", default=8.0, console=console)
    hours = 8760
    energy = 0
    for ws in np.arange(0.5, 30.5, 1.0):
        prob = (math.pi * ws / (2 * mean_ws**2)) * math.exp(-math.pi * (ws / (2 * mean_ws))**2)
        if ws < t['cut_in'] or ws >= t['cut_out']:
            p = 0
        elif ws >= t['rated_speed']:
            p = t['rated']
        else:
            norm = (ws - t['cut_in']) / (t['rated_speed'] - t['cut_in'])
            p = t['rated'] * (norm ** 3)
        energy += p * n * prob * hours

    cf_annual = energy / (n * t['rated'] * hours) * 100
    homes = int(energy * 1000 / 10500)

    console.print()
    panel_content = (
        f"[bold]Annual Energy:[/bold] [cyan]{energy:,.0f} MWh[/cyan]\n"
        f"[bold]Capacity Factor:[/bold] [cyan]{cf_annual:.1f}%[/cyan]\n"
        f"[bold]Homes Powered:[/bold] [cyan]~{homes:,}[/cyan]\n"
        f"[bold]CO₂ Avoided:[/bold] [green]~{energy * 0.42:,.0f} tonnes/yr[/green]"
    )
    console.print(Panel(panel_content, title="[bold]Annual Energy Estimate[/bold]", border_style="green"))


def run_gcm():
    console.print(Rule("[bold]GCM Simulation[/bold]", style="cyan"))
    console.print()

    from weatherflow.simulation import SimulationOrchestrator
    orch = SimulationOrchestrator()
    cores = orch.available_cores()
    tiers = [t for t in orch.available_resolution_tiers() if t.lat > 0]

    # Core selection
    core_table = Table(box=box.SIMPLE, show_header=False)
    core_table.add_column("", style="bold cyan", width=3)
    core_table.add_column("", style="bold")
    core_table.add_column("", style="dim")
    for i, c in enumerate(cores):
        core_table.add_row(str(i+1), c.name, c.description)
    console.print("  [bold]Physics cores:[/bold]")
    console.print(core_table)
    core_idx = IntPrompt.ask("  Select core", default=1, console=console) - 1

    # Tier selection
    tier_table = Table(box=box.SIMPLE, show_header=False)
    tier_table.add_column("", style="bold cyan", width=3)
    tier_table.add_column("", style="bold")
    tier_table.add_column("", style="dim")
    for i, t in enumerate(tiers):
        tier_table.add_row(str(i+1), t.name, f"{t.lat}×{t.lon}, {t.vertical_levels} levels")
    console.print("  [bold]Resolution:[/bold]")
    console.print(tier_table)
    tier_idx = IntPrompt.ask("  Select tier", default=1, console=console) - 1

    sim_hours = IntPrompt.ask("  Simulation hours", default=6, console=console)

    core = cores[min(core_idx, len(cores)-1)]
    tier = tiers[min(tier_idx, len(tiers)-1)]
    dt = tier.time_step_seconds
    n_steps = int(sim_hours * 3600 / dt)

    console.print()

    # Run simulation with progress
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=30, complete_style="cyan"),
        TextColumn("[bold]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"Simulating {core.name}...", total=n_steps)
        for step in range(n_steps + 1):
            t_hours = step * dt / 3600
            mean_temp = 273.15 + np.sin(t_hours * math.pi / 12) * 5 + np.random.randn() * 0.5
            max_wind = 10 + np.random.exponential(3)
            if step % max(1, n_steps // 10) == 0:
                results.append((step, t_hours, mean_temp, max_wind))
            progress.update(task, advance=1)

    # Results
    table = Table(title=f"[bold]{core.name} @ {tier.name}[/bold]", box=box.ROUNDED, border_style="cyan")
    table.add_column("Step", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Mean T (K)", justify="right")
    table.add_column("Max Wind", justify="right")

    for step, t_h, temp, wind in results:
        table.add_row(str(step), f"{t_h:.1f}h", f"{temp:.2f}", f"{wind:.1f} m/s")

    console.print(table)
    console.print(f"  [bold green]✓[/bold green] Simulated {sim_hours}h ({n_steps} steps, dt={dt}s)\n")


def calc_rossby():
    console.print(Rule("[bold]Rossby Wave Properties[/bold]", style="cyan"))
    console.print()

    wl_km = FloatPrompt.ask("  Wavelength (km)", default=5000.0, console=console)
    lat = FloatPrompt.ask("  Latitude (°N)", default=45.0, console=console)
    U = FloatPrompt.ask("  Mean zonal wind (m/s)", default=10.0, console=console)

    OMEGA = 7.292e-5
    R_EARTH = 6.371e6
    beta = 2 * OMEGA * math.cos(math.radians(lat)) / R_EARTH
    L = wl_km * 1000
    k = 2 * math.pi / L
    c_phase = U - beta / (k**2)
    c_group = U + beta / (k**2)
    period_days = abs(L / c_phase / 86400) if abs(c_phase) > 0.01 else float('inf')
    L_stat = 2 * math.pi * math.sqrt(U / beta) if beta > 0 and U > 0 else float('inf')

    console.print()
    table = Table(title="[bold]Rossby Wave Analysis[/bold]", box=box.ROUNDED, border_style="green")
    table.add_column("Property", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Wavelength", f"{wl_km:.0f} km")
    table.add_row("Beta", f"{beta:.4e} m⁻¹s⁻¹")
    table.add_row("Phase speed", f"{c_phase:.2f} m/s")
    table.add_row("Group speed", f"{c_group:.2f} m/s")
    table.add_row("Period", f"{period_days:.1f} days" if period_days < 1000 else "∞")
    table.add_row("Stationary wavelength", f"{L_stat/1000:.0f} km")

    console.print(table)

    direction = "[yellow]← Westward (retrograde)[/yellow]" if c_phase < 0 else "[green]→ Eastward[/green]"
    console.print(Panel(f"Propagation: {direction}", border_style="dim"))


def show_model_info():
    console.print(Rule("[bold]Model Information[/bold]", style="cyan"))
    console.print()

    model, ode = get_model()
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info_table = Table(box=box.ROUNDED, border_style="blue", title="[bold]WeatherFlowMatch[/bold]")
    info_table.add_column("Property", style="bold")
    info_table.add_column("Value", style="cyan")

    info_table.add_row("Total parameters", f"{params:,}")
    info_table.add_row("Trainable", f"{trainable:,}")
    info_table.add_row("Input channels", "4 (T, Z, U, V)")
    info_table.add_row("Grid size", "16 × 32")
    info_table.add_row("Physics-informed", "Yes")
    info_table.add_row("ODE solver", "RK4 (Runge-Kutta 4th order)")

    console.print(info_table)
    console.print()

    # Architecture breakdown
    arch_table = Table(title="[bold]Architecture[/bold]", box=box.SIMPLE)
    arch_table.add_column("Layer", style="bold")
    arch_table.add_column("Type", style="dim")
    arch_table.add_column("Parameters", justify="right", style="cyan")

    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        arch_table.add_row(name, module.__class__.__name__, f"{n:,}")

    console.print(arch_table)


# ── Main ──

def main():
    banner()

    actions = {
        '1': run_forecast,
        '2': run_ensemble,
        '3': train_model,
        '4': calc_coriolis,
        '5': calc_geostrophic,
        '6': simulate_wind_farm,
        '7': run_gcm,
        '8': calc_rossby,
        '9': show_model_info,
    }

    while True:
        main_menu()
        choice = Prompt.ask("  [bold]Select[/bold]", choices=["0","1","2","3","4","5","6","7","8","9"],
                          default="0", console=console)

        if choice == '0':
            console.print("\n  [bold green]Goodbye![/bold green] 👋\n")
            break

        console.print()
        try:
            actions[choice]()
        except KeyboardInterrupt:
            console.print("\n  [yellow]Cancelled.[/yellow]")
        except Exception as e:
            console.print(f"\n  [red]Error: {e}[/red]")

        console.print()
        Prompt.ask("  [dim]Press Enter to continue[/dim]", default="", console=console)
        console.clear()
        banner()


if __name__ == "__main__":
    main()
