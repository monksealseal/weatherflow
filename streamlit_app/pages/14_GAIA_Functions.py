"""GAIA Function Studio for WeatherFlow."""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st
import torch
import xarray as xr

from weatherflow.data.gaia.normalize import NormalizationStats
from weatherflow.data.gaia.regrid import (
    RegridStrategy,
    TargetGrid,
    _bilinear_regrid,
    _coarsen_factor,
    _conservative_regrid,
    _grid_step,
    _latitude_weights,
    _validate_alignment,
    regrid_dataset,
)
from weatherflow.data.gaia.schema import GaiaVariableSchema
from weatherflow.data.gaia.shards import ShardManifest, _hash_file, write_time_shards
from weatherflow.data.gaia.sources import (
    ERA5ZarrSource,
    _get_env,
    verify_access_credentials,
)
from weatherflow.evaluation.gaia.metrics import (
    EvaluationSelectionConfig,
    _apply_selection_pair,
    _normalize_axis,
    _resolve_indices,
    _to_numpy,
    _validate_axis,
    acc,
    apply_selection,
    brier_score,
    crps_ensemble,
    mae,
    reliability_curve,
    rmse,
)
from weatherflow.gaia.constraints import (
    Constraint,
    ConstraintApplier,
    MeanPreservingConstraint,
    NonNegativeConstraint,
    RangeConstraint,
)
from weatherflow.gaia.decoder import (
    ConditionalDiffusionDecoder,
    DecoderConfig,
    EnsembleDecoder,
    GaiaDecoder,
)
from weatherflow.gaia.encoder import GaiaGridEncoder, _icosahedron, _subdivide
from weatherflow.gaia.model import GaiaConfig, GaiaModel
from weatherflow.gaia.processor import (
    GaiaProcessor,
    GNNBlock,
    LatAwareBias,
    ProcessorConfig,
    SpectralMixingBlock,
)
from weatherflow.gaia.sampling import gaia_sample
from weatherflow.training.gaia import calibrate as gaia_calibrate
from weatherflow.training.gaia import finetune as gaia_finetune
from weatherflow.training.gaia import losses as gaia_losses
from weatherflow.training.gaia import pretrain as gaia_pretrain
from weatherflow.training.gaia.schedules import (
    RolloutCurriculum,
    linear_rollout_schedule,
)


st.set_page_config(
    page_title="GAIA Function Studio",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 GAIA Function Studio")
st.write(
    "Every GAIA utility, model component, data pipeline, and training helper is exposed here "
    "with a runnable demo or visualization. Use the expanders to explore each callable."
)


DemoFn = Callable[[], None]


def _render_function_card(
    title: str, func: Callable[..., Any], demo_fn: DemoFn, notes: str | None = None
) -> None:
    signature = str(inspect.signature(func))
    identifier = f"{func.__module__}.{func.__qualname__}"
    with st.expander(title, expanded=False):
        st.code(f"{identifier}{signature}")
        st.write(inspect.getdoc(func) or "No docstring available.")
        if notes:
            st.caption(notes)
        run_demo = st.checkbox("Run demo", key=f"demo-{identifier}")
        if run_demo:
            demo_fn()


@st.cache_resource
def _demo_encoder_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    lat = torch.linspace(-1.2, 1.2, 4)
    lon = torch.linspace(-2.0, 2.0, 4)
    x = torch.randn(1, 2, lat.numel(), lon.numel())
    return x, lat, lon


@st.cache_resource
def _demo_mesh_vertices() -> torch.Tensor:
    encoder = GaiaGridEncoder(input_channels=2, hidden_dim=8, subdivisions=0)
    return encoder.mesh_vertices.clone()


@st.cache_resource
def _demo_gaia_model() -> GaiaModel:
    config = GaiaConfig(
        input_channels=2, output_channels=2, hidden_dim=16, mesh_subdivisions=0
    )
    return GaiaModel(config)


st.markdown("## 🛰️ GAIA Core Model")

col1, col2 = st.columns(2)

with col1:
    _render_function_card(
        "_icosahedron()",
        _icosahedron,
        demo_fn=lambda: _demo_icosahedron(),
    )

with col2:
    _render_function_card(
        "_subdivide(verts, faces)",
        _subdivide,
        demo_fn=lambda: _demo_subdivide(),
    )


def _demo_icosahedron() -> None:
    verts, faces = _icosahedron()
    st.metric("Vertices", verts.shape[0])
    st.metric("Faces", faces.shape[0])
    st.dataframe(pd.DataFrame(verts.numpy(), columns=["x", "y", "z"]).head())


def _demo_subdivide() -> None:
    verts, faces = _icosahedron()
    new_verts, new_faces = _subdivide(verts, faces)
    st.metric("Vertices", new_verts.shape[0])
    st.metric("Faces", new_faces.shape[0])
    st.dataframe(pd.DataFrame(new_verts.numpy(), columns=["x", "y", "z"]).head())


st.markdown("### Encoder")

_render_function_card(
    "GaiaGridEncoder._grid_positions(lat, lon)",
    GaiaGridEncoder._grid_positions,
    demo_fn=lambda: _demo_grid_positions(),
)


def _demo_grid_positions() -> None:
    _, lat, lon = _demo_encoder_inputs()
    positions = GaiaGridEncoder._grid_positions(lat, lon)
    st.write(f"Grid positions shape: {tuple(positions.shape)}")
    st.dataframe(
        pd.DataFrame(positions.reshape(-1, 3).numpy(), columns=["x", "y", "z"]).head()
    )


_render_function_card(
    "GaiaGridEncoder._tokenize(x, lat, lon)",
    GaiaGridEncoder._tokenize,
    demo_fn=lambda: _demo_tokenize(),
    notes="Uses the default token grid; output is tokens and mesh positions.",
)


def _demo_tokenize() -> None:
    x, lat, lon = _demo_encoder_inputs()
    encoder = GaiaGridEncoder(input_channels=2, hidden_dim=8, subdivisions=0)
    tokens, positions = encoder._tokenize(x, lat, lon)
    st.write(f"Tokens shape: {tuple(tokens.shape)}")
    st.write(f"Positions shape: {tuple(positions.shape)}")
    st.dataframe(pd.DataFrame(tokens[0].numpy()).head())


_render_function_card(
    "GaiaGridEncoder.forward(x, lat, lon)",
    GaiaGridEncoder.forward,
    demo_fn=lambda: _demo_encoder_forward(),
)


def _demo_encoder_forward() -> None:
    x, lat, lon = _demo_encoder_inputs()
    encoder = GaiaGridEncoder(input_channels=2, hidden_dim=8, subdivisions=0)
    features, mesh = encoder(x, lat, lon)
    st.write(f"Mesh features shape: {tuple(features.shape)}")
    st.write(f"Mesh vertices shape: {tuple(mesh.shape)}")
    st.dataframe(pd.DataFrame(features[0].detach().numpy()).head())


st.markdown("### Processor")

_render_function_card(
    "GaiaProcessor._build_graph(mesh_vertices, knn_k)",
    GaiaProcessor._build_graph,
    demo_fn=lambda: _demo_build_graph(),
)


def _demo_build_graph() -> None:
    mesh_vertices = _demo_mesh_vertices()
    knn_idx, order_idx, lat = GaiaProcessor._build_graph(mesh_vertices, knn_k=4)
    st.write(f"knn_idx shape: {tuple(knn_idx.shape)}")
    st.write(f"order_idx shape: {tuple(order_idx.shape)}")
    st.write(f"lat shape: {tuple(lat.shape)}")
    st.dataframe(pd.DataFrame(knn_idx[:5].numpy()))


_render_function_card(
    "LatAwareBias.forward(lat)",
    LatAwareBias.forward,
    demo_fn=lambda: _demo_lat_bias(),
)


def _demo_lat_bias() -> None:
    mesh_vertices = _demo_mesh_vertices()
    _, _, lat = GaiaProcessor._build_graph(mesh_vertices, knn_k=4)
    bias = LatAwareBias(hidden_dim=8)(lat)
    st.write(f"Bias shape: {tuple(bias.shape)}")
    st.dataframe(pd.DataFrame(bias.squeeze(0).detach().numpy()).head())


_render_function_card(
    "GNNBlock.forward(x, lat_bias)",
    GNNBlock.forward,
    demo_fn=lambda: _demo_gnn_block(),
)


def _demo_gnn_block() -> None:
    mesh_vertices = _demo_mesh_vertices()
    knn_idx, _, lat = GaiaProcessor._build_graph(mesh_vertices, knn_k=4)
    block = GNNBlock(hidden_dim=8, knn_idx=knn_idx, dropout=0.0)
    x = torch.randn(1, mesh_vertices.shape[0], 8)
    lat_bias = LatAwareBias(8)(lat).expand_as(x)
    out = block(x, lat_bias)
    st.write(f"GNN output shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0].detach().numpy()).head())


_render_function_card(
    "SpectralMixingBlock.forward(x, lat_bias)",
    SpectralMixingBlock.forward,
    demo_fn=lambda: _demo_spectral_block(),
)


def _demo_spectral_block() -> None:
    mesh_vertices = _demo_mesh_vertices()
    _, order_idx, lat = GaiaProcessor._build_graph(mesh_vertices, knn_k=4)
    block = SpectralMixingBlock(hidden_dim=8, order_idx=order_idx, dropout=0.0)
    x = torch.randn(1, mesh_vertices.shape[0], 8)
    lat_bias = LatAwareBias(8)(lat).expand_as(x)
    out = block(x, lat_bias)
    st.write(f"Output shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0].detach().numpy()).head())


_render_function_card(
    "GaiaProcessor.forward(x, mesh_vertices)",
    GaiaProcessor.forward,
    demo_fn=lambda: _demo_processor_forward(),
)


def _demo_processor_forward() -> None:
    mesh_vertices = _demo_mesh_vertices()
    processor = GaiaProcessor(
        hidden_dim=8, mesh_vertices=mesh_vertices, config=ProcessorConfig(num_blocks=2)
    )
    x = torch.randn(1, mesh_vertices.shape[0], 8)
    out = processor(x, mesh_vertices)
    st.write(f"Processed shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0].detach().numpy()).head())


st.markdown("### Decoder")

_render_function_card(
    "ConditionalDiffusionDecoder.forward(x)",
    ConditionalDiffusionDecoder.forward,
    demo_fn=lambda: _demo_diffusion_decoder(),
)


def _demo_diffusion_decoder() -> None:
    decoder = ConditionalDiffusionDecoder(hidden_dim=8, output_channels=2, steps=3)
    x = torch.randn(1, 6, 8)
    out = decoder(x)
    st.write(f"Output shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0].detach().numpy()).head())


_render_function_card(
    "EnsembleDecoder.forward(x)",
    EnsembleDecoder.forward,
    demo_fn=lambda: _demo_ensemble_decoder(),
)


def _demo_ensemble_decoder() -> None:
    decoder = EnsembleDecoder(hidden_dim=8, output_channels=2, members=3)
    x = torch.randn(1, 6, 8)
    out = decoder(x)
    st.write(f"Ensemble output shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0, 0].detach().numpy()).head())


_render_function_card(
    "GaiaDecoder.forward(x)",
    GaiaDecoder.forward,
    demo_fn=lambda: _demo_gaia_decoder(),
)


def _demo_gaia_decoder() -> None:
    decoder = GaiaDecoder(
        hidden_dim=8,
        output_channels=2,
        config=DecoderConfig(mode="diffusion", diffusion_steps=2),
    )
    x = torch.randn(1, 6, 8)
    out = decoder(x)
    st.write(f"Decoder output shape: {tuple(out.shape)}")
    st.dataframe(pd.DataFrame(out[0].detach().numpy()).head())


st.markdown("### Constraints & Sampling")

_render_function_card(
    "Constraint.apply(tensor)",
    Constraint.apply,
    demo_fn=lambda: _demo_constraint_base(),
    notes="Base class raises NotImplementedError.",
)


def _demo_constraint_base() -> None:
    try:
        Constraint().apply(torch.randn(1, 2, 2, 2))
    except Exception as exc:  # noqa: BLE001 - demo intentionally catches base error
        st.error(f"Expected error: {exc}")


_render_function_card(
    "NonNegativeConstraint.apply(tensor)",
    NonNegativeConstraint.apply,
    demo_fn=lambda: _demo_non_negative(),
)


def _demo_non_negative() -> None:
    tensor = torch.tensor([[[-1.0, 2.0], [3.0, -4.0]]]).unsqueeze(0)
    constraint = NonNegativeConstraint(channels=[0])
    constrained = constraint.apply(tensor)
    st.dataframe(pd.DataFrame(constrained[0, 0].numpy()))


_render_function_card(
    "RangeConstraint.apply(tensor)",
    RangeConstraint.apply,
    demo_fn=lambda: _demo_range_constraint(),
)


def _demo_range_constraint() -> None:
    tensor = torch.tensor([[[5.0, -1.0], [0.5, 2.5]]]).unsqueeze(0)
    constraint = RangeConstraint(channels=[0], min_value=0.0, max_value=2.0)
    constrained = constraint.apply(tensor)
    st.dataframe(pd.DataFrame(constrained[0, 0].numpy()))


_render_function_card(
    "MeanPreservingConstraint.apply(tensor)",
    MeanPreservingConstraint.apply,
    demo_fn=lambda: _demo_mean_constraint(),
)


def _demo_mean_constraint() -> None:
    tensor = torch.randn(2, 1, 3, 3)
    constraint = MeanPreservingConstraint(channels=[0])
    constrained = constraint.apply(tensor)
    st.metric("Mean before", float(tensor[:, 0].mean()))
    st.metric("Mean after", float(constrained[:, 0].mean()))


_render_function_card(
    "ConstraintApplier.apply(tensor)",
    ConstraintApplier.apply,
    demo_fn=lambda: _demo_constraint_applier(),
)


def _demo_constraint_applier() -> None:
    tensor = torch.tensor([[[5.0, -1.0], [0.5, 2.5]]]).unsqueeze(0)
    applier = ConstraintApplier(
        [
            NonNegativeConstraint(channels=[0]),
            RangeConstraint(channels=[0], min_value=0.0, max_value=2.0),
        ]
    )
    constrained = applier.apply(tensor)
    st.dataframe(pd.DataFrame(constrained[0, 0].numpy()))


_render_function_card(
    "gaia_sample(model, inputs, lat, lon)",
    gaia_sample,
    demo_fn=lambda: _demo_gaia_sample(),
)


def _demo_gaia_sample() -> None:
    model = _demo_gaia_model()
    x, lat, lon = _demo_encoder_inputs()
    outputs = gaia_sample(model, x, lat, lon)
    st.write(f"Sample output shape: {tuple(outputs.shape)}")
    st.dataframe(pd.DataFrame(outputs[0, 0].detach().numpy()).head())


st.markdown("### GaiaModel")

_render_function_card(
    "GaiaModel._grid_positions(lat, lon)",
    GaiaModel._grid_positions,
    demo_fn=lambda: _demo_model_grid_positions(),
)


def _demo_model_grid_positions() -> None:
    _, lat, lon = _demo_encoder_inputs()
    positions = GaiaModel._grid_positions(lat, lon)
    st.write(f"Positions shape: {tuple(positions.shape)}")
    st.dataframe(pd.DataFrame(positions.reshape(-1, 3).numpy()).head())


_render_function_card(
    "GaiaModel._mesh_to_grid(mesh_output, mesh_vertices, lat, lon)",
    GaiaModel._mesh_to_grid,
    demo_fn=lambda: _demo_mesh_to_grid(),
)


def _demo_mesh_to_grid() -> None:
    model = _demo_gaia_model()
    mesh_vertices = model.encoder.mesh_vertices
    x = torch.randn(1, mesh_vertices.shape[0], model.config.output_channels)
    _, lat, lon = _demo_encoder_inputs()
    grid = model._mesh_to_grid(x, mesh_vertices, lat, lon)
    st.write(f"Grid output shape: {tuple(grid.shape)}")
    st.dataframe(pd.DataFrame(grid[0, 0].detach().numpy()))


_render_function_card(
    "GaiaModel.forward(x, lat, lon)",
    GaiaModel.forward,
    demo_fn=lambda: _demo_model_forward(),
)


def _demo_model_forward() -> None:
    model = _demo_gaia_model()
    x, lat, lon = _demo_encoder_inputs()
    outputs = model(x, lat, lon)
    st.write(f"Model output shape: {tuple(outputs.shape)}")
    st.dataframe(pd.DataFrame(outputs[0, 0].detach().numpy()).head())


st.markdown("## 📦 GAIA Data Pipeline")

st.markdown("### Schema")

_render_function_card(
    "GaiaVariableSchema.from_default()",
    GaiaVariableSchema.from_default,
    demo_fn=lambda: _demo_schema_from_default(),
)


def _demo_schema_from_default() -> None:
    schema = GaiaVariableSchema.from_default()
    st.metric("Variables", len(schema.variables))
    st.dataframe(
        pd.DataFrame(
            list(schema.variables.items())[:5], columns=["variable", "metadata"]
        )
    )


_render_function_card(
    "GaiaVariableSchema.from_file(schema_path)",
    GaiaVariableSchema.from_file,
    demo_fn=lambda: _demo_schema_from_file(),
)


def _demo_schema_from_file() -> None:
    schema = GaiaVariableSchema.from_default()
    with tempfile.TemporaryDirectory() as tmpdir:
        sample_path = f"{tmpdir}/schema.yaml"
        Path(sample_path).write_text(
            "variables:\n  temp:\n    units: K\n", encoding="utf-8"
        )
        loaded = GaiaVariableSchema.from_file(sample_path)
    st.write("Loaded variables:")
    st.dataframe(
        pd.DataFrame(list(loaded.variables.items()), columns=["variable", "metadata"])
    )
    st.write(f"Default schema variables: {len(schema.variables)}")


_render_function_card(
    "GaiaVariableSchema.validate(metadata)",
    GaiaVariableSchema.validate,
    demo_fn=lambda: _demo_schema_validate(),
)


def _demo_schema_validate() -> None:
    schema = GaiaVariableSchema.from_default()
    metadata = {
        "variables": {
            name: {
                "units": info.get("units"),
                "levels": info.get("levels"),
            }
            for name, info in schema.variables.items()
        }
    }
    try:
        schema.validate(metadata)
        st.success("Schema validation passed for full metadata set.")
    except Exception as exc:  # noqa: BLE001 - demo shows validation failures
        st.error(f"Validation error: {exc}")


st.markdown("### Normalization")

_render_function_card(
    "NormalizationStats.compute(dataset, variables, split, version)",
    NormalizationStats.compute,
    demo_fn=lambda: _demo_norm_compute(),
)


def _demo_norm_compute() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(
        dataset, ["temp", "wind"], split="train", version="1"
    )
    st.json(stats.to_dict())


_render_function_card(
    "NormalizationStats.to_dict()",
    NormalizationStats.to_dict,
    demo_fn=lambda: _demo_norm_to_dict(),
)


def _demo_norm_to_dict() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(dataset, ["temp"], split="train", version="1")
    st.json(stats.to_dict())


_render_function_card(
    "NormalizationStats.save(output_dir)",
    NormalizationStats.save,
    demo_fn=lambda: _demo_norm_save(),
)


def _demo_norm_save() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(dataset, ["temp"], split="train", version="1")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = stats.save(tmpdir)
        st.write(f"Saved to: {path}")
        st.text(path.read_text(encoding="utf-8"))


_render_function_card(
    "NormalizationStats.load(artifact_path)",
    NormalizationStats.load,
    demo_fn=lambda: _demo_norm_load(),
)


def _demo_norm_load() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(dataset, ["temp"], split="train", version="1")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = stats.save(tmpdir)
        loaded = NormalizationStats.load(path)
        st.json(loaded.to_dict())


_render_function_card(
    "NormalizationStats.apply(dataset)",
    NormalizationStats.apply,
    demo_fn=lambda: _demo_norm_apply(),
)


def _demo_norm_apply() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(dataset, ["temp"], split="train", version="1")
    normalized = stats.apply(dataset)
    st.dataframe(normalized["temp"].to_pandas().head())


_render_function_card(
    "NormalizationStats.denormalize(dataset)",
    NormalizationStats.denormalize,
    demo_fn=lambda: _demo_norm_denormalize(),
)


def _demo_norm_denormalize() -> None:
    dataset = _demo_dataset()
    stats = NormalizationStats.compute(dataset, ["temp"], split="train", version="1")
    normalized = stats.apply(dataset)
    restored = stats.denormalize(normalized)
    st.dataframe(restored["temp"].to_pandas().head())


st.markdown("### Regridding")

_render_function_card(
    "regrid_dataset(dataset, target_grid, strategy)",
    regrid_dataset,
    demo_fn=lambda: _demo_regrid_dataset(),
)


def _demo_regrid_dataset() -> None:
    dataset, target_grid = _demo_regrid_inputs()
    regridded = regrid_dataset(dataset, target_grid, RegridStrategy.BILINEAR)
    st.dataframe(regridded["temp"].to_pandas())


_render_function_card(
    "_bilinear_regrid(dataset, target_grid, variable_names)",
    _bilinear_regrid,
    demo_fn=lambda: _demo_bilinear_regrid(),
)


def _demo_bilinear_regrid() -> None:
    dataset, target_grid = _demo_regrid_inputs()
    regridded = _bilinear_regrid(dataset, target_grid, ["temp", "wind"])
    st.dataframe(regridded["temp"].to_pandas())


_render_function_card(
    "_conservative_regrid(dataset, target_grid, variable_names)",
    _conservative_regrid,
    demo_fn=lambda: _demo_conservative_regrid(),
)


def _demo_conservative_regrid() -> None:
    dataset, target_grid = _demo_regrid_inputs(conservative=True)
    regridded = _conservative_regrid(dataset, target_grid, ["temp"])
    st.dataframe(regridded["temp"].to_pandas())


_render_function_card(
    "_coarsen_factor(source, target)",
    _coarsen_factor,
    demo_fn=lambda: _demo_coarsen_factor(),
)


def _demo_coarsen_factor() -> None:
    source = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([0.5, 2.5])
    st.write(f"Coarsen factor: {_coarsen_factor(source, target)}")


_render_function_card(
    "_grid_step(values)",
    _grid_step,
    demo_fn=lambda: _demo_grid_step(),
)


def _demo_grid_step() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    st.write(f"Grid step: {_grid_step(values)}")


_render_function_card(
    "_validate_alignment(source, target, factor, label)",
    _validate_alignment,
    demo_fn=lambda: _demo_validate_alignment(),
)


def _demo_validate_alignment() -> None:
    source = np.array([0.0, 1.0, 2.0, 3.0])
    target = np.array([0.5, 2.5])
    try:
        _validate_alignment(
            source, target, factor=2, label="latitude", allow_block_mean=True
        )
        st.success("Alignment validated for block-mean target grid.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Alignment error: {exc}")


_render_function_card(
    "_latitude_weights(latitude)",
    _latitude_weights,
    demo_fn=lambda: _demo_latitude_weights(),
)


def _demo_latitude_weights() -> None:
    latitude = xr.DataArray(np.linspace(-90, 90, 5), dims=("latitude",))
    weights = _latitude_weights(latitude)
    st.dataframe(weights.to_pandas())


st.markdown("### Sharding")

_render_function_card(
    "ShardManifest.to_dict()",
    ShardManifest.to_dict,
    demo_fn=lambda: _demo_shard_to_dict(),
)


def _demo_shard_to_dict() -> None:
    manifest = ShardManifest(
        version="1",
        created_at="2023-01-01T00:00:00Z",
        time_dim="time",
        variables=("temp",),
        shards=(),
    )
    st.json(manifest.to_dict())


_render_function_card(
    "ShardManifest.save(output_dir)",
    ShardManifest.save,
    demo_fn=lambda: _demo_shard_save(),
)


def _demo_shard_save() -> None:
    manifest = ShardManifest(
        version="1",
        created_at="2023-01-01T00:00:00Z",
        time_dim="time",
        variables=("temp",),
        shards=(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = manifest.save(tmpdir)
        st.write(f"Saved manifest: {path}")
        st.text(path.read_text(encoding="utf-8"))


_render_function_card(
    "ShardManifest.load(manifest_path)",
    ShardManifest.load,
    demo_fn=lambda: _demo_shard_load(),
)


def _demo_shard_load() -> None:
    manifest = ShardManifest(
        version="1",
        created_at="2023-01-01T00:00:00Z",
        time_dim="time",
        variables=("temp",),
        shards=(),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = manifest.save(tmpdir)
        loaded = ShardManifest.load(path)
        st.json(loaded.to_dict())


_render_function_card(
    "write_time_shards(dataset, output_dir, shard_size)",
    write_time_shards,
    demo_fn=lambda: _demo_write_time_shards(),
    notes="Writes NetCDF shards; errors are caught for missing IO dependencies.",
)


def _demo_write_time_shards() -> None:
    dataset = _demo_dataset(time_dim="time")
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            manifest = write_time_shards(dataset, tmpdir, shard_size=2)
            st.json(manifest.to_dict())
        except Exception as exc:  # noqa: BLE001 - demo may fail without netCDF backend
            st.error(f"Sharding failed: {exc}")


_render_function_card(
    "_hash_file(path)",
    _hash_file,
    demo_fn=lambda: _demo_hash_file(),
)


def _demo_hash_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "demo.txt"
        path.write_text("gaia", encoding="utf-8")
        st.write(f"Hash: {_hash_file(path)}")


st.markdown("### Sources")

_render_function_card(
    "verify_access_credentials(env_vars, config_paths)",
    verify_access_credentials,
    demo_fn=lambda: _demo_verify_credentials(),
    notes="Expects CDS API credentials; demonstration reports missing credentials.",
)


def _demo_verify_credentials() -> None:
    try:
        verify_access_credentials(env_vars=("FAKE_VAR",), config_paths=())
        st.success("Credentials found.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Credential check error: {exc}")


_render_function_card(
    "_get_env(name)",
    _get_env,
    demo_fn=lambda: _demo_get_env(),
)


def _demo_get_env() -> None:
    st.write(f"PATH exists: {_get_env('PATH') is not None}")


_render_function_card(
    "ERA5ZarrSource.open_dataset()",
    ERA5ZarrSource.open_dataset,
    demo_fn=lambda: _demo_open_dataset(),
    notes="Uses a dummy store URL; exceptions are expected without credentials.",
)


def _demo_open_dataset() -> None:
    source = ERA5ZarrSource(store_url="s3://example/era5.zarr")
    try:
        source.open_dataset()
        st.success("Opened dataset successfully.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Open dataset error: {exc}")


st.markdown("## 📊 GAIA Evaluation")

_render_function_card(
    "_to_numpy(data)",
    _to_numpy,
    demo_fn=lambda: _demo_to_numpy(),
)


def _demo_to_numpy() -> None:
    arr = _to_numpy([1, 2, 3])
    st.write(arr)


_render_function_card(
    "_normalize_axis(axis, ndim)",
    _normalize_axis,
    demo_fn=lambda: _demo_normalize_axis(),
)


def _demo_normalize_axis() -> None:
    st.write(f"Axis -1 normalized in ndim 3: {_normalize_axis(-1, 3)}")


_render_function_card(
    "_validate_axis(axis, ndim, label)",
    _validate_axis,
    demo_fn=lambda: _demo_validate_axis(),
)


def _demo_validate_axis() -> None:
    try:
        _validate_axis(1, 3, "variable")
        st.success("Axis validated.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Axis error: {exc}")


_render_function_card(
    "_resolve_indices(indices, names, metadata_values, label)",
    _resolve_indices,
    demo_fn=lambda: _demo_resolve_indices(),
)


def _demo_resolve_indices() -> None:
    resolved = _resolve_indices(
        indices=None, names=["temp"], metadata_values=["temp", "wind"], label="variable"
    )
    st.write(f"Resolved indices: {resolved}")


_render_function_card(
    "apply_selection(data, config, metadata)",
    apply_selection,
    demo_fn=lambda: _demo_apply_selection(),
)


def _demo_apply_selection() -> None:
    data = np.random.randn(2, 3)
    config = EvaluationSelectionConfig(variable_axis=1, variable_indices=[0, 2])
    selected = apply_selection(data, config)
    st.dataframe(pd.DataFrame(selected))


_render_function_card(
    "_apply_selection_pair(pred, target, selection, metadata)",
    _apply_selection_pair,
    demo_fn=lambda: _demo_apply_selection_pair(),
)


def _demo_apply_selection_pair() -> None:
    pred = np.random.randn(2, 3)
    target = np.random.randn(2, 3)
    config = EvaluationSelectionConfig(variable_axis=1, variable_indices=[0])
    pred_sel, target_sel = _apply_selection_pair(pred, target, config, None)
    st.dataframe(pd.DataFrame({"pred": pred_sel.ravel(), "target": target_sel.ravel()}))


_render_function_card(
    "rmse(pred, target)",
    rmse,
    demo_fn=lambda: _demo_rmse(),
)


def _demo_rmse() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.5])
    st.metric("RMSE", rmse(pred, target))


_render_function_card(
    "mae(pred, target)",
    mae,
    demo_fn=lambda: _demo_mae(),
)


def _demo_mae() -> None:
    pred = np.array([1.0, 2.0, 3.0])
    target = np.array([1.5, 2.5, 2.5])
    st.metric("MAE", mae(pred, target))


_render_function_card(
    "acc(pred, target)",
    acc,
    demo_fn=lambda: _demo_acc(),
)


def _demo_acc() -> None:
    pred = np.random.randn(4, 3)
    target = pred + np.random.normal(scale=0.1, size=pred.shape)
    st.metric("ACC", acc(pred, target))


_render_function_card(
    "crps_ensemble(ensemble, observations)",
    crps_ensemble,
    demo_fn=lambda: _demo_crps_ensemble(),
)


def _demo_crps_ensemble() -> None:
    ensemble = np.random.randn(5, 3)
    obs = np.random.randn(3)
    st.metric("CRPS", crps_ensemble(ensemble, obs))


_render_function_card(
    "reliability_curve(probabilities, observations)",
    reliability_curve,
    demo_fn=lambda: _demo_reliability_curve(),
)


def _demo_reliability_curve() -> None:
    probs = np.random.rand(50)
    obs = (probs > 0.6).astype(float)
    result = reliability_curve(probs, obs, bins=5)
    df = pd.DataFrame(
        {
            "bin_start": result.bin_edges[:-1],
            "bin_end": result.bin_edges[1:],
            "forecast": result.forecast_probabilities,
            "observed": result.observed_frequencies,
            "count": result.counts,
        }
    )
    st.dataframe(df)


_render_function_card(
    "brier_score(probabilities, observations)",
    brier_score,
    demo_fn=lambda: _demo_brier_score(),
)


def _demo_brier_score() -> None:
    probs = np.random.rand(20)
    obs = (probs > 0.5).astype(float)
    st.metric("Brier score", brier_score(probs, obs))


st.markdown("## 🧪 GAIA Training")

_render_function_card(
    "losses.rmse(prediction, target)",
    gaia_losses.rmse,
    demo_fn=lambda: _demo_loss_rmse(),
)


def _demo_loss_rmse() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 2.5])
    st.metric("RMSE", float(gaia_losses.rmse(pred, target)))


_render_function_card(
    "losses.mae(prediction, target)",
    gaia_losses.mae,
    demo_fn=lambda: _demo_loss_mae(),
)


def _demo_loss_mae() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 2.5])
    st.metric("MAE", float(gaia_losses.mae(pred, target)))


_render_function_card(
    "losses.crps_ensemble(ensemble, target)",
    gaia_losses.crps_ensemble,
    demo_fn=lambda: _demo_loss_crps(),
)


def _demo_loss_crps() -> None:
    ensemble = torch.randn(5, 3)
    target = torch.randn(3)
    st.metric("CRPS", float(gaia_losses.crps_ensemble(ensemble, target)))


_render_function_card(
    "losses.spectral_crps(ensemble, target)",
    gaia_losses.spectral_crps,
    demo_fn=lambda: _demo_loss_spectral_crps(),
)


def _demo_loss_spectral_crps() -> None:
    ensemble = torch.randn(5, 8)
    target = torch.randn(8)
    st.metric("Spectral CRPS", float(gaia_losses.spectral_crps(ensemble, target)))


_render_function_card(
    "losses.weighted_loss_sum(losses, weights)",
    gaia_losses.weighted_loss_sum,
    demo_fn=lambda: _demo_weighted_loss_sum(),
)


def _demo_weighted_loss_sum() -> None:
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    weights = [0.25, 0.75]
    total = gaia_losses.weighted_loss_sum(losses, weights)
    st.metric("Weighted sum", float(total))


_render_function_card(
    "linear_rollout_schedule(step, start, end, total_steps)",
    linear_rollout_schedule,
    demo_fn=lambda: _demo_linear_schedule(),
)


def _demo_linear_schedule() -> None:
    st.write(f"Horizon at step 5/10: {linear_rollout_schedule(5, 1, 8, 10)}")


_render_function_card(
    "RolloutCurriculum.horizon_for_step(step)",
    RolloutCurriculum.horizon_for_step,
    demo_fn=lambda: _demo_curriculum_horizon(),
)


def _demo_curriculum_horizon() -> None:
    curriculum = RolloutCurriculum(start_horizon=1, end_horizon=8, total_steps=10)
    st.write(f"Horizon at step 7: {curriculum.horizon_for_step(7)}")


_render_function_card(
    "finetune.autoregressive_rollout_loss(model, context, targets, horizon)",
    gaia_finetune.autoregressive_rollout_loss,
    demo_fn=lambda: _demo_autoregressive_loss(),
)


def _demo_autoregressive_loss() -> None:
    model = _demo_simple_model()
    context = torch.randn(2, 3, 4)
    targets = torch.randn(2, 5, 4)
    loss = gaia_finetune.autoregressive_rollout_loss(model, context, targets, horizon=3)
    st.metric("Rollout loss", float(loss))


_render_function_card(
    "finetune.finetune_step(model, batch, curriculum, global_step)",
    gaia_finetune.finetune_step,
    demo_fn=lambda: _demo_finetune_step(),
)


def _demo_finetune_step() -> None:
    model = _demo_simple_model()
    context = torch.randn(2, 3, 4)
    targets = torch.randn(2, 5, 4)
    batch = gaia_finetune.FinetuneBatch(context=context, targets=targets)
    curriculum = RolloutCurriculum(start_horizon=1, end_horizon=4, total_steps=10)
    loss, metrics = gaia_finetune.finetune_step(model, batch, curriculum, global_step=5)
    st.metric("Loss", float(loss))
    st.json(metrics)


_render_function_card(
    "pretrain.apply_variable_mask(inputs, mask_ratio)",
    gaia_pretrain.apply_variable_mask,
    demo_fn=lambda: _demo_apply_variable_mask(),
)


def _demo_apply_variable_mask() -> None:
    inputs = torch.randn(2, 3)
    masked, mask = gaia_pretrain.apply_variable_mask(inputs, mask_ratio=0.3)
    st.dataframe(pd.DataFrame(masked.numpy()))
    st.dataframe(pd.DataFrame(mask.numpy().astype(int)))


_render_function_card(
    "pretrain.masked_reconstruction_loss(model, inputs, mask)",
    gaia_pretrain.masked_reconstruction_loss,
    demo_fn=lambda: _demo_masked_reconstruction_loss(),
)


def _demo_masked_reconstruction_loss() -> None:
    model = _demo_simple_model()
    inputs = torch.randn(2, 4)
    masked, mask = gaia_pretrain.apply_variable_mask(inputs, mask_ratio=0.5)
    loss = gaia_pretrain.masked_reconstruction_loss(model, masked, mask)
    st.metric("Reconstruction loss", float(loss))


_render_function_card(
    "pretrain.temporal_ordering_loss(model, sequence)",
    gaia_pretrain.temporal_ordering_loss,
    demo_fn=lambda: _demo_temporal_ordering_loss(),
)


def _demo_temporal_ordering_loss() -> None:
    model = _demo_temporal_model()
    sequence = torch.randn(4, 3, 6)
    loss = gaia_pretrain.temporal_ordering_loss(model, sequence)
    st.metric("Ordering loss", float(loss))


_render_function_card(
    "pretrain.pretrain_step(model, batch, mask_ratio, ordering_weight)",
    gaia_pretrain.pretrain_step,
    demo_fn=lambda: _demo_pretrain_step(),
)


def _demo_pretrain_step() -> None:
    model = _demo_temporal_model()
    batch = gaia_pretrain.PretrainBatch(inputs=torch.randn(4, 3, 6))
    loss, metrics = gaia_pretrain.pretrain_step(
        model, batch, mask_ratio=0.2, ordering_weight=0.5
    )
    st.metric("Total loss", float(loss))
    st.json(metrics)


_render_function_card(
    "calibrate.calibration_step(model, batch, spectral_weight)",
    gaia_calibrate.calibration_step,
    demo_fn=lambda: _demo_calibration_step(),
)


def _demo_calibration_step() -> None:
    model = _demo_ensemble_model()
    batch = gaia_calibrate.CalibrationBatch(
        context=torch.randn(2, 3), targets=torch.randn(2, 3)
    )
    loss, metrics = gaia_calibrate.calibration_step(model, batch, spectral_weight=0.1)
    st.metric("Calibration loss", float(loss))
    st.json(metrics)


st.markdown("## 🧩 Demo Helpers")

st.caption("These helper objects power the GAIA demos above.")


@st.cache_resource
def _demo_dataset(time_dim: str = "time") -> xr.Dataset:
    data = xr.Dataset(
        {
            "temp": ("time", np.linspace(270, 280, 5)),
            "wind": ("time", np.linspace(5, 9, 5)),
        },
        coords={time_dim: np.arange(5)},
    )
    return data


@st.cache_resource
def _demo_regrid_inputs(conservative: bool = False) -> tuple[xr.Dataset, TargetGrid]:
    lat = np.array([0.0, 1.0, 2.0, 3.0])
    lon = np.array([10.0, 11.0, 12.0, 13.0])
    data = xr.Dataset(
        {
            "temp": (("latitude", "longitude"), np.arange(16).reshape(4, 4)),
            "wind": (("latitude", "longitude"), np.arange(16).reshape(4, 4) * 0.1),
        },
        coords={"latitude": lat, "longitude": lon},
    )
    if conservative:
        target_lat = np.array([0.5, 2.5])
        target_lon = np.array([10.5, 12.5])
    else:
        target_lat = np.linspace(0.0, 3.0, 3)
        target_lon = np.linspace(10.0, 13.0, 3)
    return data, TargetGrid(latitude=target_lat, longitude=target_lon)


class _DemoModel(torch.nn.Module):
    def __init__(self, features: int = 4) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(features, features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _DemoTemporalModel(torch.nn.Module):
    def __init__(self, features: int = 6) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(features, features)
        self.classifier = torch.nn.Linear(features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    def temporal_ordering_logits(self, sequence: torch.Tensor) -> torch.Tensor:
        pooled = sequence.mean(dim=1)
        return self.classifier(pooled)


class _DemoEnsembleModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, features = x.shape
        ensemble = torch.randn(3, batch, features)
        return ensemble


@st.cache_resource
def _demo_simple_model() -> _DemoModel:
    return _DemoModel()


@st.cache_resource
def _demo_temporal_model() -> _DemoTemporalModel:
    return _DemoTemporalModel()


@st.cache_resource
def _demo_ensemble_model() -> _DemoEnsembleModel:
    return _DemoEnsembleModel()


st.write("Helper classes: ")
st.json(
    {
        "_DemoModel": {"type": "Linear projection"},
        "_DemoTemporalModel": {"type": "Temporal ordering classifier"},
        "_DemoEnsembleModel": {"type": "Random ensemble generator"},
    }
)
