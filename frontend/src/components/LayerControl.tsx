import type { MapLayer } from '../types/weather';

interface LayerControlProps {
  activeLayers: MapLayer[];
  onToggle: (layer: MapLayer) => void;
}

interface LayerDef {
  id: MapLayer;
  label: string;
  color: string;
}

const LAYERS: LayerDef[] = [
  { id: 'radar',         label: 'Radar',     color: '#3b82f6' },
  { id: 'satellite',     label: 'Satellite',  color: '#6366f1' },
];

export default function LayerControl({ activeLayers, onToggle }: LayerControlProps) {
  return (
    <div className="layer-control">
      <div className="layer-control__title">Layers</div>
      {LAYERS.map((layer) => {
        const active = activeLayers.includes(layer.id);
        return (
          <button
            key={layer.id}
            className={`layer-control__item ${active ? 'layer-control__item--active' : ''}`}
            onClick={() => onToggle(layer.id)}
          >
            <span
              className="layer-control__dot"
              style={{ backgroundColor: active ? layer.color : 'transparent', borderColor: layer.color }}
            />
            <span className="layer-control__label">{layer.label}</span>
          </button>
        );
      })}
    </div>
  );
}
