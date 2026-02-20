import { useState, useCallback } from 'react';
import type { LatLng, WeatherModel, MapLayer, ViewMode } from './types/weather';
import Sidebar from './components/Sidebar';
import MapView from './components/MapView';
import TimeSlider from './components/TimeSlider';
import ForecastPanel from './components/ForecastPanel';
import Meteogram from './components/Meteogram';
import SkewT from './components/SkewT';
import LocationSearch from './components/LocationSearch';
import LayerControl from './components/LayerControl';
import ModelComparison from './components/ModelComparison';
import StormTracker from './components/StormTracker';
import './App.css';

type Panel = 'none' | 'forecast' | 'meteogram' | 'sounding' | 'comparison' | 'tropical';

export default function App() {
  // Navigation & layout
  const [viewMode, setViewMode] = useState<ViewMode>('map');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Map state
  const [activeLayers, setActiveLayers] = useState<MapLayer[]>(['radar']);
  const [selectedLocation, setSelectedLocation] = useState<LatLng | null>(null);
  const [locationName, setLocationName] = useState<string>('');
  const [selectedModel, setSelectedModel] = useState<WeatherModel>('best_match');

  // Time & animation
  const [forecastHour, setForecastHour] = useState(0);
  const [radarFrameCount, setRadarFrameCount] = useState(0);
  const [satelliteFrameCount, setSatelliteFrameCount] = useState(0);
  const [radarFrame, setRadarFrame] = useState(-1);
  const [satelliteFrame, setSatelliteFrame] = useState(-1);

  // Panel
  const [activePanel, setActivePanel] = useState<Panel>('none');

  const handleMapClick = useCallback((latlng: LatLng) => {
    setSelectedLocation(latlng);
    setLocationName(`${latlng.lat.toFixed(3)}, ${latlng.lng.toFixed(3)}`);
    setActivePanel('forecast');
  }, []);

  const handleLocationSelect = useCallback((latlng: LatLng, name: string) => {
    setSelectedLocation(latlng);
    setLocationName(name);
    setActivePanel('forecast');
  }, []);

  const handleLayerToggle = useCallback((layer: MapLayer) => {
    setActiveLayers((prev) =>
      prev.includes(layer) ? prev.filter((l) => l !== layer) : [...prev, layer],
    );
  }, []);

  const handleNavigate = useCallback((view: ViewMode) => {
    setViewMode(view);
    switch (view) {
      case 'radar':
        setActiveLayers((prev) => prev.includes('radar') ? prev : [...prev, 'radar']);
        break;
      case 'satellite':
        setActiveLayers((prev) => prev.includes('satellite') ? prev : [...prev, 'satellite']);
        break;
      case 'tropical':
        setActivePanel('tropical');
        break;
      case 'soundings':
        if (selectedLocation) setActivePanel('sounding');
        break;
      case 'models':
        if (selectedLocation) setActivePanel('comparison');
        break;
      default:
        break;
    }
  }, [selectedLocation]);

  const showRadar = activeLayers.includes('radar');
  const showSatellite = activeLayers.includes('satellite');

  const isOverlayMode = showRadar || showSatellite;
  const timeMax = showRadar
    ? radarFrameCount - 1
    : showSatellite
      ? satelliteFrameCount - 1
      : 168;
  const timeValue = showRadar
    ? (radarFrame < 0 ? Math.max(0, radarFrameCount - 1) : radarFrame)
    : showSatellite
      ? (satelliteFrame < 0 ? Math.max(0, satelliteFrameCount - 1) : satelliteFrame)
      : forecastHour;

  const handleTimeChange = useCallback(
    (val: number) => {
      if (showRadar) setRadarFrame(val);
      else if (showSatellite) setSatelliteFrame(val);
      else setForecastHour(val);
    },
    [showRadar, showSatellite],
  );

  const formatTimeLabel = useCallback(
    (val: number) => {
      if (isOverlayMode) return `Frame ${val + 1} / ${timeMax + 1}`;
      return `F${String(val).padStart(3, '0')}`;
    },
    [isOverlayMode, timeMax],
  );

  return (
    <div className="app">
      <Sidebar
        currentView={viewMode}
        onNavigate={handleNavigate}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      <div className={`app__main ${sidebarCollapsed ? 'app__main--expanded' : ''}`}>
        {/* Top bar */}
        <div className="top-bar">
          <LocationSearch onSelect={handleLocationSelect} />

          <div className="top-bar__center">
            <TimeSlider
              value={timeValue}
              max={timeMax > 0 ? timeMax : 1}
              onChange={handleTimeChange}
              label={showRadar ? 'Radar' : showSatellite ? 'Satellite' : 'Forecast'}
              formatLabel={formatTimeLabel}
            />
          </div>

          <div className="top-bar__right">
            <LayerControl activeLayers={activeLayers} onToggle={handleLayerToggle} />
          </div>
        </div>

        {/* Map */}
        <div className="map-wrapper">
          <MapView
            showRadar={showRadar}
            showSatellite={showSatellite}
            radarFrame={radarFrame}
            satelliteFrame={satelliteFrame}
            onMapClick={handleMapClick}
            onRadarFramesLoaded={setRadarFrameCount}
            onSatelliteFramesLoaded={setSatelliteFrameCount}
            selectedLocation={selectedLocation}
          />

          {selectedLocation && (
            <div className="map-info">
              <span className="map-info__location">{locationName}</span>
              <span className="map-info__coords">
                {selectedLocation.lat.toFixed(4)}N, {selectedLocation.lng.toFixed(4)}E
              </span>
            </div>
          )}

          {selectedLocation && activePanel === 'none' && (
            <div className="quick-actions">
              <button className="quick-action" onClick={() => setActivePanel('forecast')}>
                Forecast
              </button>
              <button className="quick-action" onClick={() => setActivePanel('meteogram')}>
                Meteogram
              </button>
              <button className="quick-action" onClick={() => setActivePanel('sounding')}>
                Sounding
              </button>
              <button className="quick-action" onClick={() => setActivePanel('comparison')}>
                Compare Models
              </button>
            </div>
          )}
        </div>

        {/* Panels */}
        {activePanel === 'forecast' && selectedLocation && (
          <ForecastPanel
            location={selectedLocation}
            model={selectedModel}
            forecastHour={forecastHour}
            onModelChange={setSelectedModel}
            onClose={() => setActivePanel('none')}
            onOpenMeteogram={() => setActivePanel('meteogram')}
            onOpenSounding={() => setActivePanel('sounding')}
          />
        )}

        {activePanel === 'meteogram' && selectedLocation && (
          <Meteogram
            location={selectedLocation}
            model={selectedModel}
            onClose={() => setActivePanel('none')}
          />
        )}

        {activePanel === 'sounding' && selectedLocation && (
          <SkewT
            location={selectedLocation}
            model={selectedModel}
            forecastHour={forecastHour}
            onClose={() => setActivePanel('none')}
          />
        )}

        {activePanel === 'comparison' && selectedLocation && (
          <ModelComparison
            location={selectedLocation}
            onClose={() => setActivePanel('none')}
          />
        )}

        {activePanel === 'tropical' && (
          <StormTracker
            onClose={() => setActivePanel('none')}
            onSelectStorm={(lat, lon) => {
              setSelectedLocation({ lat, lng: lon });
              setLocationName(`Storm @ ${lat.toFixed(1)}N, ${Math.abs(lon).toFixed(1)}${lon < 0 ? 'W' : 'E'}`);
            }}
          />
        )}
      </div>
    </div>
  );
}
