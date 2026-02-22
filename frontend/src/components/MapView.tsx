import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { LatLng, RainViewerFrame } from '../types/weather';
import {
  fetchRainViewerData,
  radarTileUrl,
  satelliteTileUrl,
} from '../api/rainViewer';

// Fix default marker icon issue with bundlers
import iconUrl from 'leaflet/dist/images/marker-icon.png';
import iconRetinaUrl from 'leaflet/dist/images/marker-icon-2x.png';
import shadowUrl from 'leaflet/dist/images/marker-shadow.png';

L.Icon.Default.mergeOptions({ iconUrl, iconRetinaUrl, shadowUrl });

interface MapViewProps {
  center?: LatLng;
  zoom?: number;
  showRadar: boolean;
  showSatellite: boolean;
  radarFrame?: number;       // index into available frames
  satelliteFrame?: number;
  onMapClick?: (latlng: LatLng) => void;
  onRadarFramesLoaded?: (count: number) => void;
  onSatelliteFramesLoaded?: (count: number) => void;
  selectedLocation?: LatLng | null;
  baseMapStyle?: string;
}

const DARK_TILES = 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png';
const DARK_ATTR = '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>';

export default function MapView({
  center = { lat: 39.0, lng: -98.0 },
  zoom = 4,
  showRadar,
  showSatellite,
  radarFrame = -1,
  satelliteFrame = -1,
  onMapClick,
  onRadarFramesLoaded,
  onSatelliteFramesLoaded,
  selectedLocation,
  baseMapStyle: _baseMapStyle,
}: MapViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const radarLayerRef = useRef<L.TileLayer | null>(null);
  const satelliteLayerRef = useRef<L.TileLayer | null>(null);
  const markerRef = useRef<L.Marker | null>(null);

  const [radarFrames, setRadarFrames] = useState<RainViewerFrame[]>([]);
  const [satFrames, setSatFrames] = useState<RainViewerFrame[]>([]);

  // Initialize map
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [center.lat, center.lng],
      zoom,
      zoomControl: false,
      attributionControl: true,
    });

    L.control.zoom({ position: 'topright' }).addTo(map);

    L.tileLayer(DARK_TILES, {
      attribution: DARK_ATTR,
      maxZoom: 19,
      subdomains: 'abcd',
    }).addTo(map);

    mapRef.current = map;

    // Handle map click
    map.on('click', (e: L.LeafletMouseEvent) => {
      onMapClick?.({ lat: e.latlng.lat, lng: e.latlng.lng });
    });

    return () => {
      map.remove();
      mapRef.current = null;
    };
    // Only run once
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Store the latest onMapClick in a ref so we don't need it as a dep
  const onMapClickRef = useRef(onMapClick);
  onMapClickRef.current = onMapClick;

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    const handler = (e: L.LeafletMouseEvent) => {
      onMapClickRef.current?.({ lat: e.latlng.lat, lng: e.latlng.lng });
    };

    map.off('click');
    map.on('click', handler);
  }, []);

  // Pan to new center when prop changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !center) return;
    map.setView([center.lat, center.lng], 8, { animate: true });
  }, [center]);

  // Load RainViewer data
  const loadFrames = useCallback(async () => {
    try {
      const data = await fetchRainViewerData();
      const rFrames = [...data.radar.past, ...data.radar.nowcast];
      const sFrames = data.satellite.infrared;
      setRadarFrames(rFrames);
      setSatFrames(sFrames);
      onRadarFramesLoaded?.(rFrames.length);
      onSatelliteFramesLoaded?.(sFrames.length);
    } catch (err) {
      console.error('Failed to load RainViewer data:', err);
    }
  }, [onRadarFramesLoaded, onSatelliteFramesLoaded]);

  useEffect(() => {
    loadFrames();
    const interval = setInterval(loadFrames, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [loadFrames]);

  // Update radar layer
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (radarLayerRef.current) {
      map.removeLayer(radarLayerRef.current);
      radarLayerRef.current = null;
    }

    if (showRadar && radarFrames.length > 0) {
      const idx = radarFrame >= 0 ? Math.min(radarFrame, radarFrames.length - 1) : radarFrames.length - 1;
      const frame = radarFrames[idx];
      if (frame) {
        const layer = L.tileLayer(radarTileUrl(frame), {
          opacity: 0.7,
          maxZoom: 19,
        });
        layer.addTo(map);
        radarLayerRef.current = layer;
      }
    }
  }, [showRadar, radarFrame, radarFrames]);

  // Update satellite layer
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (satelliteLayerRef.current) {
      map.removeLayer(satelliteLayerRef.current);
      satelliteLayerRef.current = null;
    }

    if (showSatellite && satFrames.length > 0) {
      const idx = satelliteFrame >= 0 ? Math.min(satelliteFrame, satFrames.length - 1) : satFrames.length - 1;
      const frame = satFrames[idx];
      if (frame) {
        const layer = L.tileLayer(satelliteTileUrl(frame), {
          opacity: 0.6,
          maxZoom: 19,
        });
        layer.addTo(map);
        satelliteLayerRef.current = layer;
      }
    }
  }, [showSatellite, satelliteFrame, satFrames]);

  // Update marker for selected location
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (markerRef.current) {
      map.removeLayer(markerRef.current);
      markerRef.current = null;
    }

    if (selectedLocation) {
      const pulseIcon = L.divIcon({
        className: 'map-pulse-marker',
        html: '<div class="pulse-dot"></div><div class="pulse-ring"></div>',
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });
      const marker = L.marker([selectedLocation.lat, selectedLocation.lng], { icon: pulseIcon });
      marker.addTo(map);
      markerRef.current = marker;
    }
  }, [selectedLocation]);

  return <div ref={containerRef} className="map-container" />;
}
