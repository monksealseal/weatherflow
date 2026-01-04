import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AmbientLight,
  AxesHelper,
  CanvasTexture,
  Color,
  DirectionalLight,
  Group,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshPhongMaterial,
  MeshStandardMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Points,
  PointsMaterial,
  Scene,
  SphereGeometry,
  Vector3,
  WebGLRenderer
} from '../vendor/three-lite';

type CameraMode = 'pilot' | 'orbital';
type Overlay = 'none' | 'temperature' | 'moisture';

const ORBITAL_DISTANCE = 28;
const PILOT_DISTANCE = 12;

const createIntersectionLine = (height: number): Line => {
  const points = [new Vector3(0, -height / 2, 0), new Vector3(0, height / 2, 0)];
  const geometry = new LineBasicMaterial({ color: 0xffff00 }).linewidth ? 
    (() => { const g = new PlaneGeometry(); g.setFromPoints?.(points); return g; })() : 
    new PlaneGeometry();
  const material = new LineBasicMaterial({ color: 0xffff00 });
  return new Line(geometry as any, material);
};

const createSlicePlane = (
  width: number,
  height: number,
  color: Color | string,
  position: Vector3,
  rotation: Vector3,
  overlayTexture?: CanvasTexture
): Mesh => {
  const geometry = new PlaneGeometry(width, height);
  const material = new MeshPhongMaterial({
    color,
    transparent: true,
    opacity: 0.28,
    side: 2,
    map: overlayTexture ?? null,
    depthWrite: false
  });
  const plane = new Mesh(geometry, material);
  plane.position.copy(position);
  plane.rotation.set(rotation.x, rotation.y, rotation.z);
  return plane;
};

const createNoiseTexture = (size = 256, hue = 210) => {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    return null;
  }
  const imageData = ctx.createImageData(size, size);
  for (let i = 0; i < size * size; i += 1) {
    const value = Math.floor(Math.random() * 255);
    const index = i * 4;
    imageData.data[index] = value * 0.7;
    imageData.data[index + 1] = value * 0.8;
    imageData.data[index + 2] = Math.min(255, value + hue * 0.2);
    imageData.data[index + 3] = Math.floor(70 + (value / 255) * 120);
  }
  ctx.putImageData(imageData, 0, 0);
  const texture = new CanvasTexture(canvas);
  texture.needsUpdate = true;
  texture.wrapS = texture.wrapT;
  return texture;
};

const createVolumetricGroup = (): Group => {
  const group = new Group();
  const layerCount = 6;
  for (let i = 0; i < layerCount; i += 1) {
    const size = 20 - i * 1.2;
    const opacity = 0.14 + i * 0.04;
    const texture = createNoiseTexture(256, 200 + i * 8);
    const material = new MeshStandardMaterial({
      map: texture ?? undefined,
      transparent: true,
      opacity,
      color: new Color('#dbeafe'),
      depthWrite: false
    });
    const plane = new Mesh(new PlaneGeometry(size, size), material);
    plane.position.set(0, 3 + i * 0.4, 0);
    plane.rotation.x = -Math.PI / 2;
    group.add(plane);
  }
  const particles = new Points(
    new PlaneGeometry(24, 24, 64, 64),
    new PointsMaterial({
      color: '#e0f2fe',
      size: 0.06,
      transparent: true,
      opacity: 0.4
    })
  );
  particles.rotation.x = -Math.PI / 2;
  particles.position.set(0, 3, 0);
  group.add(particles);
  return group;
};

const AtmosphereViewer = (): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const pilotCameraRef = useRef<PerspectiveCamera | null>(null);
  const orbitalCameraRef = useRef<PerspectiveCamera | null>(null);
  const [mode, setMode] = useState<CameraMode>('orbital');
  const [overlay, setOverlay] = useState<Overlay>('temperature');
  const [sliceTilt, setSliceTilt] = useState(0);
  const [fallbackMessage, setFallbackMessage] = useState<string | null>(null);

  const overlayTexture = useMemo(() => createNoiseTexture(256, overlay === 'temperature' ? 10 : 140), [overlay]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 480;

    const scene = new Scene();
    scene.background = new Color(0x0b1224);

    let renderer: WebGLRenderer | null = null;
    try {
      renderer = new WebGLRenderer({ antialias: true, alpha: false });
      renderer.setSize(width, height);
      renderer.setClearColor(new Color(0x0b1224));
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
      container.appendChild(renderer.domElement);
    } catch (err) {
      console.warn('WebGL renderer could not be created', err);
      setFallbackMessage('WebGL unavailable: showing fallback view.');
      return;
    }

    const pilotCamera = new PerspectiveCamera(60, width / height, 0.1, 200);
    pilotCamera.position.set(0, 5, PILOT_DISTANCE);

    const orbitalCamera = new PerspectiveCamera(60, width / height, 0.1, 400);
    orbitalCamera.position.set(0, 18, ORBITAL_DISTANCE);
    orbitalCamera.lookAt(0, 0, 0);

    pilotCameraRef.current = pilotCamera;
    orbitalCameraRef.current = orbitalCamera;
    sceneRef.current = scene;
    rendererRef.current = renderer;

    const ambient = new AmbientLight(0xffffff, 0.65);
    const dirLight = new DirectionalLight(0xffffff, 1.15);
    dirLight.position.set(12, 22, 14);
    scene.add(ambient, dirLight);

    // Earth scaffold
    const globe = new Mesh(
      new SphereGeometry(5, 48, 32),
      new MeshPhongMaterial({
        color: new Color('#0b4f6c'),
        emissive: new Color('#0c4a6e'),
        shininess: 8,
        transparent: true,
        opacity: 0.92
      })
    );
    globe.position.set(0, 0, 0);
    scene.add(globe);

    // Lat–height and lon–height slice planes (simplified scaffold)
    const sliceHeight = 12;
    const sliceWidth = 18;
    const latHeightPlane = createSlicePlane(sliceWidth, sliceHeight, '#1d4ed8', new Vector3(0, 0, 0), new Vector3(0, 0, 0), overlayTexture ?? undefined);
    const lonHeightPlane = createSlicePlane(
      sliceWidth,
      sliceHeight,
      '#be123c',
      new Vector3(0, 0, 0),
      new Vector3(0, Math.PI / 2, 0),
      overlayTexture ?? undefined
    );

    // Intersection tracer
    const intersectionLine = createIntersectionLine(sliceHeight);
    scene.add(latHeightPlane, lonHeightPlane, intersectionLine);

    const axes = new AxesHelper(8);
    scene.add(axes);

    // Lightweight volumetric proxy for clouds
    const volumeGroup = createVolumetricGroup();
    scene.add(volumeGroup);

    const handleResize = () => {
      const newWidth = container.clientWidth || width;
      const newHeight = container.clientHeight || height;
      renderer.setSize(newWidth, newHeight);
      [pilotCamera, orbitalCamera].forEach((camera) => {
        camera.aspect = newWidth / newHeight;
        camera.updateProjectionMatrix();
      });
    };
    window.addEventListener('resize', handleResize);

    let animationFrame = 0;
    // Render once immediately to avoid blank first frame in headless captures
    renderer.render(scene, orbitalCamera);

    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      globe.rotation.y += 0.0015;
      volumeGroup.rotation.y += 0.0008;
      latHeightPlane.rotation.y = sliceTilt;
      lonHeightPlane.rotation.y = Math.PI / 2 + sliceTilt * 0.3;
      intersectionLine.rotation.y = sliceTilt;
      const camera = mode === 'pilot' ? pilotCamera : orbitalCamera;
      if (mode === 'pilot') {
        camera.lookAt(0, 0, 0);
      }
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animationFrame);
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      if (renderer.domElement && renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      scene.clear();
    };
  }, [mode, overlayTexture, sliceTilt]);

  return (
    <section className="section-card game-viewer">
      <div className="game-viewer__header">
        <div>
          <h2>Atmospheric game scaffold (Phase 1)</h2>
          <p>Three.js scene with slice planes, intersection tracer, and dual camera modes.</p>
        </div>
        <div className="game-viewer__actions">
          <button
            type="button"
            className={mode === 'orbital' ? 'primary-button' : 'ghost-button'}
            onClick={() => setMode('orbital')}
          >
            Satellite view
          </button>
          <button
            type="button"
            className={mode === 'pilot' ? 'primary-button' : 'ghost-button'}
            onClick={() => setMode('pilot')}
          >
            Pilot view
          </button>
        </div>
      </div>
      <div className="game-viewer__controls">
        <label className="control-field">
          Overlay
          <select value={overlay} onChange={(event) => setOverlay(event.target.value as Overlay)}>
            <option value="temperature">Temperature shading</option>
            <option value="moisture">Moisture shading</option>
            <option value="none">No overlay</option>
          </select>
        </label>
        <label className="control-field slider">
          Slice tilt
          <input
            type="range"
            min={-0.6}
            max={0.6}
            step={0.02}
            value={sliceTilt}
            onChange={(event) => setSliceTilt(Number(event.target.value))}
          />
        </label>
      </div>
      <div className="game-viewer__canvas" ref={containerRef}>
        {fallbackMessage && <div className="game-viewer__fallback">{fallbackMessage}</div>}
      </div>
    </section>
  );
};

export default AtmosphereViewer;
