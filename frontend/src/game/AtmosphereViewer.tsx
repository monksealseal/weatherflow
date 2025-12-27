import { useEffect, useRef, useState } from 'react';
import {
  AmbientLight,
  AxesHelper,
  BoxGeometry,
  Color,
  DirectionalLight,
  Line,
  LineBasicMaterial,
  Mesh,
  MeshPhongMaterial,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  SphereGeometry,
  Vector3,
  WebGLRenderer
} from 'three';

type CameraMode = 'pilot' | 'orbital';

const ORBITAL_DISTANCE = 28;
const PILOT_DISTANCE = 12;

const createIntersectionLine = (height: number): Line => {
  const points = [new Vector3(0, -height / 2, 0), new Vector3(0, height / 2, 0)];
  const geometry = new Line().geometry.setFromPoints(points);
  const material = new LineBasicMaterial({ color: 0xffff00, linewidth: 2 });
  return new Line(geometry, material);
};

const createSlicePlane = (
  width: number,
  height: number,
  color: Color | string,
  position: Vector3,
  rotation: Vector3
): Mesh => {
  const geometry = new PlaneGeometry(width, height);
  const material = new MeshPhongMaterial({
    color,
    transparent: true,
    opacity: 0.28,
    side: 2
  });
  const plane = new Mesh(geometry, material);
  plane.position.copy(position);
  plane.rotation.set(rotation.x, rotation.y, rotation.z);
  return plane;
};

const AtmosphereViewer = (): JSX.Element => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const sceneRef = useRef<Scene | null>(null);
  const pilotCameraRef = useRef<PerspectiveCamera | null>(null);
  const orbitalCameraRef = useRef<PerspectiveCamera | null>(null);
  const [mode, setMode] = useState<CameraMode>('orbital');

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const width = container.clientWidth || 800;
    const height = container.clientHeight || 480;

    const scene = new Scene();
    scene.background = new Color(0x020617);

    const renderer = new WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    container.appendChild(renderer.domElement);

    const pilotCamera = new PerspectiveCamera(60, width / height, 0.1, 200);
    pilotCamera.position.set(0, 5, PILOT_DISTANCE);

    const orbitalCamera = new PerspectiveCamera(60, width / height, 0.1, 400);
    orbitalCamera.position.set(0, 18, ORBITAL_DISTANCE);
    orbitalCamera.lookAt(0, 0, 0);

    pilotCameraRef.current = pilotCamera;
    orbitalCameraRef.current = orbitalCamera;
    sceneRef.current = scene;
    rendererRef.current = renderer;

    const ambient = new AmbientLight(0xffffff, 0.45);
    const dirLight = new DirectionalLight(0xffffff, 1.0);
    dirLight.position.set(10, 20, 10);
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
    const latHeightPlane = createSlicePlane(
      sliceWidth,
      sliceHeight,
      '#1d4ed8',
      new Vector3(0, 0, 0),
      new Vector3(0, 0, 0)
    );
    const lonHeightPlane = createSlicePlane(
      sliceWidth,
      sliceHeight,
      '#be123c',
      new Vector3(0, 0, 0),
      new Vector3(0, Math.PI / 2, 0)
    );

    // Intersection tracer
    const intersectionLine = createIntersectionLine(sliceHeight);
    scene.add(latHeightPlane, lonHeightPlane, intersectionLine);

    const axes = new AxesHelper(8);
    scene.add(axes);

    // Decorative cloud proxy box to hint at volume space
    const cloudProxy = new Mesh(
      new BoxGeometry(16, 8, 16),
      new MeshPhongMaterial({
        color: new Color('#7dd3fc'),
        opacity: 0.15,
        transparent: true
      })
    );
    cloudProxy.position.set(0, 3, 0);
    scene.add(cloudProxy);

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
    const animate = () => {
      animationFrame = requestAnimationFrame(animate);
      globe.rotation.y += 0.0015;
      cloudProxy.rotation.y += 0.001;
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
      container.removeChild(renderer.domElement);
      scene.clear();
    };
  }, [mode]);

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
      <div className="game-viewer__canvas" ref={containerRef} />
    </section>
  );
};

export default AtmosphereViewer;
