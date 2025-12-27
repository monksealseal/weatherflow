// Lightweight, non-rendering stand-in for Three.js to allow the game scaffold to run
// in environments without npm registry access. Only the minimal surface used by
// AtmosphereViewer is implemented.

type ConstructorOpts = Record<string, unknown>;

export class Vector3 {
  constructor(public x = 0, public y = 0, public z = 0) {}
  set(x: number, y: number, z: number) {
    this.x = x;
    this.y = y;
    this.z = z;
    return this;
  }
  copy(v: Vector3) {
    this.x = v.x;
    this.y = v.y;
    this.z = v.z;
    return this;
  }
}

class Object3D {
  position = new Vector3();
  rotation = new Vector3();
  children: Object3D[] = [];
  add(...objs: Object3D[]) {
    this.children.push(...objs);
  }
}

export class Color {
  constructor(public value: string | number) {}
}

export class Scene extends Object3D {
  background: Color | null = null;
  clear() {
    this.children = [];
  }
}

export class PerspectiveCamera extends Object3D {
  aspect: number;
  constructor(public fov: number, aspect: number, public near: number, public far: number) {
    super();
    this.aspect = aspect;
  }
  lookAt(_x: number, _y: number, _z: number) {
    return;
  }
  updateProjectionMatrix() {
    return;
  }
}

export class Geometry {
  constructor(public args: unknown[] = []) {}
}

export class PlaneGeometry extends Geometry {
  constructor(width: number, height: number, ...rest: unknown[]) {
    super([width, height, ...rest]);
  }
}

export class SphereGeometry extends Geometry {
  constructor(radius: number, ...rest: unknown[]) {
    super([radius, ...rest]);
  }
}

export class Material {
  constructor(public opts: ConstructorOpts = {}) {}
}

export class MeshPhongMaterial extends Material {}
export class MeshStandardMaterial extends Material {}
export class LineBasicMaterial extends Material {}
export class PointsMaterial extends Material {}

export class Mesh extends Object3D {
  constructor(public geometry: Geometry, public material: Material) {
    super();
  }
}

class LineGeometry {
  points: Vector3[] = [];
  setFromPoints(points: Vector3[]) {
    this.points = points;
    return this;
  }
}

export class Line extends Object3D {
  geometry: LineGeometry = new LineGeometry();
  constructor(public material: Material = new LineBasicMaterial()) {
    super();
  }
}

export class Points extends Object3D {
  constructor(public geometry: Geometry, public material: Material) {
    super();
  }
}

export class Group extends Object3D {}

export class AmbientLight extends Object3D {
  constructor(public color: string | number, public intensity: number) {
    super();
  }
}

export class DirectionalLight extends Object3D {
  constructor(public color: string | number, public intensity: number) {
    super();
  }
}

export class AxesHelper extends Object3D {
  constructor(public size: number) {
    super();
  }
}

export class CanvasTexture {
  needsUpdate = false;
  wrapS: unknown;
  wrapT: unknown;
  constructor(public canvas: HTMLCanvasElement) {}
}

export class WebGLRenderer {
  domElement: HTMLDivElement;
  constructor(_opts?: ConstructorOpts) {
    this.domElement = document.createElement('div');
  }
  setSize(_w: number, _h: number) {
    return;
  }
  setPixelRatio(_r: number) {
    return;
  }
  render(_scene: Scene, _camera: PerspectiveCamera) {
    return;
  }
  dispose() {
    return;
  }
}
