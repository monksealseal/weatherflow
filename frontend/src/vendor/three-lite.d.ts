type ConstructorOpts = Record<string, unknown>;
export declare class Vector3 {
    x: number;
    y: number;
    z: number;
    constructor(x?: number, y?: number, z?: number);
    set(x: number, y: number, z: number): this;
    copy(v: Vector3): this;
}
declare class Object3D {
    position: Vector3;
    rotation: Vector3;
    children: Object3D[];
    add(...objs: Object3D[]): void;
}
export declare class Color {
    value: string | number;
    constructor(value: string | number);
}
export declare class Scene extends Object3D {
    background: Color | null;
    clear(): void;
}
export declare class PerspectiveCamera extends Object3D {
    fov: number;
    near: number;
    far: number;
    aspect: number;
    constructor(fov: number, aspect: number, near: number, far: number);
    lookAt(_x: number, _y: number, _z: number): void;
    updateProjectionMatrix(): void;
}
export declare class Geometry {
    args: unknown[];
    constructor(args?: unknown[]);
}
export declare class PlaneGeometry extends Geometry {
    constructor(width: number, height: number, ...rest: unknown[]);
}
export declare class SphereGeometry extends Geometry {
    constructor(radius: number, ...rest: unknown[]);
}
export declare class Material {
    opts: ConstructorOpts;
    constructor(opts?: ConstructorOpts);
}
export declare class MeshPhongMaterial extends Material {
}
export declare class MeshStandardMaterial extends Material {
}
export declare class LineBasicMaterial extends Material {
}
export declare class PointsMaterial extends Material {
}
export declare class Mesh extends Object3D {
    geometry: Geometry;
    material: Material;
    constructor(geometry: Geometry, material: Material);
}
declare class LineGeometry {
    points: Vector3[];
    setFromPoints(points: Vector3[]): this;
}
export declare class Line extends Object3D {
    material: Material;
    geometry: LineGeometry;
    constructor(material?: Material);
}
export declare class Points extends Object3D {
    geometry: Geometry;
    material: Material;
    constructor(geometry: Geometry, material: Material);
}
export declare class Group extends Object3D {
}
export declare class AmbientLight extends Object3D {
    color: string | number;
    intensity: number;
    constructor(color: string | number, intensity: number);
}
export declare class DirectionalLight extends Object3D {
    color: string | number;
    intensity: number;
    constructor(color: string | number, intensity: number);
}
export declare class AxesHelper extends Object3D {
    size: number;
    constructor(size: number);
}
export declare class CanvasTexture {
    canvas: HTMLCanvasElement;
    needsUpdate: boolean;
    wrapS: unknown;
    wrapT: unknown;
    constructor(canvas: HTMLCanvasElement);
}
export declare class WebGLRenderer {
    domElement: HTMLDivElement;
    constructor(_opts?: ConstructorOpts);
    setSize(_w: number, _h: number): void;
    setPixelRatio(_r: number): void;
    render(_scene: Scene, _camera: PerspectiveCamera): void;
    dispose(): void;
}
export {};
