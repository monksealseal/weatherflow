// Lightweight, non-rendering stand-in for Three.js to allow the game scaffold to run
// in environments without npm registry access. Only the minimal surface used by
// AtmosphereViewer is implemented.
export class Vector3 {
    constructor(x = 0, y = 0, z = 0) {
        Object.defineProperty(this, "x", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: x
        });
        Object.defineProperty(this, "y", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: y
        });
        Object.defineProperty(this, "z", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: z
        });
    }
    set(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        return this;
    }
    copy(v) {
        this.x = v.x;
        this.y = v.y;
        this.z = v.z;
        return this;
    }
}
class Object3D {
    constructor() {
        Object.defineProperty(this, "position", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Vector3()
        });
        Object.defineProperty(this, "rotation", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new Vector3()
        });
        Object.defineProperty(this, "children", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
    }
    add(...objs) {
        this.children.push(...objs);
    }
}
export class Color {
    constructor(value) {
        Object.defineProperty(this, "value", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: value
        });
    }
}
export class Scene extends Object3D {
    constructor() {
        super(...arguments);
        Object.defineProperty(this, "background", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: null
        });
    }
    clear() {
        this.children = [];
    }
}
export class PerspectiveCamera extends Object3D {
    constructor(fov, aspect, near, far) {
        super();
        Object.defineProperty(this, "fov", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: fov
        });
        Object.defineProperty(this, "near", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: near
        });
        Object.defineProperty(this, "far", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: far
        });
        Object.defineProperty(this, "aspect", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.aspect = aspect;
    }
    lookAt(_x, _y, _z) {
        return;
    }
    updateProjectionMatrix() {
        return;
    }
}
export class Geometry {
    constructor(args = []) {
        Object.defineProperty(this, "args", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: args
        });
    }
}
export class PlaneGeometry extends Geometry {
    constructor(width, height, ...rest) {
        super([width, height, ...rest]);
    }
}
export class SphereGeometry extends Geometry {
    constructor(radius, ...rest) {
        super([radius, ...rest]);
    }
}
export class Material {
    constructor(opts = {}) {
        Object.defineProperty(this, "opts", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: opts
        });
    }
}
export class MeshPhongMaterial extends Material {
}
export class MeshStandardMaterial extends Material {
}
export class LineBasicMaterial extends Material {
}
export class PointsMaterial extends Material {
}
export class Mesh extends Object3D {
    constructor(geometry, material) {
        super();
        Object.defineProperty(this, "geometry", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: geometry
        });
        Object.defineProperty(this, "material", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: material
        });
    }
}
class LineGeometry {
    constructor() {
        Object.defineProperty(this, "points", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: []
        });
    }
    setFromPoints(points) {
        this.points = points;
        return this;
    }
}
export class Line extends Object3D {
    constructor(material = new LineBasicMaterial()) {
        super();
        Object.defineProperty(this, "material", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: material
        });
        Object.defineProperty(this, "geometry", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: new LineGeometry()
        });
    }
}
export class Points extends Object3D {
    constructor(geometry, material) {
        super();
        Object.defineProperty(this, "geometry", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: geometry
        });
        Object.defineProperty(this, "material", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: material
        });
    }
}
export class Group extends Object3D {
}
export class AmbientLight extends Object3D {
    constructor(color, intensity) {
        super();
        Object.defineProperty(this, "color", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: color
        });
        Object.defineProperty(this, "intensity", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: intensity
        });
    }
}
export class DirectionalLight extends Object3D {
    constructor(color, intensity) {
        super();
        Object.defineProperty(this, "color", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: color
        });
        Object.defineProperty(this, "intensity", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: intensity
        });
    }
}
export class AxesHelper extends Object3D {
    constructor(size) {
        super();
        Object.defineProperty(this, "size", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: size
        });
    }
}
export class CanvasTexture {
    constructor(canvas) {
        Object.defineProperty(this, "canvas", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: canvas
        });
        Object.defineProperty(this, "needsUpdate", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: false
        });
        Object.defineProperty(this, "wrapS", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        Object.defineProperty(this, "wrapT", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
    }
}
export class WebGLRenderer {
    constructor(_opts) {
        Object.defineProperty(this, "domElement", {
            enumerable: true,
            configurable: true,
            writable: true,
            value: void 0
        });
        this.domElement = document.createElement('div');
    }
    setSize(_w, _h) {
        return;
    }
    setPixelRatio(_r) {
        return;
    }
    render(_scene, _camera) {
        return;
    }
    dispose() {
        return;
    }
}
