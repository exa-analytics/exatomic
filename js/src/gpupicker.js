// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0

/*

Source obtained from https://github.com/brianxu/GPUPicker
Slightly modified

The MIT License (MIT)

Copyright (c) 2015 Baoxuan(Brian) Xu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

"use strict";

var THREE = require("three")

var add_gpupicker = (THREE) => {
    var FaceIDShader = {

        vertex_shader: [
            "attribute float size;",
            "attribute vec3 color;",
            "varying vec3 vColor;",
            "",
            "void main() {",
            "    vColor = color;",
            "    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);",
            "    gl_PointSize = size * (2000.0 / length(mvPosition.xyz));",
            "    gl_Position = projectionMatrix * mvPosition;",
            "}"
        ].join("\n"),

        fragmentShader: [
            "#ifdef GL_ES",
            "precision highp float;",
            "#endif",
            "",
            "varying vec4 worldId;",
            "",
            "void main() {",
            "  gl_FragColor = worldId;",
            "}"
        ].join("\n"),

        point_frag_shader: [
            "varying vec3 vColor;",
            "",
            "void main() {",
            "    if (length(gl_PointCoord * 2.0 - 1.0) > 1.0)",
            "        discard;",
            "    gl_FragColor = vec4(vColor, 1.0);",
            "}"
        ].join("\n"),

        vertexShader: [
            "attribute float id;",
            "",
            "uniform float size;",
            "uniform float scale;",
            "uniform float baseId;",
            "",
            "varying vec4 worldId;",
            "",
            "void main() {",
            "  vec4 mvPosition = modelViewMatrix * vec4( position, 1.0 );",
            "  gl_PointSize = size * ( scale / length( mvPosition.xyz ) );",
            "  float i = baseId + id;",
            "  vec3 a = fract(vec3(1.0/255.0, 1.0/(255.0*255.0), 1.0/(255.0*255.0*255.0)) * i);",
            "  a -= a.xxy * vec3(0.0, 1.0/255.0, 1.0/255.0);",
            "  worldId = vec4(a,1);",
            "  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );",
            "}"
        ].join("\n")

    };

    var FaceIDMaterial = function () {
        THREE.ShaderMaterial.call(this, {
            uniforms: {
                baseId: {
                    type: "f",
                    value: 0
                },
                size: {
                    type: "f",
                    value: 0.01,
                },
                scale: {
                    type: "f",
                    value: 400,
                }
            },
            vertexShader: FaceIDShader.vertexShader,
            fragmentShader: FaceIDShader.fragmentShader

        });
    };
    FaceIDMaterial.prototype = Object.create(THREE.ShaderMaterial.prototype);
    FaceIDMaterial.prototype.constructor = FaceIDMaterial;
    FaceIDMaterial.prototype.setBaseID = function (baseId) {
        this.uniforms.baseId.value = baseId;
    };
    FaceIDMaterial.prototype.setPointSize = function (size) {
        this.uniforms.size.value = size;
    };
    FaceIDMaterial.prototype.setPointScale = function (scale) {
        this.uniforms.scale.value = scale;
    };

    //add a originalObject to Object3D
    (function (clone) {
        THREE.Object3D.prototype.clone = function (recursive) {
            var object = clone.call(this, recursive);
            // keep a ref to originalObject
            object.originalObject = this;
            object.priority = this.priority;
            return object;
        };
    }(THREE.Object3D.prototype.clone));
    //add a originalObject to Points
    (function (clone) {
        THREE.Points.prototype.clone = function (recursive) {
            var object = clone.call(this, recursive);
            // keep a ref to originalObject
            object.originalObject = this;
            object.priority = this.priority;
            return object;
        };
    }(THREE.Points.prototype.clone));
    //add a originalObject to Mesh
    (function (clone) {
        THREE.Mesh.prototype.clone = function () {
            var object = clone.call(this);
            // keep a ref to originalObject
            object.originalObject = this;
            object.priority = this.priority;
            return object;
        };
    }(THREE.Mesh.prototype.clone));
    //add a originalObject to Line
    (function (clone) {
        THREE.Line.prototype.clone = function () {
            var object = clone.call(this);
            // keep a ref to originalObject
            object.originalObject = this;
            object.priority = this.priority;
            return object;
        };
    }(THREE.Line.prototype.clone));

    THREE.Mesh.prototype.raycastWithID = (function () {
        var vA = new THREE.Vector3();
        var vB = new THREE.Vector3();
        var vC = new THREE.Vector3();
        var inverseMatrix = new THREE.Matrix4();
        var ray = new THREE.Ray();
        var triangle = new THREE.Triangle();
        var intersectionPointWorld = new THREE.Vector3();
        var intersectionPoint = new THREE.Vector3();
        function checkIntersection(object, raycaster, ray, pA, pB, pC, point) {
            var intersect;
            intersect = ray.intersectTriangle(pC, pB, pA, false, point);
            var plane = new THREE.Plane();
            plane.setFromCoplanarPoints(pA, pB, pC);
            intersect = ray.intersectPlane(plane, new THREE.Vector3());
            if (intersect === null) return null;
            intersectionPointWorld.copy(point);
            intersectionPointWorld.applyMatrix4(object.matrixWorld);
            var distance = raycaster.ray.origin.distanceTo(intersectionPointWorld);
            if (distance < raycaster.near || distance > raycaster.far) return null;
            return {
                distance: distance,
                point: intersectionPointWorld.clone(),
                object: object
            };

        }

        return function (elID, raycaster) {
            var geometry = this.geometry;
            var attributes = geometry.attributes;
            inverseMatrix.getInverse(this.matrixWorld);
            ray.copy(raycaster.ray).applyMatrix4(inverseMatrix);
            var a, b, c;
            if (geometry.index !== null) {
                console.log("WARNING: raycastWithID does not support indexed vertices");
            } else {
                var position = attributes.position;
                var j = elID * 3;
                a = j;
                b = j + 1;
                c = j + 2;
                vA.fromBufferAttribute(position, a);
                vB.fromBufferAttribute(position, b);
                vC.fromBufferAttribute(position, c);
            }
            var intersection = checkIntersection(this, raycaster, ray, vA, vB, vC, intersectionPoint);
            if (intersection === null) {
                console.log("WARNING: intersectionPoint missing");
                return;
            }

            var face = new THREE.Face3(a, b, c);
            THREE.Triangle.getNormal(vA, vB, vC, face.normal);

            intersection.face = face;
            intersection.faceIndex = a;
            return intersection;
        };

    }());

    THREE.Line.prototype.raycastWithID = (function () {
        var inverseMatrix = new THREE.Matrix4();
        var ray = new THREE.Ray();

        var vStart = new THREE.Vector3();
        var vEnd = new THREE.Vector3();
        var interSegment = new THREE.Vector3();
        var interRay = new THREE.Vector3();
        return function (elID, raycaster) {
            inverseMatrix.getInverse(this.matrixWorld);
            ray.copy(raycaster.ray).applyMatrix4(inverseMatrix);
            var geometry = this.geometry;
            if (geometry instanceof THREE.BufferGeometry) {

                var attributes = geometry.attributes;

                if (geometry.index !== null) {
                    console.log("WARNING: raycastWithID does not support indexed vertices");
                } else {

                    var positions = attributes.position.array;
                    var i = elID * 6;
                    vStart.fromArray(positions, i);
                    vEnd.fromArray(positions, i + 3);

                    var distSq = ray.distanceSqToSegment(vStart, vEnd, interRay, interSegment);
                    var distance = ray.origin.distanceTo(interRay);

                    if (distance < raycaster.near || distance > raycaster.far) return;

                    var intersect = {

                        distance: distance,
                        // What do we want? intersection point on the ray or on the segment??
                        // point: raycaster.ray.at( distance ),
                        point: interSegment.clone().applyMatrix4(this.matrixWorld),
                        index: i,
                        face: null,
                        faceIndex: null,
                        object: this

                    };
                    return intersect;

                }

            }
        };

    })();

    THREE.Points.prototype.raycastWithID = (function () {

        var inverseMatrix = new THREE.Matrix4();
        var ray = new THREE.Ray();

        return function (elID, raycaster) {
            var object = this;
            var geometry = object.geometry;

            inverseMatrix.getInverse(this.matrixWorld);
            ray.copy(raycaster.ray).applyMatrix4(inverseMatrix);
            var position = new THREE.Vector3();

            var testPoint = function (point, index) {
                var rayPointDistance = ray.distanceToPoint(point);
                var target = new THREE.Vector3();
                var intersectPoint = ray.closestPointToPoint(point, target);
                intersectPoint.applyMatrix4(object.matrixWorld);

                var distance = raycaster.ray.origin.distanceTo(intersectPoint);

                if (distance < raycaster.near || distance > raycaster.far) return;

                var intersect = {

                    target: target,
                    distance: distance,
                    distanceToRay: rayPointDistance,
                    point: intersectPoint.clone(),
                    index: index,
                    face: null,
                    object: object

                };
                return intersect;
            };
            var attributes = geometry.attributes;
            var positions = attributes.position.array;
            position.fromArray(positions, elID * 3);

            return testPoint(position, elID);

        };

    }());

    THREE.GPUPicker = function (option) {
        if (option === undefined) {
            option = {};
        }
        this.pickingScene = new THREE.Scene();
        this.pickingTexture = new THREE.WebGLRenderTarget();
        this.pickingTexture.texture.minFilter = THREE.LinearFilter;
        this.pickingTexture.texture.generateMipmaps = false;
        this.lineShell = option.lineShell !== undefined ? option.lineShell : 4;
        this.pointShell = option.pointShell !== undefined ? option.pointShell : 0.1;
        this.debug = option.debug !== undefined ? option.debug : false;
        if (this.debug) {
            console.log("GPUPicker initialized with:", option)
        }
        this.needUpdate = true;
        if (option.renderer) {
            this.setRenderer(option.renderer);
        }

        // array of original objects
        this.container = [];
        this.objectsMap = {};
        //default filter
        this.setFilter();
    };
    THREE.GPUPicker.prototype.setRenderer = function (renderer) {
        this.renderer = renderer;
        var size = renderer.getSize();
        if (this.debug) console.log("GPUPicker setting renderer", size.width, size.height * 2)
        this.resizeTexture(size.width, size.height * 2);
        this.needUpdate = true;
    };
    THREE.GPUPicker.prototype.resizeTexture = function (width, height) {
        if (this.debug) console.log("GPUPicker resizing texture", width, height)
        this.pickingTexture.setSize(width, height);
        this.pixelBuffer = new Uint8Array(4 * width * height);
        this.needUpdate = true;
    };
    THREE.GPUPicker.prototype.setCamera = function (camera) {
        if (this.debug) console.log("GPUPicker setting camera", camera)
        this.camera = camera;
        this.needUpdate = true;
    };
    THREE.GPUPicker.prototype.update = function () {
        if (this.needUpdate) {
            this.renderer.render(this.pickingScene, this.camera, this.pickingTexture);
            //read the rendering texture
            console.log("reading renderer into pixelBuffer")
            this.renderer.readRenderTargetPixels(this.pickingTexture, 0, 0, this.pickingTexture.width, this.pickingTexture.height, this.pixelBuffer);
            this.needUpdate = false;
            if (this.debug) console.log("GPUPicker rendering updated");
        }
    };
    THREE.GPUPicker.prototype.setFilter = function (func) {
        if (func instanceof Function) {
            this.filterFunc = func;
        } else {
            //default filter
            this.filterFunc = function (object) {
                return true;
            };
        }

    };
    THREE.GPUPicker.prototype.setScene = function (scene) {
        this.pickingScene = scene.clone();
        this._processObject(this.pickingScene, 0);
        this.needUpdate = true;
    };


    THREE.GPUPicker.prototype.pick = function (mouse, raycaster) {
        /*"""
        Args:
            mouse (THREE.Vector2): {x: in [-1,1], y: in [-1,1]}
            raycaster (THREE.Raycaster): caster

        Note:
            pixelBuffer is all the pixels read from the renderer
            readRenderTargetPixels target rectangle starting from
            the lower left corner 

            event.clientX, event.clientY is offset from larger window. unuseful
            this.getBoundingClientRect 

        Plan to recompute mouse position to pixel buffer mapping
        from the [-1,1] space.

        */
        this.needUpdate = true
        this.update()
//        var calcx = Math.round((mouse.x / 2 + 0.5) * this.pickingTexture.width)
//        var calcy = Math.round((mouse.y / 2 + 0.5) * this.pickingTexture.height)
        let calcx = Math.round((mouse.x + 1) / 2 * this.pickingTexture.width)
        let calcy = Math.round((mouse.y + 1) / 2 * this.pickingTexture.height)
        var index = calcx + calcy * this.pickingTexture.width
        //interpret the pixel as an ID
        var id = ((this.pixelBuffer[index * 4 + 2] * 255 * 255) +
                  (this.pixelBuffer[index * 4 + 1] * 255) +
                  (this.pixelBuffer[index * 4    ]))
        if (id === 0) {
            return;
        }
        console.log("pick id:", id)
        var result = this._getObject(this.pickingScene, 0, id);
        var object = result[1];
        var elementId = id - result[0];
        let a, b, c
        if (object) {
            if (object.raycastWithID) {
                var intersect = object.raycastWithID(elementId, raycaster);
                a = this.pixelBuffer[index * 4 + 2]
                b = this.pixelBuffer[index * 4 + 1]
                c = this.pixelBuffer[index * 4    ]
                if (false) {
                    console.log("gpu calc", calcx, calcy, index, index * 4)
                    console.log("gpu pidx", a, b, c)
                    console.log("gpu col ", a * 255 * 255, b * 255, c)
                }
                // console.log("gpu buf ", this.pixelBuffer)
                intersect.object = object.originalObject;
                if (intersect.object) {
                    intersect.object.point_idx = id
                    
                    intersect.object.name = `${intersect.object.base_name}:${id}`
                }
                return intersect;
            }
        }
        return;
    };

    /*
     * get object by id
     */
    THREE.GPUPicker.prototype._getObject = function (object, baseId, id) {
        // if (this.debug) console.log("_getObject ",baseId);
        if (object.elementsCount !== undefined && id >= baseId && id < baseId + object.elementsCount) {
            return [baseId, object];
        }
        if (object.elementsCount !== undefined) {
            baseId += object.elementsCount;
        }
        var result = [baseId, undefined];
        for (var i = 0; i < object.children.length; i++) {
            result = this._getObject(object.children[i], result[0], id);
            if (result[1] !== undefined)
                break;
        }
        return result;
    };

    /*
     * process the object to add elementId information
     */
    THREE.GPUPicker.prototype._processObject = function (object, baseId) {
        if (this.debug) console.log("GPUPicker processing object", object, baseId)
        baseId += this._addElementID(object, baseId);
        for (var i = 0; i < object.children.length; i++) {
            baseId = this._processObject(object.children[i], baseId);
        }
        return baseId;
    };

    THREE.GPUPicker.prototype._addElementID = function (object, baseId) {
        if (this.debug) console.log("adding element ID")
        if (!this.filterFunc(object) && object.geometry !== undefined) {
            object.visible = false;
            return 0;
        }

        if (object.geometry) {
            var __pickingGeometry;
            //check if geometry has cached geometry for picking
            if (object.geometry.__pickingGeometry) {
                __pickingGeometry = object.geometry.__pickingGeometry;
            } else {
                __pickingGeometry = object.geometry;
                // convert geometry to buffer geometry
                if (object.geometry instanceof THREE.Geometry) {
                    if (this.debug) console.log("convert geometry to buffer geometry");
                    __pickingGeometry = new THREE.BufferGeometry().setFromObject(object);
                }
                var units = 1;
                if (object instanceof THREE.Points) {
                    units = 1;
                } else if (object instanceof THREE.Line) {
                    units = 2;
                } else if (object instanceof THREE.Mesh) {
                    units = 3;
                }
                var el, el3, elementsCount, i, indices, positionBuffer, vertex3, verts, vertexIndex3;
                if (__pickingGeometry.index !== null) {
                    __pickingGeometry = __pickingGeometry.clone();
                    if (this.debug) console.log("convert indexed geometry to non-indexed geometry");

                    indices = __pickingGeometry.index.array;
                    verts = __pickingGeometry.attributes.position.array;
                    delete __pickingGeometry.attributes.position;
                    __pickingGeometry.index = null;
                    delete __pickingGeometry.attributes.normal;
                    elementsCount = indices.length / units;
                    positionBuffer = new Float32Array(elementsCount * 3 * units);

                    __pickingGeometry.addAttribute('position', new THREE.BufferAttribute(positionBuffer, 3));
                    for (el = 0; el < elementsCount; ++el) {
                        el3 = units * el;
                        for (i = 0; i < units; ++i) {
                            vertexIndex3 = 3 * indices[el3 + i];
                            vertex3 = 3 * (el3 + i);
                            positionBuffer[vertex3] = verts[vertexIndex3];
                            positionBuffer[vertex3 + 1] = verts[vertexIndex3 + 1];
                            positionBuffer[vertex3 + 2] = verts[vertexIndex3 + 2];
                        }
                    }

                    __pickingGeometry.computeVertexNormals();
                }
                if (object instanceof THREE.Line && !(object instanceof THREE.LineSegments)) {
                    if (this.debug) console.log("convert Line to LineSegments");
                    verts = __pickingGeometry.attributes.position.array;
                    delete __pickingGeometry.attributes.position;
                    elementsCount = verts.length / 3 - 1;
                    positionBuffer = new Float32Array(elementsCount * units * 3);

                    __pickingGeometry.addAttribute('position', new THREE.BufferAttribute(positionBuffer, 3));
                    for (el = 0; el < elementsCount; ++el) {
                        el3 = 3 * el;
                        vertexIndex3 = el3;
                        vertex3 = el3 * 2;
                        positionBuffer[vertex3] = verts[vertexIndex3];
                        positionBuffer[vertex3 + 1] = verts[vertexIndex3 + 1];
                        positionBuffer[vertex3 + 2] = verts[vertexIndex3 + 2];
                        positionBuffer[vertex3 + 3] = verts[vertexIndex3 + 3];
                        positionBuffer[vertex3 + 4] = verts[vertexIndex3 + 4];
                        positionBuffer[vertex3 + 5] = verts[vertexIndex3 + 5];

                    }

                    __pickingGeometry.computeVertexNormals();
                    object.__proto__ = THREE.LineSegments.prototype; //make the renderer render as line segments
                }
                var attributes = __pickingGeometry.attributes;
                var positions = attributes.position.array;
                var vertexCount = positions.length / 3;
                var ids = new THREE.Float32BufferAttribute(vertexCount, 1);
                //set vertex id color

                for (var i = 0, il = vertexCount / units; i < il; i++) {
                    for (var j = 0; j < units; ++j) {
                        ids.array[i * units + j] = i;
                    }
                }
                __pickingGeometry.addAttribute('id', ids);
                __pickingGeometry.elementsCount = vertexCount / units;
                //cache __pickingGeometry inside geometry
                object.geometry.__pickingGeometry = __pickingGeometry;
            }

            //use __pickingGeometry in the picking mesh
            object.geometry = __pickingGeometry;
            object.elementsCount = __pickingGeometry.elementsCount;//elements count

            var pointSize = object.material.size || 0.01;
            var linewidth = object.material.linewidth || 1;
            object.material = new FaceIDMaterial();
            object.material.linewidth = linewidth + this.lineShell;//make the line a little wider to hit
            object.material.setBaseID(baseId);
            object.material.setPointSize(pointSize + this.pointShell);//make the point a little wider to hit
            object.material.setPointScale(this.renderer.getSize().height * this.renderer.getPixelRatio() / 2);
            return object.elementsCount;
        }
        return 0;
    };
    return THREE;
};

module.exports = {
    add_gpupicker: add_gpupicker
}
