/*"""
Atomic Three.js Application
````````````````````````````````
Creates a 3D visualization of the atomic Universe container.

This code should fully be expected to operate without any knowledge of
where it is being rendered (e.g. custom web gui or IPython widget).
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/atomic/lib/three.min': {
            exports: 'THREE'
        },
        'nbextensions/exa/atomic/lib/TrackballControls': {
            deps: ['nbextensions/exa/atomic/lib/three.min'],
            exports: 'THREE.TrackballControls'
        },
        'nbextensions/exa/atomic/marchingcubes': {
            exports: 'MarchingCubes'
        },
    },
});


define([
    'nbextensions/exa/atomic/lib/three.min',
    'nbextensions/exa/atomic/lib/TrackballControls',
    'nbextensions/exa/atomic/marchingcubes'
], function(
    THREE,
    TrackballControls,
    MarchingCubes
) {
    var vertex_shader = "\
        attribute float size;\
        attribute vec3 color;\
        varying vec3 vColor;\
        \
        void main() {\
            vColor = color;\
            vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);\
            gl_PointSize = size * (450.0 / length(mvPosition.xyz));\
            gl_Position = projectionMatrix * mvPosition;\
        }\
    ";

    var point_frag_shader = "\
        varying vec3 vColor;\
        \
        void main() {\
            if (length(gl_PointCoord * 2.0 - 1.0) > 1.0)\
                discard;\
            gl_FragColor = vec4(vColor, 1.0);\
        }\
    ";

    var circle_frag_shader = "\
        varying vec3 vColor;\
        \
        void main() {\
            if (length(gl_PointCoord * 2.0 - 1.0) > 1.0)\
                discard;\
            if (length(gl_PointCoord * 2.0 - 1.0) < 0.9)\
                discard;\
            gl_FragColor = vec4(vColor, 1.0);\
        }\
    ";

    var AtomicThreeJS = function(canvas) {
        /*"""
        AtomicThreeJS
        ```````````````````````````````````````````````````
        Three.js application for rendering an atomic Universe as an IPython
        widget
        */
        var self = this;
        this.c = canvas;
        this.width = this.c.width();
        this.height = this.c.height();
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.c.get(0),
            antialias: true
        });
        this.renderer.setClearColor(0xFFFFFF);
        this.renderer.setSize(this.width, this.height);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.0001, 10000);
        this.controls = new TrackballControls(this.camera, this.c.get(0));
        this.controls.rotateSpeed = 10.0;
        this.controls.zoomSpeed = 5.0;
        this.controls.panSpeed = 0.5;
        this.controls.noZoom = false;
        this.controls.noPan = false;
        this.controls.staticMoving = true;
        this.controls.dynamicDampingFactor = 0.3;
        this.controls.keys = [65, 83, 68];
        this.controls.addEventListener('change', this.render.bind(this));
        var dlight1 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight3 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight4 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight5 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight6 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight7 = new THREE.DirectionalLight(0xffffff, 0.3);
        var dlight8 = new THREE.DirectionalLight(0xffffff, 0.3);
        dlight1.position.set(100, 100, 100);
        dlight2.position.set(-100, 100, 100);
        dlight3.position.set(100, -100, 100);
        dlight4.position.set(100, 100, -100);
        dlight5.position.set(-100, -100, 100);
        dlight6.position.set(-100, 100, -100);
        dlight7.position.set(100, -100, -100);
        dlight8.position.set(-100, -100, -100);
        this.scene.add(dlight1);
        this.scene.add(dlight2);
        this.scene.add(dlight3);
        this.scene.add(dlight4);
        this.scene.add(dlight5);
        this.scene.add(dlight6);
        this.scene.add(dlight7);
        this.scene.add(dlight8);
    };

    AtomicThreeJS.prototype.resize = function() {
        /*"""
        Resize
        ``````````````
        Allows for the renderer to be resized on canvas resize
        */
        this.width = this.c.width();
        this.height = this.c.height();
        this.renderer.setSize(this.width, this.height);
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.controls.handleResize();
        this.render();
    };

    AtomicThreeJS.prototype.render = function() {
        /*"""
        Render the Three.js Scene
        ```````````````````````````
        */
        this.renderer.render(this.scene, this.camera);
    };

    AtomicThreeJS.prototype.animate = function() {
        /*"""
        Animate
        ```````````````````
        */
        window.requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
    };

    AtomicThreeJS.prototype.add_coordinate_axis = function() {
        /*"""
        */
        var x_vec = new THREE.Vector3(1, 0, 0);
        var y_vec = new THREE.Vector3(0, 1, 0);
        var z_vec = new THREE.Vector3(0, 0, 1);
        var origin = new THREE.Vector3(0, 0, 0);
        var x_arrow = new THREE.ArrowHelper(x_vec, origin, 1, 0x000000);
        var y_arrow = new THREE.ArrowHelper(y_vec, origin, 1, 0x000000);
        var z_arrow = new THREE.ArrowHelper(z_vec, origin, 1, 0x000000);
        x_arrow.line.material.linewidth = 5;
        y_arrow.line.material.linewidth = 5;
        z_arrow.line.material.linewidth = 5;
        this.scene.add(x_arrow);
        this.scene.add(y_arrow);
        this.scene.add(z_arrow);
    };

    AtomicThreeJS.prototype.add_cell = function(xi, xj, xk, yi, yj, yk, zi, zj, zk, ox, oy, oz) {
        /*""""
        Create Unit Cell
        ````````````````````
        */
        var geometry = new THREE.Geometry();
        geometry.vertices.push(new THREE.Vector3(xi, xj, xk));
        geometry.vertices.push(new THREE.Vector3(yi, yj, yk));
        geometry.vertices.push(new THREE.Vector3(zi, zj, zk));
        geometry.vertices.push(new THREE.Vector3(ox, oy, oz));
        var material = new THREE.MeshBasicMaterial({
            'transparent': true,
            'opacity': 0.2,
            'wireframeLinewidth': 10,
            'wireframe': true
        });
        this.scene.remove(this.cell);
        var unitmesh = new THREE.Mesh(geometry, material);
        this.cell = new THREE.BoxHelper(unitmesh);
        this.cell.material.color.set(0x000000);
        this.scene.add(this.cell);
    };

    AtomicThreeJS.prototype.add_points = function(x, y, z, r, c, filled) {
        /*"""
        Add GL Points
        ````````````````````
        */
        var material = new THREE.ShaderMaterial({
            vertexShader: vertex_shader,
            fog: true,
            fragmentShader: point_frag_shader,
            transparent: false
        });
        if (filled == false) {
            material = new THREE.ShaderMaterial({
                vertexShader: vertex_shader,
                fog: true,
                fragmentShader: circle_frag_shader,
                transparent: false
            });
        };
        var geometry = new THREE.BufferGeometry();
        var color = new THREE.Color();
        var n = x.length;
        var positions = new Float32Array(n * 3);
        var colors = new Float32Array(n * 3);
        var sizes = new Float32Array(n);
        for (var i = 0, i3 = 0; i < n; i++, i3 += 3) {
            positions[i3 + 0] = x[i];
            positions[i3 + 1] = y[i];
            positions[i3 + 2] = z[i];
            color.setHex(c[i]);
            colors[i3 + 0] = color.r;
            colors[i3 + 1] = color.g;
            colors[i3 + 2] = color.b;
            sizes[i] = r[i];
        };
        geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.addAttribute('size', new THREE.BufferAttribute(sizes, 1));
        this.scene.remove(this.atom);
        this.atom = new THREE.Points(geometry, material);
        this.scene.add(this.atom);
    };

    AtomicThreeJS.prototype.add_bonds = function(bonds, x, y, z) {
        /*"""
        Add bonds
        `````````````
        */
        var material = new THREE.LineBasicMaterial({
            color: 0x4d4d4d,
            linewidth: 5
        });
        var geometry = new THREE.Geometry();
        var n = bonds.length;
        for (var i = 0; i < n; i++) {
            var a = bonds[i][0];
            var b = bonds[i][1];
            var va = new THREE.Vector3(x[a], y[a], z[a]);
            var vb = new THREE.Vector3(x[b], y[b], z[b]);
            geometry.vertices.push(va);
            geometry.vertices.push(vb);
        };
        this.scene.remove(this.bond);
        this.bond = new THREE.LineSegments(geometry, material);
        this.scene.add(this.bond);
    };

    AtomicThreeJS.prototype.add_surface = function(field, dims, orig, scale, iso) {
        var cpos = '#003399';
        var cneg = '#FF9900';
        var opac = 0.5;
        var marched = MarchingCubes(field, dims, orig, scale, iso);
        var verts = marched['vertices'];
        var faces = marched['faces'];
        var nverts = marched['nvertices'];
        var nfaces = marched['nfaces']
        var vs = verts.length;
        var fs = faces.length;
        var nvs = nverts.length;
        var nfs = nfaces.length;

        var geom = new THREE.Geometry();
        var ngeom = new THREE.Geometry();
        var mat = new THREE.MeshPhongMaterial({
            'color': cpos, 'specular': cpos,
            'transparent': true, 'opacity': opac,
            'shininess': 30
        });
        var nmat = new THREE.MeshPhongMaterial({
            'color': cneg, 'specular': cneg,
            'transparent': true, 'opacity': opac,
            'shininess': 30
        });

        for (var i = 0; i < vs; i++) {
            geom.vertices.push(new THREE.Vector3(
                verts[i][0], verts[i][1], verts[i][2]));
        };
        for (var i = 0; i < fs; i++) {
            geom.faces.push(new THREE.Face3(
                faces[i][0], faces[i][1], faces[i][2]));
        };
        for (var i = 0; i < nvs; i++) {
            ngeom.vertices.push(new THREE.Vector3(
                nverts[i][0], nverts[i][1], nverts[i][2]));
        };
        for (var i = 0; i < nfs; i++) {
            ngeom.faces.push(new THREE.Face3(
                nfaces[i][0], nfaces[i][1], nfaces[i][2]));
        };

        geom.mergeVertices();
        geom.computeFaceNormals();
        geom.computeVertexNormals();
        ngeom.mergeVertices();
        ngeom.computeFaceNormals();
        ngeom.computeVertexNormals();
        nmat.side = THREE.BackSide;
        this.scene.remove(this.surf);
        this.scene.remove(this.nsurf);
        this.surf = new THREE.Mesh(geom, mat);
        this.nsurf = new THREE.Mesh(ngeom, nmat);
        this.scene.add(this.surf);
        this.scene.add(this.nsurf);
    };

    AtomicThreeJS.prototype.update_cam_ctrl = function(center) {
        /*"""
        */
        var center = new THREE.Vector3(center[0], center[1], center[2]);
        var cx = center[0] * 2 + 15;
        var cy = center[1] * 2 + 15;
        var cz = center[2] * 2 + 15;
        this.camera.position.x = cx;
        this.camera.position.y = cy;
        this.camera.position.z = cz;
        this.camera.lookAt(center);
        this.controls.target = center;
    };

    AtomicThreeJS.prototype.update_camera_and_controls = function() {
        /*"""
        Automatically Update Camera and Controls
        `````````````````````````````````````````````````
        */
        this.atom.geometry.computeBoundingBox();
        var bbox = this.atom.geometry.boundingBox;
        var cx = (bbox.max.x - bbox.min.x) / 2;
        var cy = (bbox.max.y - bbox.min.y) / 2;
        var cz = (bbox.max.z - bbox.min.z) / 2;
        this.center = new THREE.Vector3(cx, cy, cz);
        cx = cx * 2 + 15;
        cy = cy * 2 + 15;
        cz = cz * 2 + 15;
        var nx = this.atom.geometry;
        //ny = this.atom.geometry.vertices.y.length;
        //nz = this.atom.geometry.vertices.x.length;
        this.cam_pos = new THREE.Vector3(cx, cy, cz);
        this.camera.position.x = this.cam_pos.x;
        this.camera.position.y = this.cam_pos.y;
        this.camera.position.z = this.cam_pos.z;
        this.camera.lookAt(this.center);
        this.controls.target = this.center;
    };

    return AtomicThreeJS;
});
