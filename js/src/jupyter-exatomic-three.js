// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
3D Visualization
##################

*/
"use strict";
var THREE = require("three");
var TBC = require("three-trackballcontrols");
var utils = require("./jupyter-exatomic-utils.js");


class App3D {
    /*"""
    App3D
    =========
    A 3D visualization application built on top of threejs
    that accepts an empty jupyter widget view and defines
    a simple threejs scene with a simplified threejs API.
    */

    constructor(view) {
        this.view = view;
        this.meshes = {"generic": [], "frame": [],
                       "contour": [], "field": [],
                       "atom": [], "two": []};
        this.ACTIVE = null;
        this.set_dims();
    };

    set_dims() {
        this.w = this.view.model.get("layout").get("width");
        this.h = this.view.model.get("layout").get("height");
        this.w2 = this.w / 2;
        this.h2 = this.h / 2;
    };

    init_camera() {
        var camera = new THREE.PerspectiveCamera(35, this.w / this.h, 1, 1000);
        return Promise.resolve(camera);
    };

    init_scene() {
        var scene = new THREE.Scene();
        var amlight = new THREE.AmbientLight(0xFFFFFF, 0.5);
        var dlight0 = new THREE.DirectionalLight(0xFFFFFF, 0.3);
        var dlight1 = new THREE.DirectionalLight(0xFFFFFF, 0.3);
        dlight0.position.set(-100, -100, -100);
        dlight1.position.set(100, 100, 100);
        scene.add(amlight);
        scene.add(dlight0);
        scene.add(dlight1);
        return Promise.resolve(scene);
    };

    init_renderer() {
        var renderer = new THREE.WebGLRenderer({antialias: true,
                                                alpha: true});
        renderer.setSize(this.w, this.h);
        renderer.autoClear = false;
        return Promise.resolve(renderer);
    };

    init_controls() {
        var controls = new TBC(this.camera,
                               this.renderer.domElement);
        controls.rotateSpeed = 10.0;
        controls.zoomSpeed = 5.0;
        controls.panSpeed = 0.5;
        controls.noZoom = false;
        controls.noPan = false;
        controls.staticMoving = true;
        controls.dynamicDampingFactor = 0.3;
        controls.keys = [65, 83, 68];
        controls.target = new THREE.Vector3(0.0, 0.0, 0.0);
        return Promise.resolve(controls)
    }

    init_raycaster() {
        return Promise.resolve(new THREE.Raycaster());
    };

    init_hud_scene() {
        return Promise.resolve(new THREE.Scene());
    };

    init_hud_camera() {
        var camera = new THREE.OrthographicCamera(
            -this.w2,  this.w2,
             this.h2, -this.h2, 1, 1500);
        camera.position.z = 1000;
        return Promise.resolve(camera);

    };

    init_mouse() {
        return Promise.resolve(new THREE.Vector2());
    };

    resize(w, h) {
        w = (w === undefined) ? this.w : w;
        h = (h === undefined) ? this.h : h;
        var w2 = w / 2;
        var h2 = h / 2;
        this.set_dims();
        this.set_hud();
        this.renderer.setSize(w, h);
        this.camera.aspect = w / h;
        this.hudcamera.left   = -w2;
        this.hudcamera.right  =  w2;
        this.hudcamera.top    =  h2;
        this.hudcamera.bottom = -h2;
        this.camera.updateProjectionMatrix();
        this.hudcamera.updateProjectionMatrix();
        this.controls.handleResize();
    };

    animate() {
        window.requestAnimationFrame(this.animate.bind(this));
        this.controls.update();
        this.resize();
        this.render();
    };

    render() {
        this.renderer.clear();
        this.renderer.render(this.scene, this.camera);
        this.renderer.clearDepth();
        this.renderer.render(this.hudscene, this.hudcamera);
    };

    finalize(promise) {
        return promise.then(this.animate.bind(this));
    };

    lighten_color(color) {
        // Need to check if color being passed is hex or int
        var R = (color >> 16);
        var G = (color >> 8 & 0x00FF);
        var B = (color & 0x0000FF);
        R = (R == 0) ? 110 : R + 76;
        G = (G == 0) ? 110 : G + 76;
        B = (B == 0) ? 110 : B + 76;
        R = R < 255 ? R < 1 ? 0 : R : 255;
        G = G < 255 ? G < 1 ? 0 : G : 255;
        B = B < 255 ? B < 1 ? 0 : B : 255;
        return (0x1000000 + R * 0x10000 + G * 0x100 + B);
    };

    highlight_active(intersects) {
        if (intersects.length > 0) {
            if (this.ACTIVE != intersects[0].object) {
                if (this.ACTIVE != null) {
                    if (this.ACTIVE.material.color) {
                        this.ACTIVE.material.color.setHex(this.ACTIVE.currentHex);
                    };
                };
                this.ACTIVE = intersects[0].object;
                if (this.ACTIVE.material.color) {
                    this.ACTIVE.currentHex = this.ACTIVE.material.color.getHex();
                    var newHex = this.lighten_color(this.ACTIVE.currentHex);
                    this.ACTIVE.material.color.setHex(newHex);
                };
                if (this.ACTIVE.name) { this.set_hud() }
                else { this.unset_hud() };
            };
        } else {
            if (this.ACTIVE != null) {
                if (this.ACTIVE.material.color) {
                    this.ACTIVE.material.color.setHex(this.ACTIVE.currentHex);
                };
            };
            this.ACTIVE = null;
            this.unset_hud();
        };
    };

    finalize_mouse_over() {
        var that = this;
        this.view.el.addEventListener('mousemove',
        function(event) {
            event.preventDefault();
            var pos = that.renderer.domElement.getBoundingClientRect();
            that.mouse.x =  ((event.clientX - pos.x) / that.w) * 2 - 1;
            that.mouse.y = -((event.clientY - pos.y) / that.h) * 2 + 1;
            that.ray.setFromCamera(that.mouse, that.camera);
            var intersects = that.ray.intersectObjects(that.scene.children);
            that.highlight_active(intersects);
        }, false);
    };

    init_promise() {
        var that = this;
        return Promise.all([
            this.init_scene().then(function(o) {that.scene = o}),
            Promise.all([
                this.init_camera().then(function(o) {that.camera = o}),
                this.init_renderer().then(function(o) {
                    that.renderer = o;
                    that.view.el.appendChild(that.renderer.domElement);
                })
            ]).then(this.init_controls.bind(this))
                .then(function(o) {
                    that.controls = o;
                    that.controls.addEventListener(
                        "change", that.render.bind(that));
            }).then(this.init_hud_canvas.bind(this))
                .then(function(o) {that.hudcanvas = o})
                .then(this.finalize_hud_canvas.bind(this)),
            this.init_raycaster().then(function(o) {that.ray = o}),
            this.init_hud_scene().then(function(o) {that.hudscene = o}),
            this.init_hud_camera().then(function(o) {that.hudcamera = o}),
            this.init_mouse().then(function(o) {that.mouse = o})
                .then(this.finalize_mouse_over.bind(this))
        ]);
    };

    init_hud_canvas() {
        var canvas = document.createElement("canvas");
        canvas.width = 1024;
        canvas.height = 1024;
        return Promise.resolve(canvas);
    };

    finalize_hud_canvas() {
        this.context = this.hudcanvas.getContext("2d");
        this.context.textAlign = "bottom";
        this.context.textBaseline = "left";
        this.context.font = "64px Arial";
        this.texture = new THREE.Texture(this.hudcanvas);
        this.texture.anisotropy = this.renderer.getMaxAnisotropy();
        this.texture.minFilter = THREE.NearestMipMapLinearFilter;
        this.texture.magFilter = THREE.NearestFilter;
        this.texture.needsUpdate = true;
        var material = new THREE.SpriteMaterial({map: this.texture});
        this.sprite = new THREE.Sprite(material);
        this.sprite.position.set(1000, 1000, 1000);
        this.sprite.scale.set(256, 256, 1);
        this.hudscene.add(this.sprite);
    };

    set_hud() {
        if ((this.ACTIVE === null) ||
            (this.ACTIVE.name === "") ||
            (this.ACTIVE instanceof THREE.Points)) { return };
        this.context.clearRect(0, 0, 1024, 1024);
        this.context.fillStyle = "rgba(245,245,245,0.9)";
        var w = this.context.measureText(this.ACTIVE.name).width;
        this.context.fillRect(512 - 2, 512 - 60, w + 6, 72);
        this.context.fillStyle = "rgba(0,0,0,0.95)";
        this.context.fillText(this.ACTIVE.name, 512, 512);
        this.sprite.position.set(-this.w2 + 2, -this.h2 + 4, 1);
        this.sprite.material.needsUpdate = true;
        this.texture.needsUpdate = true;
    };

    unset_hud() {
        this.sprite.position.set(1000, 1000, 1000);
    };


    clear_meshes(kind) {
        kind = (typeof kind !== "string") ? "all" : kind;
        for (var idx in this.meshes) {
            if ((kind === "all") || (kind === idx)) {
                for (var sub in this.meshes[idx]) {
                    this.scene.remove(this.meshes[idx][sub]);
                    delete this.meshes[idx][sub];
                };
            };
        };
    };

    add_meshes(kind) {
        kind = (typeof kind !== "string") ? "all" : kind;
        for (var idx in this.meshes) {
            if ((kind === "all") || (kind === idx)) {
                for (var sub in this.meshes[idx]) {
                    this.scene.add(this.meshes[idx][sub]);
                };
            };
        };
    };

    test_mesh() {
        /*"""
        test_mesh
        ---------------
        Example of a render
        */
        var geom = new THREE.IcosahedronGeometry(2, 1);
        var mat = new THREE.MeshBasicMaterial({color: 0x000000,
                                               wireframe: true});
        var mesh = new THREE.Mesh(geom, mat);
        mesh.name = "Icosahedron";
        return [mesh];
    };

    add_parametric_surface() {
        var func = function(ou, ov) {
            var u = 2 * Math.PI * ou;
            var v = 2 * Math.PI * ov;
            var x = Math.sin(u);
            var y = Math.cos(v);
            var z = Math.cos(u + v);
            return new THREE.Vector3(x, y, z)
        };
        var geom = new THREE.ParametricGeometry(func, 24, 24);
        var pmat = new THREE.MeshLambertMaterial({color: 'green', side: THREE.FrontSide});
        var nmat = new THREE.MeshLambertMaterial({color: 'yellow', side: THREE.FrontSide});
        var psurf = new THREE.Mesh(geom, pmat);
        var nsurf = new THREE.Mesh(geom, nmat);
        psurf.name = "Positive";
        nsurf.name = "Negative";
        return [psurf, nsurf];
    };

    close() {
        console.log("Disposing exatomic THREE objects.");
        for (var idx in this.meshes) {
            for (var sub in this.meshes[idx]) {
                if (this.meshes[idx][sub].geometry) {
                    this.meshes[idx][sub].geometry.dispose();
                };
                if (this.meshes[idx][sub].material) {
                    this.meshes[idx][sub].geometry.dispose();
                };
            };
        };
        this.texture.dispose();
        this.renderer.dispose();
    };

    save() {
        this.resize(1920, 1080);
        this.render();
        var image = this.renderer.domElement.toDataURL("image/png");
        this.resize();
        this.render();
        return image;
    };

    add_points(x, y, z, c, r) {
        /*"""
        add_points
        ---------------
        Create a point cloud from x, y, z coordinates

        Args:
            x (array-like): Array like object of x values
            y (array-like): Array like object of y values
            z (array-like): Array like object of z values
            c (object): List like colors corresponding to every object
            r (object): List like radii corresponding to every object

        Returns:
            points (THREE.Points): Reference to added points object

        */
        r = r || 1;
        c = c || 0x808080;
        c = (!c.length) ? utils.repeat_obj(c, n) : c;
        r = (!r.length) ? utils.repeat_obj(r, n) : r;
        c = App3D.prototype.flatten_color(c);
        r = new Float32Array(r);
        var geometry = new THREE.BufferGeometry();
        var material = new THREE.ShaderMaterial({
            vertexShader: App3D.prototype.vertex_shader,
            fragmentShader: App3D.prototype.point_frag_shader,
            transparent: true,
            opacity: 1.0,
            fog: true
        });
        var xyz = utils.create_float_array_xyz(x, y, z);
        var n = Math.floor(xyz.length / 3);
        geometry.addAttribute("position", new THREE.BufferAttribute(xyz, 3));
        geometry.addAttribute("color", new THREE.BufferAttribute(c, 3));
        geometry.addAttribute("size", new THREE.BufferAttribute(r, 1));
        var points = new THREE.Points(geometry, material);
        return [points];
    };

    add_lines(v0, v1, x, y, z, colors) {
        /*"""
        add_lines
        ------------
        Add lines between pairs of points.

        Args:
            v0 (array): Array of first vertex in pair
            v1 (array): Array of second vertex
            x (array): Position in x of vertices
            y (array): Position in y of vertices
            z (array): Position in z of vertices
            colors (array): Colors of vertices

        Returns:
            linesegs (THREE.LineSegments): Line segment objects
        */
        var material = new THREE.LineBasicMaterial({
            vertexColors: THREE.VertexColors,
            linewidth: 4,
        });
        var geometry = new THREE.Geometry();
        var n = v0.length;
        for (var i=0; i<n; i++) {
            var j = v0[i];
            var k = v1[i];
            var vector0 = new THREE.Vector3(x[j], y[j], z[j]);
            var vector1 = new THREE.Vector3(x[k], y[k], z[k]);
            geometry.vertices.push(vector0);
            geometry.vertices.push(vector1);
            geometry.colors.push(new THREE.Color(colors[j]));
            geometry.colors.push(new THREE.Color(colors[k]));
        };
        var lines = new THREE.LineSegments(geometry, material);
        return [lines];
    };

    add_spheres(x, y, z, c, r, l) {
        /*"""
        add_spheres
        ---------------
        Create a point cloud from x, y, z coordinates and colors and radii
        (optional).

        Args:
            x (array-like): Array like object of x values
            y (array-like): Array like object of y values
            z (array-like): Array like object of z values
            c (object): List like colors corresponding to every object
            r (object): List like radii corresponding to every object
            l (array-like): Array like object of atom labels

        Returns:
            spheres (list): List of THREE.Mesh objects
        */
        var n = 1;
        r = r || 1;
        c = c || x808080;
        n = x.length || n;
        n = y.length || n;
        n = z.length || n;
        c = (!c.hasOwnProperty("length")) ? utils.repeat_obj(c, n) : c;
        r = (!r.hasOwnProperty("length")) ? utils.repeat_obj(r, n) : r;
        l = (l == "") ? utils.repeat_obj(l, n) : l;
        var geometries = {};
        // var materials = {};
        for (var i=0; i<n; i++) {
            var color = c[i];
            var radius = r[i];
            if (!geometries.hasOwnProperty(color)) {
                geometries[color] = new THREE.SphereGeometry(radius, 20, 20);
            };
            // if (materials.hasOwnProperty(color) === false) {
            //     materials[color] = new THREE.MeshPhongMaterial({
            //         color: color,
            //         specular: color,
            //         shininess: 5
            //     });
            // };
        };
        var xyz = utils.create_float_array_xyz(x, y, z);
        var meshes = [];
        for (var i=0, i3=0; i<n; i++, i3+=3) {
            var color = c[i];
            var material = new THREE.MeshPhongMaterial({
                color: color, specular: color, shininess: 5});
            var mesh = new THREE.Mesh(geometries[color], material);
            if (l[i] != "") { mesh.name = l[i] };
            mesh.position.set(xyz[i3], xyz[i3+1], xyz[i3+2]);
            meshes.push(mesh);
        };
        return meshes;
    };

    add_cylinders(v0, v1, x, y, z, colors) {
        /*"""
        add_cylinders
        ------------
        Add lines between pairs of points.

        Args:
            v0 (array): Array of first vertex in pair
            v1 (array): Array of second vertex
            x (array): Position in x of vertices
            y (array): Position in y of vertices
            z (array): Position in z of vertices
            colors (array): Colors of vertices

        Returns:
            linesegs (THREE.LineSegments): Line segment objects
        */
        var r = 0.05;
        var mat = new THREE.MeshPhongMaterial({
            vertexColors: THREE.VertexColors,
            color: 0x606060,
            specular: 0x606060,
            shininess: 5});
        var meshes = [];
        var n = v0.length;
        for (var i=0; i<n; i++) {
            var j = v0[i];
            var k = v1[i];
            var vector0 = new THREE.Vector3(x[j], y[j], z[j]);
            var vector1 = new THREE.Vector3(x[k], y[k], z[k]);
            var direction = new THREE.Vector3().subVectors(vector0, vector1);
            var center = new THREE.Vector3().addVectors(vector0, vector1);
            center.divideScalar(2.0);
            var length = direction.length();
            var geometry = new THREE.CylinderGeometry(r, r, length);
            geometry.applyMatrix(new THREE.Matrix4().makeRotationX( Math.PI / 2));
            /*var nn = geometry.faces.length;
            var color0 = new THREE.Color(colors[j]);
            var color1 = new THREE.Color(colors[k]);
            geometry.colors.push(color0.clone());
            geometry.colors.push(color1.clone());
            for (var l=0; l<nn; l++) {
                geometry.faces[l].vertexColors[0] =
            };*/
            var mesh = new THREE.Mesh(geometry, mat.clone());
            mesh.name = (length * 0.52918).toFixed(4) + "\u212B";
            mesh.position.set(center.x, center.y, center.z);
            mesh.lookAt(vector1);
            meshes.push(mesh);
        };
        return meshes;
    };

    add_wireframe(vertices, color) {
        /*"""
        add_wireframe
        -----------------
        Create a wireframe object
        */
        color = color || 0x808080;
        var geometry = new THREE.Geometry();
        for (var v of vertices) {
            geometry.vertices.push(new THREE.Vector3(v[0], v[1], v[2]));
        };
        var material = new THREE.MeshBasicMaterial({
            transparent: true,
            opacity: 0.2,
            wireframeLinewidth: 8,
            wireframe: true
        });
        var cell = new THREE.Mesh(geometry, material);
        cell = new THREE.BoxHelper(cell);
        cell.material.color.set(color);
        return [cell];
    };

    get_camera() {
        return this.camera.toJSON();
    };

    set_camera(kwargs) {
        /*"""
        set_camera
        ------------------
        Set the camera in the default position and have it look at the origin.

        Args:
            kwargs: {"x": x, "y": y, ..., "ox": ox, ..., "rx": rx, ...}
        */
        kwargs = kwargs || {"x": 40.0, "y": 40.0, "z": 40.0};
        for (var key of ["x", "y", "z"]) {
            if (!kwargs.hasOwnProperty(key)) {
                kwargs[key] = 60.0;
            } else {
                if ((!kwargs[key]) ||
                    (isNaN(kwargs[key])) ||
                    (!isFinite(kwargs[key]))) {
                        kwargs[key] = 60.0}};
        };
        for (var key of ["rx", "ry", "rz"]) {
            if (!kwargs.hasOwnProperty(key)) {
                kwargs[key] = 0.5;
            } else {
                if ((!kwargs[key]) ||
                    (isNaN(kwargs[key])) ||
                    (!isFinite(kwargs[key]))) {
                        kwargs[key] = 0.5}};
        };
        for (var key of ["ox", "oy", "oz"]) {
            if (!kwargs.hasOwnProperty(key)) {
                kwargs[key] = 0.0;
            } else {
                if ((!kwargs[key]) ||
                    (isNaN(kwargs[key])) ||
                    (!isFinite(kwargs[key]))) {
                        kwargs[key] = 0.0}};
        };

        var x = kwargs["x"] + kwargs["rx"];
        var y = kwargs["y"] + kwargs["ry"];
        var z = kwargs["z"] + kwargs["rz"];
        var ox = kwargs["ox"];
        var oy = kwargs["oy"];
        var oz = kwargs["oz"];
        this.camera.position.set(x, y, z);
        this.target = new THREE.Vector3(ox, oy, oz);
        this.camera.lookAt(this.target);
        this.controls.target = this.target;
    };

    set_camera_from_camera(camera) {
        var loader = new THREE.ObjectLoader();
        var newcam = loader.parse(camera);
        this.camera = newcam;
        var that = this;
        this.init_controls()
            .then(function(o) {
                that.controls = o;
                that.controls.addEventListener(
                    "change", that.render.bind(that))
            });
        this.render();
    };

    set_camera_from_mesh(mesh, rx, ry, rz) {
        /*"""
        */
        rx = rx || 2.0;
        ry = ry || 2.0;
        rz = rz || 2.0;
        var position;
        if (mesh.geometry.type === "BufferGeometry") {
            position = mesh.geometry.attributes.position.array;
        } else {
            var n = mesh.geometry.vertices.length;
            position = new Float32Array(n * 3);
            for (var i=0; i<n; i+=3) {
                position[i] = mesh.geometry.vertices[i].x;
                position[i+1] = mesh.geometry.vertices[i].y;
                position[i+2] = mesh.geometry.vertices[i].z;
            }
        };
        var n = position.length / 3;
        var i = n;
        var oxyz = [0.0, 0.0, 0.0];
        while (i--) {
            oxyz[0] += position[3 * i];
            oxyz[1] += position[3 * i + 1];
            oxyz[2] += position[3 * i + 2];
        };
        oxyz[0] /= n;
        oxyz[1] /= n;
        oxyz[2] /= n;
        mesh.geometry.computeBoundingBox();
        var bbox = mesh.geometry.boundingBox;
        var xyz = bbox.max;
        xyz.x *= 1.2;
        xyz.x += rx;
        xyz.y *= 1.2;
        xyz.y += ry;
        xyz.z *= 1.2;
        xyz.z += rz;
        var kwargs = {"x": xyz.x, "y": xyz.y, "z": xyz.z,
                      "ox": oxyz[0], "oy": oxyz[1], "oz": oxyz[2]};
        this.set_camera(kwargs);
    };

    set_camera_from_scene() {
        /*"""
        set_camera_from_scene
        ------------------------
        */
        var bbox = new THREE.Box3().setFromObject(this.scene);
        var min = bbox.min;
        var max = bbox.max;
        var ox = (max.x + min.x) / 2;
        var oy = (max.y + min.y) / 2;
        var oz = (max.z + min.z) / 2;
        max.x *= 2.0;
        max.y *= 2.0;
        max.z *= 2.0;
        max.x = Math.max(max.x, 30);
        max.y = Math.max(max.y, 30);
        max.z = Math.max(max.z, 30);
        var kwargs = {"x": max.x, "y": max.y, "z": max.z,
                      "ox": ox, "oy": oy, "oz": oz};
        this.set_camera(kwargs);
    };

    add_contour(field, ncontour, clims, axis, val, colors) {
        /*"""
        add_contour
        ------------------------
        Create contour lines of a scalar field
        */
        var rets = this.march_squares(field, ncontour, clims, axis, val);
        var contours = rets["verts"];
        var labels = rets["contours"];
        var ngeom, pgem, ncont2;
        var meshes = [];
        var nmat = new THREE.LineBasicMaterial({color: colors["neg"],
                                                linewidth: 4});
        var pmat = new THREE.LineBasicMaterial({color: colors["pos"],
                                                linewidth: 4});
        for (var i=0, i2=i+ncontour; i<ncontour; i++, i2++) {
            var ngeom = new THREE.Geometry();
            var pgeom = new THREE.Geometry();
            for (var vec of contours[i]) {
                ngeom.vertices.push(new THREE.Vector3(vec[0], vec[1], vec[2]));
            };
            for (var vec of contours[i2]) {
                pgeom.vertices.push(new THREE.Vector3(vec[0], vec[1], vec[2]));
            };
            var nmesh = new THREE.LineSegments(ngeom, nmat.clone());
            var pmesh = new THREE.LineSegments(pgeom, pmat.clone());
            nmesh.name = labels[i].toFixed(6);
            pmesh.name = labels[i2].toFixed(6);
            meshes.push(nmesh);
            meshes.push(pmesh);
        };
        return meshes;
    };

    add_scalar_field(field, iso, sides, colors) {
        /*"""
        add_scalar_field
        -------------------------
        Create an isosurface of a scalar field.

        When given a scalar field, creating a surface requires selecting a set
        of vertices that intersect the provided field magnitude (isovalue).
        There are a couple of algorithms that do this.
        */
        var meshes;
        iso = iso || 0;
        sides = sides || 1;
        if (sides == 1) {
            meshes = this.march_cubes1(field, iso);
        } else if (sides == 2) {
            meshes = this.march_cubes2(field, iso, colors);
        };
        return meshes;
    };

    add_unit_axis(fill) {
        /*"""
        add_unit_axis
        ---------------
        Adds a unit length coordinate axis at the origin
        */
        var r = 0.05;
        var meshes = [];
        var axes = ["X", "Y", "Z"];
        var dirs = [new THREE.Vector3(1, 0, 0),
                    new THREE.Vector3(0, 1, 0),
                    new THREE.Vector3(0, 0, 1)];
        var origin = new THREE.Vector3(0, 0, 0);
        var cols = [0xFF0000, 0x00FF00, 0x0000FF];
        for (var i=0; i < 3; i++) {
            if (fill) {
                var dir = new THREE.Vector3().subVectors(dirs[i], origin);
                var cen = new THREE.Vector3().addVectors(dirs[i], origin);
                var cln = dirs[i].clone();
                cln.multiplyScalar(1.25);
                var cdir = new THREE.Vector3().subVectors(cln, dirs[i]);
                var ccen = new THREE.Vector3().addVectors(cln, dirs[i]);
                cen.divideScalar(2.0);
                ccen.divideScalar(2.0);
                var len = dir.length();
                var clen = cdir.length();
                var g = new THREE.CylinderGeometry(r, r, len);
                var c = new THREE.CylinderGeometry(0, 3 * r, clen);
                g.applyMatrix(new THREE.Matrix4().makeRotationX(Math.PI / 2));
                c.applyMatrix(new THREE.Matrix4().makeRotationX(Math.PI / 2));
                var mat = new THREE.MeshPhongMaterial({
                    vertexColors: THREE.VertexColors,
                    color: cols[i], specular: cols[i], shininess: 5});
                var bar = new THREE.Mesh(g, mat);
                var cone = new THREE.Mesh(c, mat.clone());
                bar.name = axes[i] + " axis";
                cone.name = axes[i] + " axis";
                bar.position.set(cen.x, cen.y, cen.z);
                cone.position.set(ccen.x, ccen.y, ccen.z);
                bar.lookAt(dir);
                cone.lookAt(cln);
                meshes.push(bar);
                meshes.push(cone);
            } else {
                var mesh = new THREE.ArrowHelper(dirs[i], origin, 1.0, cols[i]);
                mesh.line.material.linewidth = 4;
                meshes.push(mesh);
            };
        };
        return meshes;
    };

    march_cubes1(field, iso) {
        var start = new Date().getTime();
        var nnx = field.nx - 1;
        var nny = field.ny - 1;
        var nnz = field.nz - 1;
        var geom = new THREE.Geometry();
        for (var i=0; i<nnx; i++) {
            for (var j=0; j<nny; j++) {
                for (var k=0; k<nnz; k++) {
                    this.traverse_cube_single(field, i, j, k, geom, iso);
                };
            };
        };
        geom.mergeVertices();
        geom.computeFaceNormals();
        geom.computeVertexNormals();
        var frame = new THREE.Mesh(geom,
            new THREE.MeshBasicMaterial({color: 0x909090, wireframe: true}));
        var filled = new THREE.Mesh(geom,
            new THREE.MeshLambertMaterial({color:0x606060, side: THREE.DoubleSide}));
        var stop = new Date().getTime();
        var diff = stop - start;
        console.log("mc1: " + diff + " ms");
        return [filled, frame];
    };

    march_cubes2(field, iso, colors) {
        var start = new Date().getTime();
        var nnx, nny, nnz;
        var nnx = field.nx - 1;
        var nny = field.ny - 1;
        var nnz = field.nz - 1;
        var pgeom = new THREE.Geometry();
        var ngeom = new THREE.Geometry();
        for (var i = 0; i < nnx; i++) {
            for (var j = 0; j < nny; j++) {
                for (var k = 0; k < nnz; k++) {
                    this.traverse_cube_double(field, i, j, k, pgeom, ngeom, iso);
                };
            };
        };
        pgeom.mergeVertices();
        ngeom.mergeVertices();
        pgeom.computeFaceNormals();
        ngeom.computeFaceNormals();
        pgeom.computeVertexNormals();
        ngeom.computeVertexNormals();
        var pmesh = new THREE.Mesh(pgeom,
            new THREE.MeshPhongMaterial({
                color: colors["pos"], specular: colors["pos"],
                side: THREE.DoubleSide, shininess: 15}));
        var nmesh = new THREE.Mesh(ngeom,
            new THREE.MeshPhongMaterial({
                color: colors["neg"], specular: colors["neg"],
                side: THREE.DoubleSide, shininess: 15}));
        pmesh.name =  iso;
        nmesh.name = -iso;
        var stop = new Date().getTime();
        var diff = stop - start;
        console.log("mc2: " + diff + " ms");
        return [pmesh, nmesh];
    };

    march_squares(field, ncontour, clims, axis, val) {
        // Get the contour values given the limits and number of lines
        var z = App3D.prototype.compute_contours(ncontour, clims);
        var nc = z.length;

        // Get square of interest and "x, y" data
        var dat = App3D.prototype.get_square(field, val, axis);
        var sq = dat.dat;
        var x = dat.x;
        var y = dat.y;

        // Relative indices of adjacent square vertices
        var im = [0, 1, 1, 0];
        var jm = [0, 0, 1, 1];
        // Indexes case values for case value switch
        var castab = [[[0, 0, 8], [0, 2, 5], [7, 6, 9]],
                      [[0, 3, 4], [1, 3, 1], [4, 3, 0]],
                      [[9, 6, 7], [5, 2, 0], [8, 0, 0]]];

        // Probably don't need as many empty arrays as number of contours
        // but need at least two for positive and negative contours.
        var cverts = [];
        for(var i = 0; i < nc; i++) {
          cverts.push([]);
        };

        // Vars to hold the relative heights and x, y values of
        // the square of interest.
        var h = new Array(5);
        var xh = new Array(5);
        var yh = new Array(5);
        var sh = new Array(5);

        // m indexes the triangles in the square of interest
        var m1;
        var m2;
        var m3;
        // case value tells us whether or not to interpolate
        // or which x and y values to use to build the contour
        var cv;
        // Place holders for the x and y values of the square of interest
        var x1 = 0;
        var x2 = 0;
        var y1 = 0;
        var y2 = 0;
        // Find out whether or not the x, y range is relevant for
        // the given contour level
        var tmp1 = 0;
        var tmp2 = 0;
        var dmin;
        var dmax;

        var jdim = x.length - 2;
        var idim = y.length - 2;
        // Starting at the end of the y array
        for(var j = jdim; j >= 0; j--) {
            // But at the beginning of the x array
            for(var i = 0; i <= idim; i++) {
                // The smallest and largest values in square of interest
                tmp1 = Math.min(sq[  i  ][j], sq[  i  ][j + 1]);
                tmp2 = Math.min(sq[i + 1][j], sq[i + 1][j + 1]);
                dmin = Math.min(tmp1, tmp2);
                tmp1 = Math.max(sq[  i  ][j], sq[  i  ][j + 1]);
                tmp2 = Math.max(sq[i + 1][j], sq[i + 1][j + 1]);
                dmax = Math.max(tmp1, tmp2);
                // If outside all contour bounds, move along
                if (dmax < z[0] || dmin > z[nc - 1]) {
                    continue;
                };
                // For each contour
                for(var c = 0; c < nc; c++) {
                    // If outside individual contour bound, move along
                    if (z[c] < dmin || z[c] > dmax) {
                        continue;
                    };
                    // The box is considered as follows:
                    //
                    //  v4  +------------------+ v3
                    //      |        m=3       |
                    //      |                  |
                    //      |  m=2    X   m=2  |  center is v0
                    //      |                  |
                    //      |        m=1       |
                    //  v1  +------------------+ v2
                    //
                    // m indexes the triangle
                    // For each vertex in the square (5 of them)
                    for(var m = 4; m >= 0; m--) {
                        if (m > 0) {
                            h[m] = sq[i + im[m-1]][j + jm[m-1]] - z[c];
                            xh[m] = x[i + im[m-1]];
                            yh[m] = y[j + jm[m-1]];
                        } else {
                            h[0] = 0.25 * (h[1] + h[2] + h[3] + h[4]);
                            xh[0] = 0.5 * (x[i] + x[i + 1]);
                            yh[0] = 0.5 * (y[j] + y[j + 1]);
                        };
                        if (h[m] > 0.0) {
                            sh[m] = 1;
                        } else if (h[m] < 0.0) {
                            sh[m] = -1;
                        } else {
                            sh[m] = 0;
                        };
                    };
                    // Now loop over the triangles in the square to find the case
                    for(var m = 1; m <= 4; m++) {
                        m1 = m;
                        m2 = 0;
                        if (m != 4) {
                          m3 = m + 1;
                        } else {
                          m3 = 1;
                        };
                        if ((cv = castab[sh[m1] + 1][sh[m2] + 1][sh[m3] + 1]) == 0) {
                            continue;
                        };
                        // Assign x and y appropriately given the case
                        switch (cv) {
                            case 1: // Vertices 1 and 2
                                x1 = xh[m1];
                                y1 = yh[m1];
                                x2 = xh[m2];
                                y2 = yh[m2];
                                break;
                            case 2: // Vertices 2 and 3
                                x1 = xh[m2];
                                y1 = yh[m2];
                                x2 = xh[m3];
                                y2 = yh[m3];
                                break;
                            case 3: // Vertices 3 and 1
                                x1 = xh[m3];
                                y1 = yh[m3];
                                x2 = xh[m1];
                                y2 = yh[m1];
                                break;
                            case 4: // Vertex 1 and side 2-3
                                x1 = xh[m1];
                                y1 = yh[m1];
                                x2 = App3D.prototype.sect(m2, m3, h, xh);
                                y2 = App3D.prototype.sect(m2, m3, h, yh);
                                break;
                            case 5: // Vertex 2 and side 3-1
                                x1 = xh[m2];
                                y1 = yh[m2];
                                x2 = App3D.prototype.sect(m3, m1, h, xh);
                                y2 = App3D.prototype.sect(m3, m1, h, yh);
                                break;
                            case 6: // Vertex 3 and side 1-2
                                x1 = xh[m3];
                                y1 = yh[m3];
                                x2 = App3D.prototype.sect(m1, m2, h, xh);
                                y2 = App3D.prototype.sect(m1, m2, h, yh);
                                break;
                            case 7: // Sides 1-2 and 2-3
                                x1 = App3D.prototype.sect(m1, m2, h, xh);
                                y1 = App3D.prototype.sect(m1, m2, h, yh);
                                x2 = App3D.prototype.sect(m2, m3, h, xh);
                                y2 = App3D.prototype.sect(m2, m3, h, yh);
                                break;
                            case 8: // Sides 2-3 and 3-1
                                x1 = App3D.prototype.sect(m2, m3, h, xh);
                                y1 = App3D.prototype.sect(m2, m3, h, yh);
                                x2 = App3D.prototype.sect(m3, m1, h, xh);
                                y2 = App3D.prototype.sect(m3, m1, h, yh);
                                break;
                            case 9: // Sides 3-1 and 1-2
                                x1 = App3D.prototype.sect(m3, m1, h, xh);
                                y1 = App3D.prototype.sect(m3, m1, h, yh);
                                x2 = App3D.prototype.sect(m1, m2, h, xh);
                                y2 = App3D.prototype.sect(m1, m2, h, yh);
                                break;
                            default:
                                break;
                        };
                        // Push the x and y positions
                        cverts[c].push([x1, y1, val]);
                        cverts[c].push([x2, y2, val]);
                    };
                };
            };
        };
        // Default behavior of algorithm
        if (axis == "z") {
            return {"verts": cverts, "contours": z};
        // cverts is populated assuming x, y correspond to x, y
        // dimensions in the cube file, but to avoid a double nested
        // if else inside the tight O(N^2) loop, just move around the
        // elements accordingly afterwards according to the axis of
        // interest
        } else if (axis == "x") {
            var rear = [];
            for(var c = 0; c < nc; c++) {
                rear.push([]);
                var maxi = cverts[c].length;
                for(var v = 0; v < maxi; v++) {
                    rear[c].push([cverts[c][v][0], val, cverts[c][v][1]]);
                };
            };
        } else if (axis == "y") {
            var rear = [];
            for(var c = 0; c < nc; c++) {
                rear.push([]);
                var maxi = cverts[c].length;
                for(var v = 0; v < maxi; v++) {
                    rear[c].push([val, cverts[c][v][0], cverts[c][v][1]]);
                };
            };
        };
        return {"verts": rear, "contours": z};
    };
};

App3D.prototype.traverse_cube_single = function(field, i, j, k, geom, iso) {
    /*"""
    traverse_cube_single
    ------------------------
    Run the marching cubes algorithm finding the volumetric shell that is
    smaller than the isovalue.

    The marching cubes algorithm takes a scalar field and for each field
    vertex looks at the nearest indices (in an evenly space field this
    forms a cube), determines along what edges the scalar field is less
    than the isovalue, and creates new vertices along the edges of the
    field's cube. The at each point in the field, a cube is created with
    vertices numbered:
           4-------5
         / |     / |
        7-------6  |
        |  0----|--1
        | /     | /
        3-------2
    Field values are given for each field vertex. Edges are
    labeled as follows (see the lookup table below).

                                        4
           o-------o                o-------o
         / |     / |            7 / | 6   / |  5
        o-------o  |             o-------o  |
        |  o----|--o             |  o----|--o
      3 | /   0 | / 1            | /     | /
        o-------o                o-------o
           2
    Edges 8, 9, 10, and 11 wrap around (clockwise looking from the top
    as drawn) the vertical edges of the cube, with 8 being the vertical
    edge between vertex 0 and 4 (see above).

    Note:
        Scalar fields are assumed to be in row major order (also known C
        style, and implies that the last index is changing the fastest).

    */
    var offset, nx, ny, nz, oi, oj, ok, ii, jj, kk, valdx, cubdx;
    cubdx = 0;
    nx = field.nx;
    ny = field.ny;
    nz = field.nz;
    for (var m = 0; m < 8; m++) {
        offset = App3D.prototype.cube_vertices[m];
        oi = offset[0];
        oj = offset[1];
        ok = offset[2];
        ii = i + oi;
        jj = j + oj;
        kk = k + ok;
        App3D.prototype.vertex_coords[m] = new THREE.Vector3(
            field.x[ii], field.y[jj], field.z[kk]);
        var valdx = ( i * ny * nz +  j * nz +  k +
                     oi * ny * nz + oj * nz + ok);
        App3D.prototype.vertex_values[m] = field.values[valdx];
        if (App3D.prototype.vertex_values[m] <  iso) {
            cubdx |= App3D.prototype.bits[m];
        };
    };
    App3D.prototype.push_vertices_and_faces(geom, cubdx,
        App3D.prototype.vertex_coords, App3D.prototype.vertex_values, iso);
};

App3D.prototype.traverse_cube_double = function(field, i, j, k, pgeom, ngeom, iso) {
    var offset, nx, ny, nz, oi, oj, ok, ii, jj, kk, valdx, pcubdx, ncubdx;
    pcubdx = 0;
    ncubdx = 0;
    nx = field.nx;
    ny = field.ny;
    nz = field.nz;
    for (var m = 0; m < 8; m++) {
        offset = App3D.prototype.cube_vertices[m];
        oi = offset[0];
        oj = offset[1];
        ok = offset[2];
        ii = i + oi;
        jj = j + oj;
        kk = k + ok;
        App3D.prototype.vertex_coords[m] = new THREE.Vector3(
            field.x[ii], field.y[jj], field.z[kk]);
        var valdx = ( i * ny * nz +  j * nz +  k +
                     oi * ny * nz + oj * nz + ok);
        App3D.prototype.vertex_values[m] = field.values[valdx];
        if (App3D.prototype.vertex_values[m] <  iso) {
            pcubdx |= App3D.prototype.bits[m];
        };
        if (App3D.prototype.vertex_values[m] < -iso) {
            ncubdx |= App3D.prototype.bits[m];
        };
    };
    App3D.prototype.push_vertices_and_faces(pgeom, pcubdx,
        App3D.prototype.vertex_coords, App3D.prototype.vertex_values,  iso);
    App3D.prototype.push_vertices_and_faces(ngeom, ncubdx,
        App3D.prototype.vertex_coords, App3D.prototype.vertex_values, -iso);
};


App3D.prototype.push_vertices_and_faces = function(geom, idx, xyz, vals, iso) {
    var check, interp, vpair, a, b;
    var edge = App3D.prototype.edge_table[idx];
    var curfaces = App3D.prototype.tri_table[idx];
    if (edge !== 0) {
        interp = 0.5;
        for (var m = 0; m < 12; m++) {
            check = 1 << m;
            if (edge & check) {
                App3D.prototype.face_vertices[m] = geom.vertices.length;
                vpair = App3D.prototype.cube_edges[m];
                a = vpair[0];
                b = vpair[1];
                interp = (iso - vals[a]) / (vals[b] - vals[a]);
                geom.vertices.push(xyz[a].clone().lerp(xyz[b], interp));
            };
        };
        var nfaces = curfaces.length;
        for (var m = 0; m < nfaces; m += 3) {
            var i0 = App3D.prototype.face_vertices[curfaces[  m  ]];
            var i1 = App3D.prototype.face_vertices[curfaces[m + 1]];
            var i2 = App3D.prototype.face_vertices[curfaces[m + 2]];
            geom.faces.push(new THREE.Face3(i0, i1, i2));
        };
    };
};


App3D.prototype.sect = function(p1, p2, rh, w) {
    // Interpolate a value along the side of a square
    // by the relative heights at the square of interest
    return (rh[p2] * w[p1] - rh[p1] * w[p2]) / (rh[p2] - rh[p1]);
};

App3D.prototype.get_square = function(field, val, axis) {
    // Given a 1D array data that is ordered by x (outer increment)
    // then y (middle increment) and z (inner increment), determine
    // the appropriate z index given the z value to plot the contour
    // over. Then select that z-axis of the cube data and convert
    // it to a 2D array for processing in marching squares.
    var dat = [];
    var x, y, z, nx, ny, nz, xidx, yidx, plidx;
    if (axis === "z") {
        xidx = 0;
        yidx = 1;
        plidx = 2;
        x = field["x"];
        y = field["y"];
        z = field["z"];
        nx = field["nx"];
        ny = field["ny"];
        nz = field["nz"];
    } else if (axis === "x") {
        xidx = 0;
        yidx = 2;
        plidx = 1;
        x = field["x"];
        y = field["z"];
        z = field["y"];
        nx = field["nx"];
        ny = field["nz"];
        nz = field["ny"];
    } else if (axis === "y") {
        xidx = 1;
        yidx = 2;
        plidx = 0;
        x = field["y"];
        y = field["z"];
        z = field["x"];
        nx = field["ny"];
        ny = field["nz"];
        nz = field["nx"];
    };

    var idx = 0;
    var cur = 0;
    // var first = orig[plidx];
    // var amax = orig[plidx] + dims[plidx] * scale[plidx];
    var first = z[0];
    var amax = z[z.length - 1];

    if (val < first) {
        idx = 0;
    } else if (val > amax) {
        idx = nz - 1;
    } else {
        for(var k = 0; k < nz; k++) {
            cur = z[k];
            if (Math.abs(val - z[k]) < Math.abs(val - first)) {
                first = z[k];
                idx = k;
            };
        };
    };

    if (axis === "z") {
        for (var xi=0; xi < nx; xi++) {
            var tmp = [];
            for(var yi = 0; yi < ny; yi++) {
                tmp.push(field["values"][(nx * xi + yi) * ny + idx]);
            };
            dat.push(tmp);
        };
    } else if (axis === "x") {
        for (var xi=0; xi<nx; xi++) {
            var tmp = [];
            for(var zi=0; zi<nz; zi++) {
                tmp.push(field["values"][(nx * xi + idx) * nz + zi]);
            };
            dat.push(tmp);
        };
    } else if (axis === "y") {
        for (var yi=0; yi<ny; yi++) {
            var tmp = [];
            for(var zi=0; zi<nz; zi++) {
                tmp.push(field["values"][(nx * idx + yi) * ny + zi]);
            };
            dat.push(tmp);
        };
    };
    return {dat: dat, x: x, y: y};
};


App3D.prototype.compute_contours = function(ncontour, clims) {
    // Determine the spacing for contour lines given the limits
    // (exponents of base ten) and the number of contour lines
    var tmp1 = [];
    var tmp2 = [];
    var d = (clims[1] - clims[0]) / ncontour;
    for(var i = 0; i < ncontour; i++) {
        tmp1.push(-Math.pow(10, -i * d));
        tmp2.push(Math.pow(10, -i * d));
    };
    tmp2.reverse()
    return tmp1.concat(tmp2);
};

App3D.prototype.flatten_color = function(colors) {
    var n = colors.length;
    var flat = new Float32Array(n * 3);
    for (var i=0, i3=0; i<n; i++, i3+=3) {
        var color = new THREE.Color(colors[i]);
        flat[i3] = color.r;
        flat[i3+1] = color.g;
        flat[i3+2] = color.b;
    };
    return flat;
};


// These are shaders written in GLSL (GLslang: OpenGL Shading Language).
// This code is executed on the GPU.
App3D.prototype.vertex_shader = "\
    attribute float size;\
    attribute vec3 color;\
    varying vec3 vColor;\
    \
    void main() {\
        vColor = color;\
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);\
        gl_PointSize = size * (2000.0 / length(mvPosition.xyz));\
        gl_Position = projectionMatrix * mvPosition;\
    }\
";

App3D.prototype.point_frag_shader = "\
    varying vec3 vColor;\
    \
    void main() {\
        if (length(gl_PointCoord * 2.0 - 1.0) > 1.0)\
            discard;\
        gl_FragColor = vec4(vColor, 1.0);\
    }\
";

App3D.prototype.circle_frag_shader = "\
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

App3D.prototype.line_frag_shader = "\
    uniform vec3 color;\
    uniform float opacity;\
    \
    vary vec3 vColor;\
    void main() {\
        gl_FragColor = vec4(vColor * color, opacity);\
    }\
";

App3D.prototype.line_vertex_shader = "\
    uniform float amplitude;\
    attribute vec3 displacement;\
    attribute vec3 customColor;\
    varying vec3 vColor;\
    \
    void main() {\
        vec3 newPosition = position + amplitude * displacement;\
        vColor = customColor;\
        gl_Position = projectionMatrix * modelViewMatrix * vec4(newPosition, 1.0);\
    }\
";

App3D.prototype.edge_table = new Uint32Array([
    0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0
]);

App3D.prototype.tri_table = [
    [],
    [0, 8, 3],
    [0, 1, 9],
    [1, 8, 3, 9, 8, 1],
    [1, 2, 10],
    [0, 8, 3, 1, 2, 10],
    [9, 2, 10, 0, 2, 9],
    [2, 8, 3, 2, 10, 8, 10, 9, 8],
    [3, 11, 2],
    [0, 11, 2, 8, 11, 0],
    [1, 9, 0, 2, 3, 11],
    [1, 11, 2, 1, 9, 11, 9, 8, 11],
    [3, 10, 1, 11, 10, 3],
    [0, 10, 1, 0, 8, 10, 8, 11, 10],
    [3, 9, 0, 3, 11, 9, 11, 10, 9],
    [9, 8, 10, 10, 8, 11],
    [4, 7, 8],
    [4, 3, 0, 7, 3, 4],
    [0, 1, 9, 8, 4, 7],
    [4, 1, 9, 4, 7, 1, 7, 3, 1],
    [1, 2, 10, 8, 4, 7],
    [3, 4, 7, 3, 0, 4, 1, 2, 10],
    [9, 2, 10, 9, 0, 2, 8, 4, 7],
    [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4],
    [8, 4, 7, 3, 11, 2],
    [11, 4, 7, 11, 2, 4, 2, 0, 4],
    [9, 0, 1, 8, 4, 7, 2, 3, 11],
    [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1],
    [3, 10, 1, 3, 11, 10, 7, 8, 4],
    [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4],
    [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3],
    [4, 7, 11, 4, 11, 9, 9, 11, 10],
    [9, 5, 4],
    [9, 5, 4, 0, 8, 3],
    [0, 5, 4, 1, 5, 0],
    [8, 5, 4, 8, 3, 5, 3, 1, 5],
    [1, 2, 10, 9, 5, 4],
    [3, 0, 8, 1, 2, 10, 4, 9, 5],
    [5, 2, 10, 5, 4, 2, 4, 0, 2],
    [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8],
    [9, 5, 4, 2, 3, 11],
    [0, 11, 2, 0, 8, 11, 4, 9, 5],
    [0, 5, 4, 0, 1, 5, 2, 3, 11],
    [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5],
    [10, 3, 11, 10, 1, 3, 9, 5, 4],
    [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10],
    [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3],
    [5, 4, 8, 5, 8, 10, 10, 8, 11],
    [9, 7, 8, 5, 7, 9],
    [9, 3, 0, 9, 5, 3, 5, 7, 3],
    [0, 7, 8, 0, 1, 7, 1, 5, 7],
    [1, 5, 3, 3, 5, 7],
    [9, 7, 8, 9, 5, 7, 10, 1, 2],
    [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3],
    [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2],
    [2, 10, 5, 2, 5, 3, 3, 5, 7],
    [7, 9, 5, 7, 8, 9, 3, 11, 2],
    [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11],
    [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7],
    [11, 2, 1, 11, 1, 7, 7, 1, 5],
    [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11],
    [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0],
    [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0],
    [11, 10, 5, 7, 11, 5],
    [10, 6, 5],
    [0, 8, 3, 5, 10, 6],
    [9, 0, 1, 5, 10, 6],
    [1, 8, 3, 1, 9, 8, 5, 10, 6],
    [1, 6, 5, 2, 6, 1],
    [1, 6, 5, 1, 2, 6, 3, 0, 8],
    [9, 6, 5, 9, 0, 6, 0, 2, 6],
    [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8],
    [2, 3, 11, 10, 6, 5],
    [11, 0, 8, 11, 2, 0, 10, 6, 5],
    [0, 1, 9, 2, 3, 11, 5, 10, 6],
    [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11],
    [6, 3, 11, 6, 5, 3, 5, 1, 3],
    [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6],
    [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9],
    [6, 5, 9, 6, 9, 11, 11, 9, 8],
    [5, 10, 6, 4, 7, 8],
    [4, 3, 0, 4, 7, 3, 6, 5, 10],
    [1, 9, 0, 5, 10, 6, 8, 4, 7],
    [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4],
    [6, 1, 2, 6, 5, 1, 4, 7, 8],
    [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7],
    [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6],
    [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9],
    [3, 11, 2, 7, 8, 4, 10, 6, 5],
    [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11],
    [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6],
    [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6],
    [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6],
    [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11],
    [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7],
    [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9],
    [10, 4, 9, 6, 4, 10],
    [4, 10, 6, 4, 9, 10, 0, 8, 3],
    [10, 0, 1, 10, 6, 0, 6, 4, 0],
    [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10],
    [1, 4, 9, 1, 2, 4, 2, 6, 4],
    [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4],
    [0, 2, 4, 4, 2, 6],
    [8, 3, 2, 8, 2, 4, 4, 2, 6],
    [10, 4, 9, 10, 6, 4, 11, 2, 3],
    [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6],
    [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10],
    [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1],
    [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3],
    [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1],
    [3, 11, 6, 3, 6, 0, 0, 6, 4],
    [6, 4, 8, 11, 6, 8],
    [7, 10, 6, 7, 8, 10, 8, 9, 10],
    [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10],
    [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0],
    [10, 6, 7, 10, 7, 1, 1, 7, 3],
    [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7],
    [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9],
    [7, 8, 0, 7, 0, 6, 6, 0, 2],
    [7, 3, 2, 6, 7, 2],
    [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7],
    [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7],
    [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11],
    [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1],
    [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6],
    [0, 9, 1, 11, 6, 7],
    [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0],
    [7, 11, 6],
    [7, 6, 11],
    [3, 0, 8, 11, 7, 6],
    [0, 1, 9, 11, 7, 6],
    [8, 1, 9, 8, 3, 1, 11, 7, 6],
    [10, 1, 2, 6, 11, 7],
    [1, 2, 10, 3, 0, 8, 6, 11, 7],
    [2, 9, 0, 2, 10, 9, 6, 11, 7],
    [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8],
    [7, 2, 3, 6, 2, 7],
    [7, 0, 8, 7, 6, 0, 6, 2, 0],
    [2, 7, 6, 2, 3, 7, 0, 1, 9],
    [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6],
    [10, 7, 6, 10, 1, 7, 1, 3, 7],
    [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8],
    [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7],
    [7, 6, 10, 7, 10, 8, 8, 10, 9],
    [6, 8, 4, 11, 8, 6],
    [3, 6, 11, 3, 0, 6, 0, 4, 6],
    [8, 6, 11, 8, 4, 6, 9, 0, 1],
    [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6],
    [6, 8, 4, 6, 11, 8, 2, 10, 1],
    [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6],
    [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9],
    [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3],
    [8, 2, 3, 8, 4, 2, 4, 6, 2],
    [0, 4, 2, 4, 6, 2],
    [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8],
    [1, 9, 4, 1, 4, 2, 2, 4, 6],
    [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1],
    [10, 1, 0, 10, 0, 6, 6, 0, 4],
    [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3],
    [10, 9, 4, 6, 10, 4],
    [4, 9, 5, 7, 6, 11],
    [0, 8, 3, 4, 9, 5, 11, 7, 6],
    [5, 0, 1, 5, 4, 0, 7, 6, 11],
    [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5],
    [9, 5, 4, 10, 1, 2, 7, 6, 11],
    [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5],
    [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2],
    [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6],
    [7, 2, 3, 7, 6, 2, 5, 4, 9],
    [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7],
    [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0],
    [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8],
    [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7],
    [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4],
    [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10],
    [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10],
    [6, 9, 5, 6, 11, 9, 11, 8, 9],
    [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5],
    [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11],
    [6, 11, 3, 6, 3, 5, 5, 3, 1],
    [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6],
    [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10],
    [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5],
    [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3],
    [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2],
    [9, 5, 6, 9, 6, 0, 0, 6, 2],
    [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8],
    [1, 5, 6, 2, 1, 6],
    [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6],
    [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0],
    [0, 3, 8, 5, 6, 10],
    [10, 5, 6],
    [11, 5, 10, 7, 5, 11],
    [11, 5, 10, 11, 7, 5, 8, 3, 0],
    [5, 11, 7, 5, 10, 11, 1, 9, 0],
    [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1],
    [11, 1, 2, 11, 7, 1, 7, 5, 1],
    [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11],
    [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7],
    [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2],
    [2, 5, 10, 2, 3, 5, 3, 7, 5],
    [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5],
    [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2],
    [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2],
    [1, 3, 5, 3, 7, 5],
    [0, 8, 7, 0, 7, 1, 1, 7, 5],
    [9, 0, 3, 9, 3, 5, 5, 3, 7],
    [9, 8, 7, 5, 9, 7],
    [5, 8, 4, 5, 10, 8, 10, 11, 8],
    [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0],
    [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5],
    [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4],
    [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8],
    [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11],
    [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5],
    [9, 4, 5, 2, 11, 3],
    [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4],
    [5, 10, 2, 5, 2, 4, 4, 2, 0],
    [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9],
    [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2],
    [8, 4, 5, 8, 5, 3, 3, 5, 1],
    [0, 4, 5, 1, 0, 5],
    [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5],
    [9, 4, 5],
    [4, 11, 7, 4, 9, 11, 9, 10, 11],
    [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11],
    [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11],
    [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4],
    [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2],
    [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3],
    [11, 7, 4, 11, 4, 2, 2, 4, 0],
    [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4],
    [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9],
    [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7],
    [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10],
    [1, 10, 2, 8, 7, 4],
    [4, 9, 1, 4, 1, 7, 7, 1, 3],
    [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1],
    [4, 0, 3, 7, 4, 3],
    [4, 8, 7],
    [9, 10, 8, 10, 11, 8],
    [3, 0, 9, 3, 9, 11, 11, 9, 10],
    [0, 1, 10, 0, 10, 8, 8, 10, 11],
    [3, 1, 10, 11, 3, 10],
    [1, 2, 11, 1, 11, 9, 9, 11, 8],
    [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9],
    [0, 2, 11, 8, 0, 11],
    [3, 2, 11],
    [2, 3, 8, 2, 8, 10, 10, 8, 9],
    [9, 10, 2, 0, 9, 2],
    [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8],
    [1, 10, 2],
    [1, 3, 8, 9, 1, 8],
    [0, 9, 1],
    [0, 3, 8],
    []
];

App3D.prototype.cube_vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                 [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]];
App3D.prototype.cube_edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6],
                              [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
App3D.prototype.bits = [1, 2, 4, 8, 16, 32, 64, 128];
App3D.prototype.face_vertices = new Int32Array(12);
App3D.prototype.vertex_coords = new Array(8);
App3D.prototype.vertex_values = new Float32Array(8);


module.exports = {
    App3D: App3D
};
