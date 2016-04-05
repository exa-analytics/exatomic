/*"""
===============
test.js
===============
Tests for universe.js
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/app.three': {
            exports: 'app3d'
        },
        'nbextensions/exa/app.gui': {
            exports: 'gui'
        },

        'nbextensions/exa/field': {
            exports: 'field'
        },
    },
});


define([
    'nbextensions/exa/app.three',
    'nbextensions/exa/app.gui',
    'nbextensions/exa/field'
], function(app3d, gui, field) {
    class TestGUI extends gui.ContainerGUI {
        /*"""
        */
        init() {
            var self = this;
            this.buttons = {
                'run all tests': function() {
                    console.log('run all clicked');
                    self.run_all_tests();
                },
            };

            this.sliders = {
                'isovalue': 0.03,
            };

            this.run_all = this.ui.add(this.buttons, 'run all tests');
            this.isovalue = this.ui.add(this.sliders, 'isovalue', 0.0001, 5);
            this.isovalue.onFinishChange(function(value) {
                console.log(value);
            });
        };

        run_all_tests() {
            console.log('running all tests');
        };
    };

    class TestApp {
        constructor(view) {
            this.view = view;
            this.app3d = new app3d.ThreeJSApp(this.view.canvas);
            this.field = new field.ScalarField({
                xmin: -10.0, xmax: 10.0, dx: 0.5,
                ymin: -10.0, ymax: 10.0, dy: 0.5,
                zmin: -10.0, zmax: 10.0, dz: 0.5
            }, field.torus);
            this.gui = new TestGUI(this);
            console.log(this.field);
            this.isovalue = 0.5;
            this.plot();
        };

        plot() {
            /*var x = [];
            var y = [];
            var z = [];
            for (let i of this.field.x) {
                for (let j of this.field.y) {
                    for (let k of this.field.z) {
                        x.push(i);
                        y.push(j);
                        z.push(k);
                    };
                };
            };
            this.points = this.app3d.add_points(x, y, z, 1, 0.3);
            this.app3d.set_camera_from_mesh(this.points);*/
            console.log(Math.min(...this.field.values));
            console.log(Math.max(...this.field.values));
            console.log(this.isovalue);
            this.field_mesh = this.app3d.add_scalar_field(this.field, this.isovalue, 2);
            console.log(this.field_mesh);
            //console.log(this.field_mesh.geometry.vertices.length);
            //console.log(this.field_mesh.geometry.faces.length);
            //console.log('-------------');
            //console.log(this.field_mesh.geometry.vertices.length);
            //console.log(this.field_mesh.geometry.faces.length);
            /*var data = this.field.values;
            var dims = [this.field.nx, this.field.ny, this.field.nz];
            var orig = [this.field.x[0], this.field.y[0], this.field.z[0]];
            var scale = [this.field.dx, this.field.dy, this.field.dz];
            console.log(dims);
            console.log(orig);
            console.log(scale);

            var results = mc.MarchingCubes(data, dims, orig, scale, isolevel);
            var meshes = this.app3d.add_temp(
                results.vertices,
                results.nvertices,
                results.faces,
                results.nfaces
            );
            console.log(results.vertices.length);
            console.log(results.nvertices.length);
            console.log(results.faces.length);
            console.log(results.nfaces.length);
            console.log(results);
            console.log(meshes);*/
            this.app3d.set_camera();
            //this.app3d.set_camera_from_mesh(this.field_mesh);
            console.log(this.obj);
        };

        resize() {
            this.app3d.resize();
        };

        func() {
            for (let i=0; i<10; i++) {
                console.log('inner i' + i.toString());
            };
        };
    };

    TestApp.prototype.obj = 42;

    return {TestApp: TestApp};
});
