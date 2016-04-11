/*"""
==================
test.js
==================
Test visualization application for the universe container (atomic package).
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/apps/app3d': {
            exports: 'App3D'
        },

        'nbextensions/exa/apps/gui': {
            exports: 'ContainerGUI'
        },

        'nbextensions/exa/atomic/ao': {
            exports: 'AO'
        },
    },
});


define([
    'nbextensions/exa/apps/app3d',
    'nbextensions/exa/apps/gui',
    'nbextensions/exa/atomic/ao'
], function(App3D, ContainerGUI, AO) {
    class UniverseTestApp {
        /*"""
        UniverseTestApp
        ==================
        A test application for the universe container (atomic package).
        */
        constructor(view) {
            /*"""
            constructor
            --------------
            Args:
                view: Backbone.js view (DOMWidgetView container representation)
            */
            console.log('here11');
            this.view = view;
            this.view.create_canvas();
            this.axis = [];
            this.active_objs = [];
            this.create_app3d();
            this.create_gui();
            this.view.container.append(this.gui.domElement);
            this.view.container.append(this.gui.custom_css);
            this.view.container.append(this.view.canvas);
            var view_self = this.view;
            this.view.on('displayed', function() {
                view_self.app.app3d.animate();
                view_self.app.app3d.controls.handleResize();
            });
        };

        create_app3d() {
            this.app3d = new App3D(this.view.canvas);
            this.axis = this.app3d.add_unit_axis();
            this.dimensions = {
                'xmin': -20.0,
                'xmax': 20.0,
                'ymin': -20.0,
                'ymax': 20.0,
                'zmin': -20.0,
                'zmax': 20.0,
                'dx': 0.5,
                'dy': 0.5,
                'dz': 0.5
            };
            this.render_orbital();
            this.app3d.set_camera({x: 5.5, y: 5.5, z: 5.5});
        };

        create_gui() {
            /*"""
            create_gui
            --------------
            Creates the standard style container gui instance and populates
            with relevant controls for this application.
            */
            var self = this;
            this.gui = new ContainerGUI(this.view.gui_width);

            this.top_level = {
                'demo': 'atomic orbitals',
                'demos': ['atomic orbitals', 'cube file', 'md trajectory'],
                'play': function() {
                    console.log('pushed play');
                },
                'fps': 24,
            };
            this.top_level['demo_dropdown'] = this.gui.add(this.top_level, 'demo', this.top_level['demos']);
            this.top_level['play_button'] = this.gui.add(this.top_level, 'play');
            this.top_level['fps_slider'] = this.gui.add(this.top_level, 'fps', 1, 60);
            this.atorb = {
                'which': '1s',
                'orbitals': ['1s', '2s', '2px', '2py', '2pz',
                             '3s', '3px', '3py', '3pz',
                             '3dz2', '3dxz', '3dyz', '3dx2-y2', '3dxy'],
                'isovalue': 0.005
            };
            this.surf = this.gui.addFolder('isosurface');
            this.atorb['which_dropdown'] = this.surf.add(this.atorb, 'which', this.atorb['orbitals']);
            this.atorb['isovalue_slider'] = this.surf.add(this.atorb, 'isovalue', 0.0, 0.4);
            this.surf.open();

            this.atorb['isovalue_slider'].onFinishChange(function(value) {
                self.atorb['isovalue'] = value;
                self.render_orbital();
            });

            this.atorb['which_dropdown'].onFinishChange(function(value) {
                self.atorb['which'] = value;
                self.render_orbital();
            });
        };

        render_orbital() {
            var iso = 0.005;
            var which = '1s';
            if (this.atorb !== undefined) {
                iso = this.atorb['isovalue'];
                which = this.atorb['which'];
            };
            this.field = new AO(this.dimensions, which);
            this.app3d.remove_meshes(this.active_objs);
            this.active_objs = this.app3d.add_scalar_field(this.field, iso, 2);
        };

        remove_surface() {
        };

        resize() {
            this.app3d.resize();
        };
    };

    return UniverseTestApp;
});

/*
define([
    'nbextensions/exa/app.three',
    'nbextensions/exa/app.gui',
    'nbextensions/exa/field'
], function(app3d, gui, field) {
    class TestGUI extends gui.ContainerGUI {
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

    class UniverseTestApp {
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
            this.app3d.set_camera_from_mesh(this.points);
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
            console.log(meshes);
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

    UniverseTestApp.prototype.obj = 42;

    return {UniverseTestApp: UniverseTestApp};
});
*/
