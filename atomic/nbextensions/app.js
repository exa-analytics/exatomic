/*"""
===========================
Universe Application
===========================
This module defines the JavaScript application that is loaded when a user
requests the HTML representation of a universe data container.
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/apps/gui': {
            exports: 'ContainerGUI'
        },

        'nbextensions/exa/apps/app3d': {
            exports: 'App3D'
        },

        'nbextensions/exa/utility': {
            exports: 'utility'
        },

        'nbextensions/exa/atomic/field': {
            exports: 'CubeField'
        }
    },
});


define([
    'nbextensions/exa/apps/gui',
    'nbextensions/exa/apps/app3d',
    'nbextensions/exa/utility',
    'nbextensions/exa/atomic/field'
], function(ContainerGUI, App3D, utility, CubeField) {
    class UniverseApp {
        /*"""
        UniverseApp
        =============
        Notebook widget application for visualization of the universe container.
        */
        constructor(view) {
            this.view = view;
            this.view.create_canvas();
            this.update_vars();
            this.app3d = new App3D(this.view.canvas);
            this.create_gui();

            this.update_fields();
            this.render_current_frame();

            this.view.container.append(this.gui.domElement);
            this.view.container.append(this.gui.custom_css);
            this.view.container.append(this.view.canvas);
            var view_self = this.view;
            this.app3d.render();
            this.view.on('displayed', function() {
                view_self.app.app3d.animate();
                view_self.app.app3d.controls.handleResize();
            });
        };

        resize() {
            this.app3d.resize();
        };

        gv(obj, index) {
            /*"""
            gv
            ---------------
            Helper function for retrieving data from the view.
            */
            if (obj === undefined) {
                return undefined;
            } else {
                var value = obj[index];
                if (value === undefined) {
                    return obj;
                };
                return value;
            };
        };

        update_vars() {
            /*"""
            init_vars
            ----------------
            Set up some application variables.
            */
            // frame
            if (typeof this.view.framelist === 'number') {
                this.framelist = [this.view.framelist];
            } else {
                this.framelist = this.view.framelist;
            };
            this.idx = 0;
            this.num_frames = this.framelist.length;
            this.last_index = this.num_frames - 1;
            this.current_frame = this.framelist[this.idx];

            this.atoms_meshes = [];
            this.bonds_meshes = [];
            this.cell_meshes = [];
            this.field_meshes = [];
        };

        create_gui() {
            /*"""
            create_gui
            ------------------
            Create the application's control set.
            */
            var self = this;
            this.playing = false;
            this.play_id = undefined;
            this.gui = new ContainerGUI(this.view.gui_width);

            this.top = {
                'pause': function() {
                    self.playing = false;
                    clearInterval(self.play_id);
                },
                'play': function() {
                    if (self.playing === true) {
                        self.top.pause()
                    } else {
                        self.playing = true;
                        if (self.idx === self.last_index) {
                            self.top.index_slider.setValue(0);
                        };
                        self.play_id = setInterval(function() {
                            if (self.idx < self.last_index) {
                                self.top.index_slider.setValue(self.idx+1);
                            } else {
                                self.top.pause();
                            };
                        }, 1000 / self.top.fps);
                    };
                },
                'index': 0,
                'frame': this.current_frame,
                'fps': this.view.fps,
            };
            this.top['play_button'] = this.gui.add(this.top, 'play');
            this.top['index_slider'] = this.gui.add(this.top, 'index', 0, this.last_index);
            this.top['frame_dropdown'] = this.gui.add(this.top, 'frame', this.framelist);
            this.top['fps_slider'] = this.gui.add(this.top, 'fps').min(0).max(240).step(1);
            this.top.index_slider.onChange(function(index) {
                self.idx = index;
                self.current_frame = self.framelist[self.idx];
                self.top['index'] = self.idx;
                self.top['frame'] = self.current_frame;
                self.top.frame_dropdown.setValue(self.current_frame);
                self.update_fields();
                self.render_current_frame();
            });
            this.top.index_slider.onFinishChange(function(index) {
                self.update_fields();
            });
            this.top.fps_slider.onFinishChange(function(value) {
                self.top.fps = value;
            });

            this.display = {
                'cell': false,
            };
            this.display['folder'] = this.gui.addFolder('display');
            this.display['cell_checkbox'] = this.display.folder.add(this.display, 'cell');

            this.display.cell_checkbox.onFinishChange(function(value) {
                self.display.cell = value;
                self.render_cell();
            });

            this.fields = {
                'isovalue': 0.03,
                'field': '',
                'cur_fields': []
            };

            this.fields['folder'] = this.gui.addFolder('fields');
            this.fields['isovalue_slider'] = this.fields.folder.add(this.fields, 'isovalue', 0.0001, 0.5);
            this.fields['field_dropdown'] = this.fields.folder.add(this.fields, 'field', this.fields['cur_fields']);
            this.fields.field_dropdown.onFinishChange(function(field_index) {
                self.fields['field'] = field_index;
                console.log(field_index);
                self.render_field();
            });
            this.fields.isovalue_slider.onFinishChange(function(value) {
                self.fields.isovalue = value;
                self.render_field();
            });
        };

        update_fields() {
            /*"""
            update_fields
            ---------------
            Updates available fields for the given frame and selection
            */
            var self = this;
            console.log('update_fields');
            var field_indices = this.gv(this.view.field_indices, this.idx);
            this.fields['cur_fields'] = field_indices;
            this.fields.folder.__controllers[1].remove();
            this.fields['field_dropdown'] = this.fields.folder.add(this.fields, 'field', this.fields['cur_fields']);
            this.fields.field_dropdown.onFinishChange(function(field_index) {
                self.fields['field'] = field_index;
                self.render_field();
            });
        };

        render_field() {
            /*"""
            render_field
            --------------
            */
            console.log('render_field');
            var nx = this.gv(this.view.ufield3d_nx, this.idx);
            var ny = this.gv(this.view.ufield3d_ny, this.idx);
            var nz = this.gv(this.view.ufield3d_nz, this.idx);
            var ox = this.gv(this.view.ufield3d_ox, this.idx);
            var oy = this.gv(this.view.ufield3d_oy, this.idx);
            var oz = this.gv(this.view.ufield3d_oz, this.idx);
            var xi = this.gv(this.view.ufield3d_xi, this.idx);
            var xj = this.gv(this.view.ufield3d_xj, this.idx);
            var xk = this.gv(this.view.ufield3d_xk, this.idx);
            var yi = this.gv(this.view.ufield3d_yi, this.idx);
            var yj = this.gv(this.view.ufield3d_yj, this.idx);
            var yk = this.gv(this.view.ufield3d_yk, this.idx);
            var zi = this.gv(this.view.ufield3d_zi, this.idx);
            var zj = this.gv(this.view.ufield3d_zj, this.idx);
            var zk = this.gv(this.view.ufield3d_zk, this.idx);
            var values = this.gv(this.view.field_values, this.fields['field']);
            this.cube_field = new CubeField(ox, oy, oz, nx, ny, nz, xi, xj, xk, yi, yj, yk, zi, zj, zk, values);
            this.app3d.remove_meshes(this.cube_field_mesh);
            this.cube_field_mesh = this.app3d.add_scalar_field(this.cube_field, this.fields.isovalue, 2);
        };

        render_current_frame() {
            /*"""
            render_current_frame
            -----------------------
            Renders atoms and bonds in the current frame (using the frame index).
            */
            var symbols = this.gv(this.view.atom_symbols, this.idx);
            var radii = utility.mapper(symbols, this.view.atom_radii_dict);
            var colors = utility.mapper(symbols, this.view.atom_colors_dict);
            var x = this.gv(this.view.atom_x, this.idx);
            var y = this.gv(this.view.atom_y, this.idx);
            var z = this.gv(this.view.atom_z, this.idx);
            var v0 = this.gv(this.view.two_bond0, this.idx);
            var v1 = this.gv(this.view.two_bond1, this.idx);
            this.app3d.scene.remove(this.atom_meshes);
            this.atom_meshes = this.app3d.add_points(x, y, z, colors, radii);
            if (v0 !== undefined && v1 !== undefined) {
                this.app3d.scene.remove(this.bond_meshes);
                this.bond_meshes = this.app3d.add_lines(v0, v1, x, y, z, colors);
            };
            if (this.idx === 0) {
                this.app3d.set_camera_from_mesh(this.atom_meshes[0], 4.0, 4.0, 4.0);
            };
        };

        render_cell() {
            /*"""
            render_cell
            -----------
            Custom rendering function that adds the unit cell.
            */
            var ox = this.gv(this.view.frame_ox, this.idx);
            if (ox === undefined) {
                return;
            };
            var oy = this.gv(this.view.frame_oy, this.idx);
            var oz = this.gv(this.view.frame_oz, this.idx);
            var xi = this.gv(this.view.frame_xi, this.idx);
            var xj = this.gv(this.view.frame_xj, this.idx);
            var xk = this.gv(this.view.frame_xk, this.idx);
            var yi = this.gv(this.view.frame_yi, this.idx);
            var yj = this.gv(this.view.frame_yj, this.idx);
            var yk = this.gv(this.view.frame_yk, this.idx);
            var zi = this.gv(this.view.frame_zi, this.idx);
            var zj = this.gv(this.view.frame_zj, this.idx);
            var zk = this.gv(this.view.frame_zk, this.idx);
            var vertices = [];
            vertices.push([ox, oy, oz]);
            vertices.push([xi, xj, xk]);
            vertices.push([yi, yj, yk]);
            vertices.push([zi, zj, zk]);
            this.app3d.scene.remove(this.cell_meshes);
            this.cell_meshes = this.app3d.add_wireframe(vertices);
        };
    };

/*    AtomicApp.prototype.init_gui = function() {
        var self = this;
        this.f1f = this.gui.addFolder('animation');
        this.f2f = this.gui.addFolder('atoms');
        this.f3f = this.gui.addFolder('bonds');
        this.f4f = this.gui.addFolder('fields');
        this.f5f = this.gui.addFolder('cells');

        this.playing = false;
        this.f1 = {
            pause: function() {
                this.playing = false;
                clearInterval(this._play_callback);
            },
            play: function() {
                if (this.playing == true) {
                    this.pause();
                } else {
                    this.playing = true;
                    if (self.index == self.last_frame_index) {
                        self.index = 0;
                        self.frame = self.framelist[self.index];
                        self.f1o.slider.setValue(self.index);
                        self.f1p.framelist.setValue(self.frame);
                    };
                    this._play_callback = setInterval(function() {
                        if (self.index < self.last_frame_index) {
                            self.index += 1;
                            self.frame = self.framelist[self.index];
                            self.f1o.slider.setValue(self.index);
                            self.f1p.framelist.setValue(self.frame);
                        } else {
                            self.f1.pause();
                        };
                    }, 1000 / self.fps);
                };
            },
            index: this.index,
            frame: this.frame,
            fps: this.fps,
            track: false,
        };

        this.f1o = {
            'play': this.f1f.add(this.f1, 'play'),
            'slider': this.f1f.add(this.f1, 'index', 0, this.last_frame_index),
            'framelist': this.f1f.add(this.f1, 'frame', this.view.framelist),
            'fps': this.f1f.add(this.f1, 'fps', 1, 60, 1),
            'track': this.f1f.add(this.f1, 'track'),
        };

        this.f1o.framelist.onChange(function(frame) {
            self.frame = frame;
            self.index = self.framelist.indexOf(self.frame);
            console.log('framelist onchange');
        });

        this.f1o.slider.onChange(function(index) {
            self.index = index;
            console.log('slider change');
        });

        this.f1o.slider.onFinishChange(function(index) {
            self.index = index;
            self.frame = self.framelist[self.index];
            console.log('slider finish change');
        });
    };


    AtomicApp.prototype.render_atoms =  function(index) {
        this.index = index;
        var symbols = this.get_value(this.view.atom_symbols, this.index);
        var radii = utility.mapper(symbols, this.view.atom_radii_dict);
        var colors = utility.mapper(symbols, this.view.atom_colors_dict);
        var x = this.get_value(this.view.atom_x, this.index);
        var y = this.get_value(this.view.atom_y, this.index);
        var z = this.get_value(this.view.atom_z, this.index);
        var v0 = this.get_value(this.view.two_bond0, this.index);
        var v1 = this.get_value(this.view.two_bond1, this.index);
        this.app3d.scene.remove(this.atoms);
        this.atoms = this.app3d.add_points(x, y, z, colors, radii);
        this.app3d.scene.remove(this.bonds);
        this.bonds = this.app3d.add_lines(v0, v1, x, y, z, colors);
        this.app3d.set_camera_from_geometry(x, y, z, this.atoms.geometry, 4.0, 4.0, 4.0);
    };


    AtomicApp.prototype.get_value = function(obj, index) {
        var value = obj[index];
        if (value == undefined) {
            return obj;
        } else {
            return value;
        };
    };*/

    return UniverseApp;
});
