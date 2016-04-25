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
            exports: 'AtomicField'
        }
    },
});


define([
    'nbextensions/exa/apps/gui',
    'nbextensions/exa/apps/app3d',
    'nbextensions/exa/utility',
    'nbextensions/exa/atomic/field'
], function(ContainerGUI, App3D, utility, AtomicField) {
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
            this.app3d.add_unit_axis();
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
            this.last_index = this.view.framelist.length - 1;
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
                    self.top.frame_dropdown.setValue(self.view.framelist[self.top.index]);
                },
                'play': function() {
                    if (self.playing === true) {
                        self.top.pause()
                    } else {
                        self.playing = true;
                        if (self.top.index === self.last_index) {
                            self.top.index_slider.setValue(0);
                        };
                        self.play_id = setInterval(function() {
                            if (self.top.index < self.last_index) {
                                self.top.index_slider.setValue(self.top.index+1);
                            } else {
                                self.top.pause();
                            };
                        }, 1000 / self.top.fps);
                    };
                },
                'index': 0,
                'frame': this.view.framelist[0],
                'fps': this.view.fps,
            };
            this.top['play_button'] = this.gui.add(this.top, 'play');
            this.top['index_slider'] = this.gui.add(this.top, 'index').min(0).max(this.last_index).step(1);
            this.top['frame_dropdown'] = this.gui.add(this.top, 'frame', this.view.framelist);
            this.top['fps_slider'] = this.gui.add(this.top, 'fps').min(1).max(240).step(1);
            this.top.index_slider.onChange(function(index) {
                self.top.index = index;
                self.top.frame = self.view.framelist[self.top.index];
                self.update_fields();
                self.render_current_frame();
            });
            this.top.index_slider.onFinishChange(function(index) {
                self.top.index = index;
                self.top.frame = self.view.framelist[self.top.index];
                self.top.frame_dropdown.setValue(self.top.frame);
            })
            this.top.fps_slider.onFinishChange(function(value) {
                self.top.fps = value;
            });
            this.top.frame_dropdown.onFinishChange(function(value) {
                self.top.frame = value;
                self.top.index = self.view.framelist.indexOf(self.top.frame);
                self.top.index_slider.setValue(self.top.index);
            });

            this.display = {
                'cell': false,
            };
            this.display['folder'] = this.gui.addFolder('display');
            this.display['cell_checkbox'] = this.display.folder.add(this.display, 'cell');

            this.display.cell_checkbox.onFinishChange(function(value) {
                self.display.cell = value;
                if (value === false) {
                    self.app3d.remove_meshes(self.cell_meshes);
                } else {
                    self.render_cell();
                };
            });

            this.fields = {
                'isovalue': 0.03,
                'field': '',
                'cur_fields': [null]
            };

            this.fields['folder'] = this.gui.addFolder('fields');
            this.fields['isovalue_slider'] = this.fields.folder.add(this.fields, 'isovalue', 0.0001, 0.5);
            this.fields['field_dropdown'] = this.fields.folder.add(this.fields, 'field', this.fields['cur_fields']);
            this.fields.field_dropdown.onFinishChange(function(field_index) {
                self.fields['field'] = field_index;
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
            var field_indices = this.gv(this.view.field_indices, this.top.index);
            if (field_indices === undefined) {
                field_indices = [];
            };
            field_indices.push(null);
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
            var nx = this.gv(this.view.atomicfield_nx, this.top.index);
            var ny = this.gv(this.view.atomicfield_ny, this.top.index);
            var nz = this.gv(this.view.atomicfield_nz, this.top.index);
            var ox = this.gv(this.view.atomicfield_ox, this.top.index);
            var oy = this.gv(this.view.atomicfield_oy, this.top.index);
            var oz = this.gv(this.view.atomicfield_oz, this.top.index);
            var dxi = this.gv(this.view.atomicfield_dxi, this.top.index);
            var dxj = this.gv(this.view.atomicfield_dxj, this.top.index);
            var dxk = this.gv(this.view.atomicfield_dxk, this.top.index);
            var dyi = this.gv(this.view.atomicfield_dyi, this.top.index);
            var dyj = this.gv(this.view.atomicfield_dyj, this.top.index);
            var dyk = this.gv(this.view.atomicfield_dyk, this.top.index);
            var dzi = this.gv(this.view.atomicfield_dzi, this.top.index);
            var dzj = this.gv(this.view.atomicfield_dzj, this.top.index);
            var dzk = this.gv(this.view.atomicfield_dzk, this.top.index);
            var values = this.gv(this.view.field_values, this.fields['field']);
            this.cube_field = new AtomicField(ox, oy, oz, nx, ny, nz, dxi, dxj, dxk,
                                            dyi, dyj, dyk, dzi, dzj, dzk, values);
            this.app3d.remove_meshes(this.cube_field_mesh);
            this.cube_field_mesh = this.app3d.add_scalar_field(this.cube_field, this.fields.isovalue, 2);
        };

        render_current_frame() {
            /*"""
            render_current_frame
            -----------------------
            Renders atoms and bonds in the current frame (using the frame index).
            */
            var symbols = this.gv(this.view.atom_symbols, this.top.index);
            var radii = utility.mapper(symbols, this.view.atom_radii_dict);
            var colors = utility.mapper(symbols, this.view.atom_colors_dict);
            var x = this.gv(this.view.atom_x, this.top.index);
            var y = this.gv(this.view.atom_y, this.top.index);
            var z = this.gv(this.view.atom_z, this.top.index);
            var v0 = this.gv(this.view.two_bond0, this.top.index);
            var v1 = this.gv(this.view.two_bond1, this.top.index);
            this.app3d.remove_meshes(this.atom_meshes);
            this.atom_meshes = this.app3d.add_points(x, y, z, colors, radii);
            if (v0 !== undefined && v1 !== undefined) {
                this.app3d.remove_meshes(this.bond_meshes);
                this.bond_meshes = this.app3d.add_lines(v0, v1, x, y, z, colors);
            };
            if (this.top.index === 0) {
                this.app3d.set_camera_from_mesh(this.atom_meshes[0], 4.0, 4.0, 4.0);
            };
        };

        render_cell() {
            /*"""
            render_cell
            -----------
            Custom rendering function that adds the unit cell.
            */
            var ox = this.gv(this.view.frame_ox, this.top.index);
            if (ox === undefined) {
                return;
            };
            var oy = this.gv(this.view.frame_oy, this.top.index);
            var oz = this.gv(this.view.frame_oz, this.top.index);
            var xi = this.gv(this.view.frame_xi, this.top.index);
            var xj = this.gv(this.view.frame_xj, this.top.index);
            var xk = this.gv(this.view.frame_xk, this.top.index);
            var yi = this.gv(this.view.frame_yi, this.top.index);
            var yj = this.gv(this.view.frame_yj, this.top.index);
            var yk = this.gv(this.view.frame_yk, this.top.index);
            var zi = this.gv(this.view.frame_zi, this.top.index);
            var zj = this.gv(this.view.frame_zj, this.top.index);
            var zk = this.gv(this.view.frame_zk, this.top.index);
            var vertices = [];
            vertices.push([ox, oy, oz]);
            vertices.push([xi, xj, xk]);
            vertices.push([yi, yj, yk]);
            vertices.push([zi, zj, zk]);
            this.app3d.remove_meshes(this.cell_meshes);
            this.cell_meshes = this.app3d.add_wireframe(vertices);
        };
    };

    return UniverseApp;
});
