/*"""
IPython Widget View of the Universe Container
````````````````````````````````````````````````
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/atomic/threejs': {
            exports: 'AtomicThreeJS'
        },
        'nbextensions/exa/atomic/gui': {
            exports: 'GUI'
        },
        'nbextensions/exa/atomic/lib/dat.gui.min': {
            exports: 'dat'
        },
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/atomic/threejs',
    'nbextensions/exa/atomic/gui',
    'nbextensions/exa/atomic/lib/dat.gui.min'
], function(widget, AtomicThreeJS, GUI, dat){
    var UniverseView = widget.DOMWidgetView.extend({
        /*"""
        UniverseView
        ``````````````````
        The UniverseView object is responsible for creating the necessary HTML
        elements required for displaying the 3D representation of the Universe
        container. It is also responsible for handling messages passed between
        the JavaScript frontend and Python backend. Drawing and rendering
        atoms, bonds, volumes, etc is handled by the atomic threejs application,
        AtomicThreeJS (in threejs.js).
        */
        render: function() {
            /*"""
            Render
            ````````````
            This is the main entry point for UniverseView. For more information
            see the documentation of Backbone.js. Here we create the widget
            three.js viewport and gui controls.
            */
            var self = this;    // Alias the instance of the view for future ref.

            // Initialize the widget pulling data from the Universe
            this.init_self();

            // UniverseView gui
            this.gui = new dat.GUI({autoPlace: false, width: this.gui_width});
            this.init_gui();
            this.gui_style = document.createElement('style');
            this.gui_style.innerHTML = GUI.style_str;

            // Threejs app
            this.app = new AtomicThreeJS(this.canvas);
            this.update_atom(true);
            this.update_bond(true);
            this.app.update_camera_and_controls();

            // Add the gui and app to the container
            this.container.append(this.canvas);
            this.container.append(this.gui.domElement);
            this.container.append(this.gui_style);
            this.setElement(this.container);

            // Animate
            this.app.render();
            this.on('displayed', function () {
                self.app.animate();
                self.app.controls.handleResize();
            });
        },

        init_self: function() {
            /*"""
            Initialize Widget
            ```````````````````````
            Get data from backend and set it locally
            */
            var self = this;
            this.atom_type = this.model.get('_atom_type');
            this.width = this.model.get('width');
            this.height = this.model.get('height');
            this.gui_width = this.model.get('_gui_width');
            this.framelist = this.model.get('_framelist');
            this.nframes = this.framelist.length;
            this.index = 0;
            this.playing = false;
            this.frame = this.framelist[this.index];
            this.fps = this.model.get('_fps');
            this.x = this.get_from_json_str('_atom_x');
            this.y = this.get_from_json_str('_atom_y');
            this.z = this.get_from_json_str('_atom_z');
            this.r = this.get_from_json_str('_atom_radius');
            this.c = this.get_from_json_str('_atom_color');
            this.bonds = this.get_from_json_str('_bonds');
            this.bonds_length = Object.keys(this.bonds).length;
            this.cell_xi = this.get_from_json_str('_frame_xi');
            this.cell_xj = this.get_from_json_str('_frame_xj');
            this.cell_xk = this.get_from_json_str('_frame_xk');
            this.cell_yi = this.get_from_json_str('_frame_yi');
            this.cell_yj = this.get_from_json_str('_frame_yj');
            this.cell_yk = this.get_from_json_str('_frame_yk');
            this.cell_zi = this.get_from_json_str('_frame_zi');
            this.cell_zj = this.get_from_json_str('_frame_zj');
            this.cell_zk = this.get_from_json_str('_frame_zk');
            this.cell_ox = this.get_from_json_str('_frame_ox');
            this.cell_oy = this.get_from_json_str('_frame_oy');
            this.cell_oz = this.get_from_json_str('_frame_oz');
            this.centers = this.get_from_json_str('_center');
            this.field = 0;
            this.fields = this.model.get('_fields');
            this.fieldframes = this.model.get('_fieldframes');
            this.nfields = this.fields.length;
            this.field_ox = this.get_from_json_str('_field_ox');
            this.field_oy = this.get_from_json_str('_field_oy');
            this.field_oz = this.get_from_json_str('_field_oz');
            this.field_nx = this.get_from_json_str('_field_nx');
            this.field_ny = this.get_from_json_str('_field_ny');
            this.field_nz = this.get_from_json_str('_field_nz');
            this.field_dxi = this.get_from_json_str('_field_xi');
            this.field_dxj = this.get_from_json_str('_field_xj');
            this.field_dxk = this.get_from_json_str('_field_xk');
            this.field_dyi = this.get_from_json_str('_field_yi');
            this.field_dyj = this.get_from_json_str('_field_yj');
            this.field_dyk = this.get_from_json_str('_field_yk');
            this.field_dzi = this.get_from_json_str('_field_zi');
            this.field_dzj = this.get_from_json_str('_field_zj');
            this.field_dzk = this.get_from_json_str('_field_zk');
            console.log(this.x);
            console.log(this.y);
            console.log(this.z);
            console.log(this.field);
            console.log(this.fields);
            console.log(this.fieldframes);
            console.log(this.field_ox);
            console.log(this.field_oy);
            console.log(this.field_oz);
            console.log(this.field_nx);
            console.log(this.field_ny);
            console.log(this.field_nz);
            console.log(this.field_xi);
            console.log(this.field_yj);
            console.log(this.field_zk);
            this.filled = true;
            // Resizable container to house the threejs app canvas and gui
            this.container = $('<div/>').width(this.width).height(this.height).resizable({
                aspectRatio: false,
                resize: function(event, ui) {
                    self.width = ui.size.width;
                    self.height = ui.size.height;
                    self.model.set('width', self.width);
                    self.model.set('height', self.height);
                    self.canvas.width(self.width - 300);
                    self.canvas.height(self.height);
                    self.app.resize();
                },
                stop: function(event, ui) {
                    self.app.render();
                }
            });
            this.canvas = $('<canvas/>').width(this.width - this.gui_width).height(this.height);
            this.canvas.css('position', 'absolute');
            this.canvas.css('top', 0);
            this.canvas.css('left', this.gui_width);
        },

        get_from_json_str: function(name) {
            /*"""
            JSON to Object
            ```````````````
            Custom getter for Python objects stored as json strings.
            */
            try {
                return JSON.parse(this.model.get(name));
            } catch(err) {
//                console.log(err);
                return {};
            };
        },

        init_gui: function() {
            /*"""
            Initialize the GUI
            ``````````````````````
            Called after created the GUI element.
            */
            var self = this;
            this.guif1 = this.gui.addFolder('animation');
            this.guif2 = this.gui.addFolder('atoms');
            this.guif3 = this.gui.addFolder('bonds');
            this.guif4 = this.gui.addFolder('cell');
            this.guif5 = this.gui.addFolder('surfaces');
            this.guif6 = this.gui.addFolder('volumes');

            // Folder 1: animation
            console.log(this.index);
            console.log(this.frame);
            this.gui_f1 = {
                pause: function() {
                    this.playing = false;
                    clearInterval(this._playing);
                },
                play: function() {
                    //var _self = this;
                    if (this.playing == true) {
                        this.pause();
                    } else {
                        this.playing = true;
                        if (self.index == self.nframes - 1) {
                            self.index = 0;
                            self.f1.index.setValue(self.index);
                        };
                        this._playing = setInterval(function() {
                            if (self.index < self.nframes - 1) {
                                self.index += 1
                                self.f1.index.setValue(self.index);
                            } else {
                                self.gui_f1.pause();
                            };
                        }, 1000 / self.fps);
                    };
                },
                index: this.index,
                frame: this.frame,
                fps: this.fps
            };
            console.log(this.gui_f1['frame']);
            this.f1 = {};
            this.f1['play'] = this.guif1.add(this.gui_f1, 'play');
            this.f1['frame'] = this.guif1.add(this.gui_f1, 'frame', this.framelist);
            this.f1['index'] = this.guif1.add(this.gui_f1, 'index', 0, this.nframes - 1);
            this.f1['fps'] = this.guif1.add(this.gui_f1, 'fps', 1, 60, 1);
            this.f1['frame'].onChange(function(frame) {
                self.frame = frame;
                self.update_atom(true);
                if (self.bonds_length > 0) {
                    self.update_bond(true);
                };
            });
            this.f1['index'].onChange(function(value) {
                self.index = value;
                self.frame = self.framelist[self.index];
                self.update_atom(true);
                if (self.bonds_length > 0) {
                    self.update_bond(true);
                };
            });
            this.f1['fps'].onFinishChange(function(value) {
                self.fps = value;
            });

            this.gui_f2 = {
                show: true,
            };
            this.f2 = {};
            this.f2['show'] = this.guif2.add(this.gui_f2, 'show');
            this.f2['show'].onFinishChange(function(value) {
                self.update_atom(value);
            });

            this.gui_f3 = {
                show: true,
            }
            this.f3 = {};
            this.f3['show'] = this.guif3.add(this.gui_f3, 'show');
            this.f3['show'].onFinishChange(function(value) {
                self.update_bond(value);
            });

            this.gui_f4 = {
                show: false,
            };
            this.f4 = {};
            this.f4['show'] = this.guif4.add(this.gui_f4, 'show');
            this.f4['show'].onFinishChange(function(value) {
                self.update_cell(value);
            });
            this.gui_f5 = {
                show: false,
                frame: this.frame,
                field: this.field,
                opacity: 1,
            };
            this.f5 = {};
            this.f5['show'] = this.guif5.add(this.gui_f5, 'show');
            //this.f5['frame'] = this.guif5.add(this.gui_f5, 'frame', this.framelist);
            this.f5['field'] = this.guif5.add(this.gui_f5, 'field', Object.keys(this.fields));
            //this.f5['opacity'] = this.guif5.add(this.gui_f5, 'opacity', 0, 1);
            this.f5['show'].onFinishChange(function(value) {
                self.update_surface(value);
            });
            //this.f5['frame'].onFinishChange(function(value) {
            //    console.log('inside gui frame change');
            //    console.log(value);
            //    self.frame = value;
            //    self.update_surface(true);
            //});
            this.f5['field'].onFinishChange(function(value) {
                console.log('inside gui field change');
                console.log(value);
                self.field = value;
                self.update_surface(true);
            });

            //this.f5['opacity'].onFinishChange(function(value) {
            //    this.gui_f5.opacity = value;
            //    self.update_surface(true);
            //});
        },

        update_surface: function(value) {
            console.log('Hit update surface');
            if (value == true) {
                console.log('if is true');
                var fd = this.field;
                var fr = this.fieldframes[fd];
                this.f1.frame.setValue(fr);
                console.log(fd);
                console.log(fr);
                console.log(this.field_nx)
                var field = this.fields[fd];
                var dims = [this.field_nx[fd],
                            this.field_ny[fd],
                            this.field_nz[fd]];
                var orig = [this.field_ox[fd],
                            this.field_oy[fd],
                            this.field_oz[fd]];
                var scale = [this.field_dxi[fd],
                             this.field_dyj[fd],
                             this.field_dzk[fd]];
                var iso = 0.03;
                this.app.add_surface(field, dims, orig, scale, iso);
            } else {
                console.log('else is true');
                this.app.scene.remove(this.app.surf);
                this.app.scene.remove(this.app.nsurf);
            }
            this.app.render();
        },

        update_atom: function(value) {
            /*"""
            Update Atomic Positions
            `````````````````````````
            */
            if (value == true) {
                var x = this.x[this.frame];
                var y = this.y[this.frame];
                var z = this.z[this.frame];
                var r = this.r[this.frame];
                var c = this.c[this.frame];
                //var center = this.centers[this.frame];
                //console.log(center);
                if (this.atom_type == 'points') {
                    this.app.add_points(x, y, z, r, c, this.filled);
                };
                //this.app.update_cam_ctrl(center);
            } else {
                this.app.scene.remove(this.app.atom);
            }
            this.app.render();
        },

        update_bond: function(value) {
            /*"""
            Bonds
            ```````````````
            */
            if (value == true && this.bonds_length > 0) {
                var x = this.x[this.frame];
                var y = this.y[this.frame];
                var z = this.z[this.frame];
                var bonds = this.bonds[this.frame];
                this.app.add_bonds(bonds, x, y, z);
            } else {
                this.app.scene.remove(this.app.bond);
            };
            this.app.render();
        },

        update_cell: function(value) {
            /*"""
            Update the Unit Cell
            ```````````````````````
            */
            if (value == true) {
                var xi = this.cell_xi[this.index];
                var xj = this.cell_xj[this.index];
                var xk = this.cell_xk[this.index];
                var yi = this.cell_yi[this.index];
                var yj = this.cell_yj[this.index];
                var yk = this.cell_yk[this.index];
                var zi = this.cell_zi[this.index];
                var zj = this.cell_zj[this.index];
                var zk = this.cell_zk[this.index];
                var ox = this.cell_ox[this.index];
                var oy = this.cell_oy[this.index];
                var oz = this.cell_oz[this.index];
                this.app.add_cell(xi, xj, xk, yi, yj, yk, zi, zj, zk, ox, oy, oz);
            } else {
                this.app.scene.remove(this.app.cell);
            };
            this.app.render();
        },

        callme: function() {
            console.log('called callme');
            var x = this.model.get('_atom_x');
            console.log(x[0]);
            var y = this.model.get('_atom_y');
            console.log(y[0]);
            this.send({'call': 'update'});
        },


    });

    return {'UniverseView': UniverseView};
});
