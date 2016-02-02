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
                console.log(err);
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
            this.f1 = {};
            this.f1['play'] = this.guif1.add(this.gui_f1, 'play');
            this.f1['frame'] = this.guif1.add(this.gui_f1, 'frame', this.framelist);
            this.f1['index'] = this.guif1.add(this.gui_f1, 'index', 0, this.nframes - 1).step(1);
            this.f1['fps'] = this.guif1.add(this.gui_f1, 'fps', 1, 60, 1);
            this.f1['index'].onChange(function(value) {
                self.index = value;
                self.frame = self.framelist[self.index];
                console.log(self.f2['show']);
                console.log(self.gui_f2['show']);
                console.log(self.f3['show']);
                console.log(self.gui_f3['show']);
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
                if (this.atom_type == 'points') {
                    this.app.add_points(x, y, z, r, c, this.filled);
                };
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
            console.log(this.cell_xi);
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
