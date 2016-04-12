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
    },
});


define([
    'nbextensions/exa/apps/gui',
    'nbextensions/exa/apps/app3d',
    'nbextensions/exa/utility'
], function(ContainerGUI, App3D, utility) {
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
            var value = obj[index];
            if (value === undefined) {
                return obj;
            } else {
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
            this.idx = 0;
            this.last_index = this.num_frames - 1;
            if (typeof this.view.framelist === 'number') {
                this.framelist = [this.view.framelist];
            } else {
                this.framelist = this.view.framelist;
            };
            this.num_frames = this.framelist.length;
            this.current_frame = this.framelist[this.idx];
            this.fps = this.view.fps;
        };

        create_gui() {
            /*"""
            create_gui
            ------------------
            Create the application's control set.
            */
            var self = this;
            this.gui = new ContainerGUI(this.view.gui_width);

            this.top = {
                'play': function() {
                    console.log('clicked play');
                },
                'playbar': this.idx,
                'frame': this.current_frame,
                'fps': this.fps,
            };

            this.top['play_button'] = this.gui.add(this.top, 'play');
            this.top['playbar_slider'] = this.gui.add(this.top, 'playbar', 0, this.last_index, 1);
            this.top['frame_dropdown'] = this.gui.add(this.top, 'frame', this.framelist);
            this.top['fps'] = this.gui.add(this.top, 'fps', 0, 60);

            this.top.playbar_slider.onChange(function(index) {
                console.log(index);
            });

            this.fields = {
                'isovalue': 0.03,
                'field': 0,
            };
            this.fields['folder'] = this.gui.addFolder('fields');

            this.display = this.gui.addFolder('display');
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
            console.log(x);
            console.log(y);
            console.log(z);
            this.app3d.scene.remove(this.atoms);
            this.atoms = this.app3d.add_points(x, y, z, colors, radii);
            this.app3d.scene.remove(this.bonds);
            if (v0 !== undefined && v1 !== undefined) {
                this.bonds = this.app3d.add_lines(v0, v1, x, y, z, colors);
            };
            this.app3d.set_camera_from_mesh(this.atoms, 4.0, 4.0, 4.0);
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
