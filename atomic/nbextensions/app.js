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
    },
});


define([
    'nbextensions/exa/apps/gui',
    'nbextensions/exa/apps/app3d',
], function(ContainerGUI, App3D) {
    class UniverseApp {
        /*"""
        UniverseApp
        =============
        Notebook widget application for visualization of the universe container.
        */
        constructor(view) {
            this.view = view;
            this.view.create_canvas();
            this.frame_index = 0;
            this.num_frames = this.view.framelist.length;
            this.last_index = this.num_frames - 1;
            this.current_frame = this.view.framelist[this.frame_index];
            this.app3d = new app3D.ThreeJSApp(this.view.canvas);
            this.create_gui();
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

        create_gui() {
            /*"""
            create_gui
            ------------------
            Create the application's control set.
            */
            var self = this;
            this.gui = new ContainerGUI(this.view.gui_width);
            this.level0 = {
                'play': function() {
                    console.log('clicked play');
                },
                'frame': this.current_frame
            };
            this.display = this.gui.addFolder('display');
            this.fields = this.gui.addFolder('fields');
        };
    };

    AtomicApp.prototype.init_gui = function() {
        /*"""
        init_gui
        ----------------
        Initialize the graphical user interface.
        */
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

    AtomicApp.prototype.resize = function() {
        this.app3d.resize();
    };

    AtomicApp.prototype.render_atoms =  function(index) {
        /*"""
        render_frame
        --------------
        */
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
        /*"""
        get_value
        --------------
        */
        var value = obj[index];
        if (value == undefined) {
            return obj;
        } else {
            return value;
        };
    };

    return UniverseApp;
});
