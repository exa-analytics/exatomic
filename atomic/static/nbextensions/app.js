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
        'nbextensions/exa/lib/dat.gui.min': {
            exports: 'dat'
        },

        'nbextensions/exa/three.app': {
            exports: 'app3D'
        },

        'nbextensions/exa/utility': {
            exports: 'utility'
        },
    },
});


define([
    'nbextensions/exa/lib/dat.gui.min',
    'nbextensions/exa/three.app',
    'nbextensions/exa/utility',
], function(dat, app3D, utility) {
    var AtomicApp = function(view) {
        /*"""
        AtomicApp
        ============
        */
        var self = this;
        this.view = view;
        this.canvas = this.view.canvas;
        this.index = 0;
        this.length = this.view.framelist.length;
        this.last_frame_index = this.length - 1;
        this.app3d = new app3D.ThreeJSApp(this.canvas);
        this.gui = new dat.GUI({autoPlace: false, width: this.view.gui_width});
        this.gui_style = document.createElement('style');
        this.gui_style.innerHTML = gui_style;
        this.init_gui();
        this.render_atoms(this.index);
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
        this.f4f = this.gui.addFolder('cell');
        this.f5f = this.gui.addFolder('surfaces');
        this.f6f = this.gui.addFolder('volumes');

        this.playing = false;
        this.f1 = {
            'pause': function() {
                this.playing = false;
                clearInterval(this._play_callback);
            },
            'play': function() {
                if (this.playing == true) {
                    this.pause();
                } else {
                    this.playing = true;
                    if (self.index == self.last_frame_index) {
                        self.index = 0;
                        self.f1.index.setValue(self.index);
                    };
                    this._play_callback = setInterval(function() {
                        if (self.index < self.last_frame_index) {
                            self.index += 1;
                            self.f1.index.setValue(self.index);
                        } else {
                            self.f1.pause();
                        };
                    }, 1000 / self.fps);
                }
            }
        };
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
        this.set_camera(x, y, z);
    };

    AtomicApp.prototype.set_camera =  function(x, y, z) {
        /*"""
        set_camera
        -------------
        */
        var n = x.length;
        var i = n;
        var oxyz = [0.0, 0.0, 0.0];
        while (i--) {
            oxyz[0] += x[i];
            oxyz[1] += y[i];
            oxyz[2] += z[i];
        };
        oxyz[0] /= n;
        oxyz[1] /= n;
        oxyz[2] /= n;
        this.atoms.geometry.computeBoundingBox();
        var bbox = this.atoms.geometry.boundingBox;
        var x = Math.pow(bbox.max.x - bbox.min.x, 2);
        var y = Math.pow(bbox.max.y - bbox.min.y, 2);
        var z = Math.pow(bbox.max.z - bbox.min.z, 2);
        var d = Math.sqrt(x + y + z);
        console.log(d);
        var xyz = bbox.max;
        xyz.x += d;
        xyz.y += d;
        xyz.z += d;
        console.log(xyz);
        this.app3d.set_camera(100, 100, 100, oxyz[0], oxyz[1], oxyz[2]);
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

    var gui_style = ".dg {\
        color: black;\
        font: 400 13px Verdana, Arial, sans-serif;\
        text-shadow: white 0 0 0;\
    }\
    .hue-field {\
        width: 10;\
    }\
    .dg .c .slider {\
        background: white\
    }\
    .dg .c .slider:hover {\
        background: white\
    }\
    .dg .c input[type=text] {\
        background: white;\
        border-color: lightgrey;\
        border-radius: 2px;\
        border-style: solid;\
        border-width: 1.1px;\
        color: black\
    }\
    .dg .c input[type=text]:active {\
        background: white;\
        color: black;\
        outline-color: lightgrey;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .c input[type=text]:focus {\
        background: white;\
        color: black;\
        outline-color: lightgrey;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .c input[type=text]:hover {\
        background: white;\
        color: black;\
        outline-color: lightgrey;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .closed li.title {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: #777;\
        text-shadow: 1px 0 0 white\
    }\
    .dg .cr.boolean:hover {\
        background: white;\
        border-bottom: 1px solid white;\
        border-right: 1px solid white\
    }\
    .dg .cr.function:hover {\
        background: white;\
        border-bottom: 1px solid white;\
        border-right: 1px solid white\
    }\
    .dg li.cr {\
        background: #fafafa;\
        border-bottom: 1px solid white;\
        border-right: 1px solid white\
    }\
    .dg li.cr:hover {\
        background: white;\
        border-bottom: 1px solid white;\
        border-right: 1px solid white\
    }\
    .dg li.title, .dg closed {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: black;\
        text-shadow: 1px 0 0 white\
    }\
    .dg li.title:hover {\
        outline-color: lightgrey;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg.main .close-button {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: black;\
        height: 27px;\
        line-height: 27px;\
        text-align: center;\
        text-shadow: 1px 0 0 white\
    }\
    .dg.main .close-button:hover {\
        outline-color: lightgrey;\
        outline-style: solid;\
        outline-width: 1.5px\
    }";

    return {'AtomicApp': AtomicApp, 'gui_style': gui_style};
});
