/*"""
IPython Widget for Visualization of the atomic Universe
`````````````````````````````````````````````````````````
This extension is allows for visualization of the atomc Universe within a
Jupyter notebook environment. Visualization utilizes the three.js library.
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'container'
        },

        'nbextensions/exa/atomic/threejs': {
            exports: 'three'
        }
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/container',
    'nbextensions/exa/atomic/threejs'
], function(widget, container, three){
    var UniverseView = container.ContainerView.extend({
        /*"""
        UniverseView
        ``````````````````
        The UniverseView object is responsible for creating the necessary HTML
        elements required for displaying the 3D representation of the universe
        container.
        */
        init: function() {
            /*"""
            init (universe)
            ``````````````````
            */
            console.log('universe init');
            this.init_vars();
            this.init_atom();
            this.init_gui();

            this.on('displayed', function() {
                console.log('displayed');
            });
        },

        init_gui: function() {
            /*"""
            init_gui
            ----------
            */
            var self = this;
            this.container = $('<div/>').width(this.width).height(this.height).resizable({
                aspectRatio: false,
                resize: function(event, ui) {
                    self.width = ui.size.width;
                    self.height = ui.size.height;
                    self.model.set('width', self.width);
                    self.model.set('height', self.height);
                    self.canvas.width(self.width - self.gui_width);
                    self.canvas.height(self.height);
                    //self.app.resize();
                },
                stop: function(event, ui) {
                    //self.app.render();
                }
            });
            this.canvas = $('<canvas/>').width(this.width - this.gui_width).height(this.height);
            this.canvas.css('position', 'absolute');
            this.canvas.css('top', 0);
            this.canvas.css('left', this.gui_width);

            this.viewer = new three.AtomicThreeJS(this.canvas);
            console.log(this.viewer);

            this.container.append(this.canvas);
            this.setElement(this.container);
        },

        init_vars: function() {
            /*"""
            init_vars
            ------------
            */
            this.meta_index = 0;    // Positional ref. to frame index values
        },

        init_atom: function() {
            /*"""
            init_atom
            ---------------
            Pulls in data from the atom dataframe
            */
            this.update_atom_x();
            this.update_atom_y();
            this.update_atom_z();
            this.model.on('change:atom_x', this.update_atom_x, this);
            this.model.on('change:atom_y', this.update_atom_y, this);
            this.model.on('change:atom_z', this.update_atom_z, this);
        },

        update_atom_x: function() {
            this.atom_x = this.get_trait('atom_x');
            console.log(this.atom_x);
        },

        update_atom_y: function() {
            this.atom_y = this.get_trait('atom_y');
            console.log(this.atom_y);
        },

        update_atom_z: function() {
            this.atom_z = this.get_trait('atom_z');
            console.log(this.atom_z);
        }
    });


    return {'UniverseView': UniverseView};
});
