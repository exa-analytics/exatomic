/*"""
=========================================================
Universe View
=========================================================
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'container'
        },

        'nbextensions/exa/atomic/app': {
            exports: 'app'
        },

        'nbextensions/exa/utility': {
            exports: 'utility'
        },
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/container',
    'nbextensions/exa/atomic/app',
    'nbextensions/exa/utility',
], function(widget, container, app, utility){
    var UniverseView = container.ContainerView.extend({
        /*"""
        UniverseView
        ====================
        Frontend representation of an atomic universe instance.
        */
        render: function() {
            /*"""
            render
            --------------
            Main entry point for the universe container frontend.
            */
            console.log('Initializing universe...');
            var self = this;
            this.model.on('change:atom_x', this.update_x, this);
            this.model.on('change:atom_y', this.update_y, this);
            this.model.on('change:atom_z', this.update_z, this);
            this.model.on('change:width', this.update_width, this);
            this.model.on('change:height', this.update_height, this);
            this.model.on('change:frame_frame', this.update_framelist, this);
            this.model.on('change:atom_symbols', this.update_atom_symbols, this);
            this.model.on('change:atom_radii', this.update_atom_radii_dict, this);
            this.model.on('change:atom_colors', this.update_atom_colors_dict, this);
            this.update_width();
            this.update_height();
            this.update_atom_x();
            this.update_atom_y();
            this.update_atom_z();
            this.update_atom_radii_dict();
            this.update_atom_colors_dict();
            this.update_atom_symbols();
            this.update_framelist();

            this.init_container();
            this.init_canvas();
            this.init_3D();
            this.render_atoms(0);

            this.container.append(this.canvas);       // Lastly set the html
            this.setElement(this.container);          // objects and run.
            this.app.render();
            this.on('displayed', function() {
                self.app.animate();
                self.app.controls.handleResize();
            });
        },

        render_atoms: function(index) {
            /*"""
            render_frame
            --------------
            */
            this.index = index;
            var symbols = this.get_value(this.atom_symbols, this.index);
            var radii = utility.mapper(symbols, this.atom_radii_dict);
            var colors = utility.mapper(symbols, this.atom_colors_dict);
            var x = this.get_value(this.atom_x, this.index);
            var y = this.get_value(this.atom_y, this.index);
            var z = this.get_value(this.atom_z, this.index);
            this.app.scene.remove(this.points);
            this.points = this.app.add_points(x, y, z, colors, radii);
            this.set_camera(x, y, z);
        },

        render_bonds: function(index) {
            /*"""
            render_bonds
            ---------------
            */
        },

        get_value: function(obj, index) {
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
        },

        set_camera: function(x, y, z) {
            /*"""
            set_camera
            -------------
            */
            var n = x.length;
            var sums = [0.0, 0.0, 0.0];
            while (n--) {
                sums[0] += x[n];
                sums[1] += y[n];
                sums[2] += z[n];
            };
            console.log(sums);
            sums[0] /= n;
            sums[1] /= n;
            sums[2] /= n;
            console.log(sums);
            this.app.set_camera(100, 100, 100, sums[0], sums[1], sums[2]);
        },

        update_atom_x: function() {
            /*"""
            update_atom_x
            -----------
            Updates x component of nuclear coordinates.
            */
            this.atom_x = this.get_trait('atom_x');
        },

        update_atom_y: function() {
            /*"""
            update_atom_y
            -----------
            Updates y component of nuclear coordinates.
            */
            this.atom_y = this.get_trait('atom_y');
        },

        update_atom_z: function() {
            /*"""
            update_atom_z
            -----------
            Updates z component of nuclear coordinates.
            */
            this.atom_z = this.get_trait('atom_z');
        },

        update_framelist: function() {
            this.framelist = this.get_trait('frame_frame');
        },

        update_atom_symbols: function() {
            this.atom_symbols = this.get_trait('atom_symbols');
        },

        update_atom_radii_dict: function() {
            this.atom_radii_dict = this.get_trait('atom_radii');
        },

        update_atom_colors_dict: function() {
            this.atom_colors_dict = this.get_trait('atom_colors');
        },
    });

    return {'UniverseView': UniverseView};
});
