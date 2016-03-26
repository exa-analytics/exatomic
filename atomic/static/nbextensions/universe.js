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
        }
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/container',
    'nbextensions/exa/atomic/app',
], function(widget, container, app){
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
            this.update_width();
            this.update_height();
            this.update_atom_x();
            this.update_atom_y();
            this.update_atom_z();

            this.init_container();
            this.init_canvas();
            this.init_3D();
            var x = this.get_value(this.atom_x, 0);
            var y = this.get_value(this.atom_y, 0);
            var z = this.get_value(this.atom_z, 0);
            this.app.add_points(x, y, z);

            this.app.default_camera();
            this.container.append(this.canvas);       // Lastly set the html
            this.setElement(this.container);          // objects and run.
            this.app.render();
            this.on('displayed', function() {
                self.app.animate();
                self.app.controls.handleResize();
            });
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
    });

    return {'UniverseView': UniverseView};
});
