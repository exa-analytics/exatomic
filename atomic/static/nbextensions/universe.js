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
            this.update_atom_x();
            this.update_atom_y();
            this.update_atom_z();
            console.log(this.atom_x);
            console.log(this.atom_y);
            console.log(this.atom_z);
            if (this.atom_z.length == undefined) {
                console.log('true');
            } else {
                console.log('false');
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
