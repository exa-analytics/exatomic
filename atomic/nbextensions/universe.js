/*"""
=========================================================
Universe View
=========================================================
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'base'
        },

        'nbextensions/exa/atomic/app': {
            exports: 'UniverseApp'
        },

        'nbextensions/exa/atomic/test': {
            exports: 'UniverseTestApp'
        },
    },
});


define([
    'nbextensions/exa/container',
    'nbextensions/exa/atomic/test',
    'nbextensions/exa/atomic/app'
], function(base, UniverseTestApp, UniverseApp) {
    class UniverseView extends base.ContainerView {
        /*"""
        UniverseView
        ================
        */
        init() {
            console.log(UniverseApp);
            this.init_listeners();
            this.if_empty();
        };

        if_empty() {
            /*"""
            if_empty
            ----------
            Create a UniverseTestApp application?
            */
            var check = this.get_trait('test');
            if (check === true) {
                console.log('Empty universe, displaying test interface!');
                this.app = new UniverseTestApp(this);
            } else {
                this.app = new UniverseApp(this);
            };
        };

        init_listeners() {
            /*"""
            init_listeners
            ---------------
            Set up the frontend to listen for changes on the backend
            */
            // Frame
            this.update_framelist();
            this.model.on('change:frame_frame', this.update_framelist, this);

            // Atom, UnitAtom, ...
            this.update_atom_x();
            this.update_atom_y();
            this.update_atom_z();
            this.update_atom_radii_dict();
            this.update_atom_colors_dict();
            this.update_atom_symbols();
            this.model.on('change:atom_x', this.update_x, this);
            this.model.on('change:atom_y', this.update_y, this);
            this.model.on('change:atom_z', this.update_z, this);
            this.model.on('change:atom_radii', this.update_atom_radii_dict, this);
            this.model.on('change:atom_colors', this.update_atom_colors_dict, this);
            this.model.on('change:atom_symbols', this.update_atom_symbols, this);

            // Two, PeriodicTwo
            this.update_two_bond0();
            this.update_two_bond1();
            this.model.on('change:two_bond0', this.update_two_bond0, this);
            this.model.on('change:two_bond1', this.update_two_bond1, this);
        };

        update_atom_x() {
            this.atom_x = this.get_trait('atom_x');
        };

        update_atom_y() {
            this.atom_y = this.get_trait('atom_y');
        };

        update_atom_z() {
            this.atom_z = this.get_trait('atom_z');
        };

        update_framelist() {
            this.framelist = this.get_trait('frame_frame');
        };

        update_atom_symbols() {
            this.atom_symbols = this.get_trait('atom_symbols');
        };

        update_atom_radii_dict() {
            this.atom_radii_dict = this.get_trait('atom_radii');
        };

        update_atom_colors_dict() {
            this.atom_colors_dict = this.get_trait('atom_colors');
        };

        update_two_bond0() {
            this.two_bond0 = this.get_trait('two_bond0');
        };

        update_two_bond1() {
            this.two_bond1 = this.get_trait('two_bond1');
        };
    };

    return {UniverseView: UniverseView};
});
