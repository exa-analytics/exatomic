// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
Universe View
###################
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'base'
        },

        'nbextensions/exa/exatomic/app': {
            exports: 'UniverseApp'
        },

        'nbextensions/exa/exatomic/test': {
            exports: 'UniverseTestApp'
        },
    },
});


define([
    'nbextensions/exa/container',
    'nbextensions/exa/exatomic/test',
    'nbextensions/exa/exatomic/app'
], function(base, UniverseTestApp, UniverseApp) {
    class UniverseView extends base.ContainerView {
        /*"""
        UniverseView
        ================
        */
        init() {
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
            this.get_framelist();
            this.get_frame_ox();
            this.get_frame_oy();
            this.get_frame_oz();
            this.get_frame_xi();
            this.get_frame_xj();
            this.get_frame_xk();
            this.get_frame_yi();
            this.get_frame_yj();
            this.get_frame_yk();
            this.get_frame_zi();
            this.get_frame_zj();
            this.get_frame_zk();
            this.model.on('change:frame_frame', this.get_framelist, this);
            this.model.on('change:frame_ox', this.get_frame_ox, this);
            this.model.on('change:frame_oy', this.get_frame_ox, this);
            this.model.on('change:frame_oz', this.get_frame_ox, this);
            this.model.on('change:frame_xi', this.get_frame_ox, this);
            this.model.on('change:frame_xj', this.get_frame_ox, this);
            this.model.on('change:frame_xk', this.get_frame_ox, this);
            this.model.on('change:frame_yi', this.get_frame_ox, this);
            this.model.on('change:frame_yj', this.get_frame_ox, this);
            this.model.on('change:frame_yk', this.get_frame_ox, this);
            this.model.on('change:frame_zi', this.get_frame_ox, this);
            this.model.on('change:frame_zj', this.get_frame_ox, this);
            this.model.on('change:frame_zk', this.get_frame_ox, this);

            // Atom, UnitAtom, ...
            this.get_atom_x();
            this.get_atom_y();
            this.get_atom_z();
            this.get_atom_radii_dict();
            this.get_atom_colors_dict();
            this.get_atom_symbols();
            this.get_atom_set();
            this.model.on('change:atom_x', this.get_x, this);
            this.model.on('change:atom_y', this.get_y, this);
            this.model.on('change:atom_z', this.get_z, this);
            this.model.on('change:atom_radii', this.get_atom_radii_dict, this);
            this.model.on('change:atom_colors', this.get_atom_colors_dict, this);
            this.model.on('change:atom_symbols', this.get_atom_symbols, this);
            this.model.on('change:atom_set', this.get_atom_set, this);

            // Two, PeriodicTwo
            this.get_two_bond0();
            this.get_two_bond1();
            this.model.on('change:two_bond0', this.get_two_bond0, this);
            this.model.on('change:two_bond1', this.get_two_bond1, this);

            //Orbital
            this.get_gaussianbasisset_d();
            this.get_gaussianbasisset_l();
            this.get_gaussianbasisset_alpha();
            this.get_gaussianbasisset_shell_function();
            this.model.on('change:gaussianbasisset_d', this.get_gaussianbasisset_d, this);
            this.model.on('change:gaussianbasisset_l', this.get_gaussianbasisset_l, this);
            this.model.on('change:gaussianbasisset_alpha', this.get_gaussianbasisset_alpha, this);
            this.model.on('change:gaussianbasisset_shell_function', this.get_gaussianbasisset_shell_function, this);

            // MOMatrix
            this.get_momatrix_coefficient();
            this.model.on('change:momatrix_coefficient', this.get_momatrix_coefficient, this);

            //SphericalGTFOrder
            this.get_sphericalgtforder_l();
            this.get_sphericalgtforder_ml();
            this.model.on('change:sphericalgtforder_l', this.get_sphericalgtforder_l, this);
            this.model.on('change:sphericalgtforder_ml', this.get_sphericalgtforder_ml, this);

            //CartesianGTFOrder
            this.get_cartesiangtforder_x();
            this.get_cartesiangtforder_y();
            this.get_cartesiangtforder_z();
            this.model.on('change:cartesiangtforder_x', this.get_cartesiangtforder_x, this);
            this.model.on('change:cartesiangtforder_y', this.get_cartesiangtforder_y, this);
            this.model.on('change:cartesiangtforder_z', this.get_cartesiangtforder_z, this);

            //UField3D
            this.get_atomicfield_ox();
            this.get_atomicfield_oy();
            this.get_atomicfield_oz();
            this.get_atomicfield_nx();
            this.get_atomicfield_ny();
            this.get_atomicfield_nz();
            this.get_atomicfield_dxi();
            this.get_atomicfield_dxj();
            this.get_atomicfield_dxk();
            this.get_atomicfield_dyi();
            this.get_atomicfield_dyj();
            this.get_atomicfield_dyk();
            this.get_atomicfield_dzi();
            this.get_atomicfield_dzj();
            this.get_atomicfield_dzk();
            this.model.on('change:atomicfield_ox', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_oy', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_oz', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_nx', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_ny', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_nz', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dxi', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dxj', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dxk', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dyi', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dyj', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dyk', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dzi', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dzj', this.get_atomicfield_ox, this);
            this.model.on('change:atomicfield_dzk', this.get_atomicfield_ox, this);
        };

        get_framelist() {
            this.framelist = this.get_trait('frame_frame');
        };

        get_atom_x() {
            this.atom_x = this.get_trait('atom_x');
        };
        get_atom_y() {
            this.atom_y = this.get_trait('atom_y');
        };
        get_atom_z() {
            this.atom_z = this.get_trait('atom_z');
        };
        get_atom_symbols() {
            this.atom_symbols = this.get_trait('atom_symbols');
        };
        get_atom_radii_dict() {
            this.atom_radii_dict = this.get_trait('atom_radii');
        };
        get_atom_colors_dict() {
            this.atom_colors_dict = this.get_trait('atom_colors');
        };
        get_atom_set() {
            this.atom_set = this.get_trait('atom_set');
        };

        get_two_bond0() {
            this.two_bond0 = this.get_trait('two_bond0');
        };
        get_two_bond1() {
            this.two_bond1 = this.get_trait('two_bond1');
        };
        get_frame_ox() {
            this.frame_ox = this.get_trait('frame_ox');
        };
        get_frame_oy() {
            this.frame_oy = this.get_trait('frame_oy');
        };
        get_frame_oz() {
            this.frame_oz = this.get_trait('frame_oz');
        };
        get_frame_xi() {
            this.frame_xi = this.get_trait('frame_xi');
        };
        get_frame_xj() {
            this.frame_xj = this.get_trait('frame_xj');
        };
        get_frame_xk() {
            this.frame_xk = this.get_trait('frame_xk');
        };
        get_frame_yi() {
            this.frame_yi = this.get_trait('frame_yi');
        };
        get_frame_yj() {
            this.frame_yj = this.get_trait('frame_yj');
        };
        get_frame_yk() {
            this.frame_yk = this.get_trait('frame_yk');
        };
        get_frame_zi() {
            this.frame_zi = this.get_trait('frame_zi');
        };
        get_frame_zj() {
            this.frame_zj = this.get_trait('frame_zj');
        };
        get_frame_zk() {
            this.frame_zk = this.get_trait('frame_zk');
        };

        get_gaussianbasisset_l() {
            this.gaussianbasisset_l = this.get_trait('gaussianbasisset_l');
        };
        get_gaussianbasisset_d() {
            this.gaussianbasisset_d = this.get_trait('gaussianbasisset_d');
        };
        get_gaussianbasisset_alpha() {
            this.gaussianbasisset_alpha = this.get_trait('gaussianbasisset_alpha');
        };
        get_gaussianbasisset_shell_function() {
            this.gaussianbasisset_shell_function = this.get_trait('gaussianbasisset_shell_function');
        };

        get_momatrix_coefficient() {
            this.momatrix_coefficient = this.get_trait('momatrix_coefficient');
        };
        get_sphericalgtforder_l() {
            this.sphericalgtforder_l = this.get_trait('sphericalgtforder_l');
        };
        get_sphericalgtforder_ml() {
            this.sphericalgtforder_ml = this.get_trait('sphericalgtforder_ml');
        };
        get_cartesiangtforder_x() {
            this.cartesiangtforder_x = this.get_trait('cartesiangtforder_x');
        };
        get_cartesiangtforder_y() {
            this.cartesiangtforder_y = this.get_trait('cartesiangtforder_y');
        };
        get_cartesiangtforder_z() {
            this.cartesiangtforder_z = this.get_trait('cartesiangtforder_z');
        };

        get_atomicfield_ox() {
            this.atomicfield_ox = this.get_trait('atomicfield_ox');
        };
        get_atomicfield_oy() {
            this.atomicfield_oy = this.get_trait('atomicfield_oy');
        };
        get_atomicfield_oz() {
            this.atomicfield_oz = this.get_trait('atomicfield_oz');
        };
        get_atomicfield_nx() {
            this.atomicfield_nx = this.get_trait('atomicfield_nx');
        };
        get_atomicfield_ny() {
            this.atomicfield_ny = this.get_trait('atomicfield_ny');
        };
        get_atomicfield_nz() {
            this.atomicfield_nz = this.get_trait('atomicfield_nz');
        };
        get_atomicfield_dxi() {
            this.atomicfield_dxi = this.get_trait('atomicfield_dxi');
        };
        get_atomicfield_dxj() {
            this.atomicfield_dxj = this.get_trait('atomicfield_dxj');
        };
        get_atomicfield_dxk() {
            this.atomicfield_dxk = this.get_trait('atomicfield_dxk');
        };
        get_atomicfield_dyi() {
            this.atomicfield_dyi = this.get_trait('atomicfield_dyi');
        };
        get_atomicfield_dyj() {
            this.atomicfield_dyj = this.get_trait('atomicfield_dyj');
        };
        get_atomicfield_dyk() {
            this.atomicfield_dyk = this.get_trait('atomicfield_dyk');
        };
        get_atomicfield_dzi() {
            this.atomicfield_dzi = this.get_trait('atomicfield_dzi');
        };
        get_atomicfield_dzj() {
            this.atomicfield_dzj = this.get_trait('atomicfield_dzj');
        };
        get_atomicfield_dzk() {
            this.atomicfield_dzk = this.get_trait('atomicfield_dzk');
        };
    };

    return {UniverseView: UniverseView};
});
