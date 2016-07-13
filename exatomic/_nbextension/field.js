// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
field.js
###############
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/field': {
            exports: 'field'
        },
        'nbextensions/exa/num': {
            exports: 'num'
        }
    },
});


define([
    'nbextensions/exa/field',
    'nbextensions/exa/num'
], function(field, num) {
    class AtomicField extends field.ScalarField {
        /*"""
        AtomicField
        =============
        JS repr. of .cube file field values and dimensions.
        */
        constructor(dimensions, values) {
            /*
            var dimensions = {
                'x': num.gen_array(nx, ox, dxi, dyi, dzi),
                'y': num.gen_array(ny, oy, dxj, dyj, dzj),
                'z': num.gen_array(nz, oz, dxk, dyk, dzk)
            };
            */
            super(dimensions, values);
        };

    };

    //var gen_array = function(nr, or, dx, dy, dz) {
        /*"""
        gen_array
        =============
        Generates discrete spatial points in a given basis. Used to generate
        x, y, z spatial values for the cube field. In most cases, for the x
        basis vector, dy and dz are zero ("cube-like").
        */
/*
        var r = new Float32Array(nr);
        r[0] = or;
        for (var i=1; i<nr; i++) {
            r[i] = r[i-1] + dx + dy + dz;
        };
        return r;
    };
        */

    return AtomicField;

});
