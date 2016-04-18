/*"""
==================
field.js
==================
Handling cube fields
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/field': {
            exports: 'field'
        },
    },
});


define([
    'nbextensions/exa/field'
], function(field) {
    class CubeField extends field.ScalarField {
        /*"""
        CubeField
        =============
        JS repr. of .cube file field values and dimensions.
        */
        constructor(ox, oy, oz, nx, ny, nz, dxi, dxj, dxk, dyi, dyj, dyk, dzi, dzj, dzk, values) {
            console.log(ox);
            console.log(nx);
            console.log(dxi);
            console.log(dyj);
            console.log(dzk);
            var dimensions = {
                'x': gen_array(nx, ox, dxi, dyi, dzi),
                'y': gen_array(ny, oy, dxj, dyj, dzj),
                'z': gen_array(nz, oz, dxk, dyk, dzk)
            };
            console.log(dimensions);
            super(dimensions, values);
        };

    };

    var gen_array = function(nr, or, dx, dy, dz) {
        var r = new Float32Array(nr);
        r[0] = or;
        for (let i=1; i<nr; i++) {
            r[i] = r[i-1] + dx + dy + dz;
        };
        return r;
    };

    return CubeField;
});
