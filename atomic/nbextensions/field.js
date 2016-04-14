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
        constructor(ox, oy, oz, nx, ny, nz, xi, xj, xk, yi, yj, yk, zi, zj, zk, values) {
            var dimensions = {
                'xmin': ox,
                'ymin': oy,
                'zmin': oz,
                'xmax': Math.sqrt(xi*xi + xj*xj + xk*xk),
                'ymax': Math.sqrt(yi*yi + yj*yj + yk*yk),
                'zmax': Math.sqrt(zi*zi + zj*zj + zk*zk),
                'nx': nx,
                'ny': ny,
                'nz': nz
            };
            super(dimensions, values);
        };
    };

    return CubeField;
});
