/*"""
==================
gto.js
==================
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
    class GTO extends field.ScalarField {
        /*"""
        */
        constructor(dimensions, which) {
            super(dimensions, primitives[which]);
        };
    };

    var primitives = {
        '1s': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return Math.exp(-r2);
        },

        '2s': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return r * Math.exp(-r2);
        },

        '2px': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return x * Math.exp(-r2);
        },

        '2py': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return y * Math.exp(-r2);
        },

        '2pz': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return z * Math.exp(-r2);
        },

        '3s': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return r2 * Math.exp(-r2);
        },

        '3px': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return x2 * Math.exp(-r2);
        },

        '3py': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return y2 * Math.exp(-r2);
        },

        '3pz': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return z2 * Math.exp(-r2);
        },
    };

    return GTO;
});
