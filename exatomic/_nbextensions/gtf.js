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
    class GTF extends field.ScalarField {
        /*"""
        */
        constructor(dimensions, which) {
            super(dimensions, primitives[which]);
            this.function = which;
        };
    };

    var primitives = {
        's': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return 5 * Math.exp(-r2);
        },
        /*
        '2s': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return Math.exp(-r2);
        },
        */
        'px': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return 5 * x * Math.exp(-r2);
        },

        'py': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return 5 * y * Math.exp(-r2);
        },

        'pz': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return 5 * z * Math.exp(-r2);
        },

        /*
        '3s': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            return Math.exp(-r2);
        },
        '3px': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return x * Math.exp(-r2);
        },

        '3py': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return y * r * Math.exp(-r2);
        },

        '3pz': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return z * r * Math.exp(-r2);
        },
        */
        'd200': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * x2 * Math.exp(-r2);
        },

        'd110': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * x * y * Math.exp(-r2);
        },

        'd101': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * x * z * Math.exp(-r2);
        },


        'd020': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * y2 * Math.exp(-r2);
        },

        'd011': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * y * z * Math.exp(-r2);
        },

        'd002': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return 20 * z2 * Math.exp(-r2);
        },
        /*
        '3dx2-y2': function(x, y, z) {
            var x2 = x*x;
            var y2 = y*y;
            var z2 = z*z;
            var r2 = x2 + y2 + z2;
            var r = Math.sqrt(r2);
            return (x2 - y2) * Math.exp(-r2);
        },
        */

    };

    return GTF;
});
