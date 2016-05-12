/*"""
==================
basis.js
==================
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/field': {
            exports: 'field'
        },

        'nbextensions/exa/num': {
            exports: 'num'
        },
    },
});


define([
    'nbextensions/exa/field',
    'nbextensions/exa/num'
], function(field, num) {
    class SolidHarmonic extends field.ScalarField {
        /*"""
        Compute the field corresponding to a real solid harmonic
        */
        constructor(dimensions, l, ml) {
            /*"""
            */
            console.log('SolidHarmonic');
            console.log(l);
            console.log(ml);
            super(dimensions, functions[l][String(ml)]);
        };
    };

    var functions = {
        0: {
            0: function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return 1 * R;
            },
        },

        1: {
            '1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return x * R ;
            },

            '0': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return z * R;
            },

            '-1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return y * R;
            },
        },

        2: {
            '2': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return 0.5 * Math.sqrt(3) * (x * x - y * y) * R;
            },

            '1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return Math.sqrt(3) * x * z * R;
            },

            '0': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return 0.5 * (3 * z * z - r2) * R;
            },

            '-1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return Math.sqrt(3) * y * z * R;
            },

            '-2': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                return Math.sqrt(3) * x * y * R;
            },
        },

        3: {
            '-3': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var sq10 = Math.sqrt(10);
                var S = 3 * sq10 * x * x * y / 4 - sq10 * y * y * y / 4;
                return R * S;
            },

            '-2': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var S = Math.sqrt(15) * x * y * z;
                return R * S;
            },

            '-1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var sq6 = Math.sqrt(6);
                var S = -sq6 * x * x * y / 4 - sq6 * y * y * y / 4 + sq6 * y * z * z;
                return R * S;
            },

            '0': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var S = -3 * x * x * z / 2 - 3 * y * y * z / 2 + z * z * z;
                return R * S;
            },

            '1': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var sq6 = Math.sqrt(6);
                var S = -sq6 * x * x * x / 4 - sq6 * x * y * y / 4 + sq6 * x * z * z;
                return R * S;
            },

            '2': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var sq15 = Math.sqrt(15);
                var S = sq15 * x * x * z / 2 - sq15 * y * y * z / 2
                return R * S;
            },

            '3': function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                var R = Math.exp(-Math.sqrt(r2));
                var sq10 = Math.sqrt(10);
                var S = sq10 * x * x * x / 4 - 3 * sq10 * x * y * y / 4
                return R * S;
            },
        },
    };

    return SolidHarmonic;
});
