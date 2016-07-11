// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
harmonics.js
##################
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
    class SolidHarmonic extends field.ScalarField {
        constructor(l, m, dimensions) {
            console.log(l);
            console.log(m);
            var sm = String(m);
            var sh = solid_harmonics[l][m];
            var func = function(x, y, z) {
                var r2 = x*x + y*y + z*z;
                return sh(x, y, z) * Math.exp(-Math.sqrt(r2));
            };
            super(dimensions, func);
            /*
            string = ''
            for i in smth:
                string = string + 'SolidHarmonic[l0][m0](x, y, z) + d1*SolidHarmonic[l1][m1](x, y, z) * Math.exp(-alpha1 * r2)'
             return new Function(string)
             */
        };
    };

    var get_solid_harmonic = function(l, m) {
        m = String(m);
        return solid_harmonics[l][m];
    };

    // Some numerical constants for convenience in the formulas below
    var t2 = Math.sqrt(2);
    var t3 = Math.sqrt(3);
    var t5 = Math.sqrt(5);
    var t6 = Math.sqrt(6);
    var t7 = Math.sqrt(7);
    var t10 = Math.sqrt(10);
    var t14 = Math.sqrt(14);
    var t15 = Math.sqrt(15);
    var t21 = Math.sqrt(21);
    var t30 = Math.sqrt(30);
    var t35 = Math.sqrt(35);
    var t42 = Math.sqrt(42);
    var t70 = Math.sqrt(70);
    var t105 = Math.sqrt(105);
    var t154 = Math.sqrt(154);
    var t210 = Math.sqrt(210);
    var t231 = Math.sqrt(231);
    var t429 = Math.sqrt(429);
    var t462 = Math.sqrt(462);
    var t6006 = Math.sqrt(6006);

    // Common multiplication convenience functions
    var r4 = function(r) {
        return r*r*r*r;
    };
    var r6 = function(r) {
        return r*r*r * r*r*r;
    };


    // What follows are real regular cartesian solid harmonics
    // The were computed using the recursion relation provided
    // in exatomic.algorithms.harmonics
    var solid_harmonics = {
        0: {
            '0': function(x, y, z) {
                return 1;
            },
        },

        1: {
            '-1': function(x, y, z) {
                return y;
            },
            '0': function(x, y, z) {
                return z;
            },
            '1': function(x, y , z) {
                return x;
            },
        },

        '2': {
            '-2': function(x, y, z) {
                return t3*x*y;
            },
            '-1': function(x, y, z) {
                return t3*y*z;
            },
            '0': function(x, y, z) {
                return -x*x/2 - y*y/2 + z*z;
            },
            '1': function(x, y, z) {
                return t3*x*z;
            },
            '2': function(x, y, z) {
                return t3/2*(x*x - y*y);
            }
        },

        '3': {
            '-3': function(x, y, z) {
                return t10/4*y*(3*x*x - y*y);
            },
            '-2': function(x, y, z) {
                return t15*x*y*z;
            },
            '-1': function(x, y, z) {
                return t6/4*y*(-x*x - y*y + 4*z*z);
            },
            '0': function(x, y, z) {
                return z/2*(-3*x*x - 3*y*y + 2*z*z);
            },
            '1': function(x, y, z) {
                return t6/4*x*(-x*x -y*y + 4*z*z);
            },
            '2': function(x, y, z) {
                return t15/2*z*(x*x - y*y);
            },
            '3': function(x, y, z) {
                return t10/4*x*(x*x - 3*y*y);
            }
        },

        '4': {
            '-4': function(x, y, z) {
                return t35/2*x*y*(x*x -y*y);
            },
            '-3': function(x, y, z) {
                return t70/4*y*z*(3*x*x - y*y);
            },
            '-2': function(x, y, z) {
                return t5/2*x*y*(-x*x - y*y +6*z*z);
            },
            '-1': function(x, y, z) {
                return t10/4*y*z*(-3*x*x - 3*y*y + 4*z*z);
            },
            '0': function(x, y, z) {
                return 3*r4(x)/8 + 3*x*x*y*y/4 - 3*x*x*z*z + 3*r4(y)/8 - 3*y*y*z*z + r4(z);
            },
            '1': function(x, y, z) {
                return t10/4*x*z*(-3*x*x - 3*y*y + 4*z*z);
            },
            '2': function(x, y, z) {
                return t5/4*(x*x - y*y)*(-x*x - y*y + 6*z*z);
            },
            '3': function(x, y, z) {
                return t70/4*x*z*(x*x - 3*y*y);
            },
            '4': function(x, y, z) {
                return t35/8*(r4(x) - 6*x*x*y*y + r4(y));
            }
        },

        '5': {
            '-5': function(x, y, z) {
                return 3/16*t14*y*(5*r4(x) - 10*x*x*y*y + r4(y));
            },
            '-4': function(x, y, z) {
                return 3/2*t35*x*y*z*(x*x - y*y);
            },
            '-3': function(x, y, z) {
                return t70/16*y*(3*x*x - y*y)*(-x*x - y*y + 8*z*z);
            },
            '-2': function(x, y, z) {
                return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z);
            },
            '-1': function(x, y, z) {
                return t15/8*y*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z));
            },
            '0': function(x, y, z) {
                return z/8*(15*r4(x) + 30*x*x*y*y - 40*x*x*z*z + 15*r4(y) - 40*y*y*z*z + 8*r4(z));
            },
            '1': function(x, y, z) {
                return t15/8*x*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z));
            },
            '2': function(x, y, z) {
                return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z);
            },
            '3': function(x, y, z) {
                return t70/16*x*(9*z*z*(x*x - 3*y*y) + (-x*x + 3*y*y)*(x*x + y*y + z*z));
            },
            '4': function(x, y, z) {
                return 3/8*t35*z*(r4(x) - 6*x*x*y*y + r4(y));
            },
            '5': function(x, y, z) {
                return 3/16*t14*x*(r4(x) - 10*x*x*y*y + 5*r4(y));
            }
        },

        '6': {
            '-6': function(x, y, z) {
                return t462/16*x*y*(3*r4(x) -10*x*x*y*y + 3*r4(y));
            },
            '-5': function(x, y, z) {
                return 3*t154*y*z*(5*r4(x) - 10*x*x*y*y + r4(y))/16;
            },
            '-4': function(x, y, z) {
                return 3*t7*x*y*(x*x - y*y)*(-x*x - y*y + 10*z*z)/4;
            },
            '-3': function(x, y, z) {
                return t210/16*y*z*(3*x*x - y*y)*(-3*x*x - 3*y*y + 8*z*z);
            },
            '-2': function(x, y, z) {
                return t210*x*y*(r4(x) + 2*x*x*y*y - 16*x*x*z*z + r4(y) - 16*y*y*z*z + 16*r4(z))/16;
            },
            '-1': function(x, y, z) {
                return t21*y*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8;
            },
            '0': function(x, y, z) {
                return -5*r6(x)/16 - 15*r4(x)*y*y/16 + 45*r4(x)*z*z/8 - 15*x*x*r4(y)/16 + 45*x*x*y*y*z*z/4 - 15*x*x*r4(z)/2 - 5*r6(y)/16 + 45*r4(y)*z*z/8 - 15*y*y*r4(z)/2 + r6(z);
            },
            '1': function(x, y, z) {
                return t21*x*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8;
            },
            '2': function(x, y, z) {
                return -t210/32*(x*x - y*y)*(11*z*z*(x*x + y*y - 2*z*z) - (x*x + y*y - 6*z*z)*(x*x + y*y + z*z));
            },
            '3': function(x, y, z) {
                return t210/16*x*z*(-x*x + 3*y*y)*(3*x*x + 3*y*y - 8*z*z);
            },
            '4': function(x, y, z) {
                return 3*t7/16*(-r6(x) + 5*r4(x)*y*y + 10*r4(x)*z*z + 5*x*x*r4(y) - 60*x*x*y*y*z*z - r6(y) + 10*r4(y)*z*z);
            },
            '5': function(x, y, z) {
                return 3/16*t154*x*z*(r4(x) - 10*x*x*y*y + 5*r4(y));
            },
            '6': function(x, y, z) {
                return t462/32*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y));
            }
        },

        7: {
            '-7': function(x, y, z) {
                return t429/32*y*(7*r6(x) - 35*r4(x)*y*y + 21*x*x*r4(y) - r6(y));
            },
            '-6': function(x, y, z) {
                return t6006/16*x*y*z*(3*r4(x) - 10*x*x*y*y + 3*r4(y));
            },
            '-5': function(x, y, z) {
                return t231/32*y*(-x*x - y*y + 12*z*z)*(-x*x*(-x*x + 3*y*y) + 4*x*x*(x*x - y*y) - y*y*(3*x*x - y*y));
            },
            '-4': function(x, y, z) {
                return t231/4*x*y*z*(x*x - y*y)*(-3*x*x - 3*y*y + 10*z*z);
            },
            '-3': function(x, y, z) {
                return t21/32*y*(3*x*x - y*y)*(z*z*(-39*x*x - 39*y*y + 104*z*z) - 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z));
            },
            '-2': function(x, y, z) {
                return t42/16*x*y*z*(15*r4(x) + 30*x*x*y*y - 80*x*x*z*z + 15*r4(y) - 80*y*y*z*z + 48*r4(z));
            },
            '-1': function(x, y, z) {
                return t7/32*y*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z));
            },
            '0': function(x, y, z) {
                return z/16*(-35*r6(x) - 105*r4(x)*y*y + 210*r4(x)*z*z - 105*x*x*r4(y) + 420*x*x*y*y*z*z - 168*x*x*r4(z) - 35*r6(y) + 210*r4(y)*z*z - 168*y*y*r4(z) + 16*r6(z));
            },
            '1': function(x, y, z) {
                return t7/32*x*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z));
            },
            '2': function(x, y, z) {
                return t42/32*z*(15*r6(x) + 15*r4(x)*y*y - 80*r4(x)*z*z - 15*x*x*r4(y) + 48*x*x*r4(z) - 15*r6(y) + 80*r4(y)*z*z - 48*y*y*r4(z));
            },
            '3': function(x, y, z) {
                return t21/32*x*(-x*x + 3*y*y)*(z*z*(39*x*x + 39*y*y - 104*z*z) + 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z));
            },
            '4': function(x, y, z) {
                return t231/16*z*(-3*r6(x) + 15*r4(x)*y*y + 10*r4(x)*z*z + 15*x*x*r4(y) - 60*x*x*y*y*z*z - 3*r6(y) + 10*r4(y)*z*z);
            },
            '5': function(x, y, z) {
                return t231/32*x*(-r6(x) + 9*r4(x)*y*y + 12*r4(x)*z*z + 5*x*x*r4(y) - 120*x*x*y*y*z*z - 5*r6(y) + 60*r4(y)*z*z);
            },
            '6': function(x, y, z) {
                return t6006/32*z*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y));
            },
            '7': function(x, y, z) {
                return t429/32*x*(r6(x) - 21*r4(x)*y*y + 35*x*x*r4(y) -7*r6(y));
            }
        }
    };

    return {
        get_solid_harmonic: get_solid_harmonic,
        solid_harmonics: solid_harmonics,
        SolidHarmonic: SolidHarmonic
    };
});
