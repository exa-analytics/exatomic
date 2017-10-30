// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
================
jupyter-exatomic-util.js
================
Helper JS functions.
*/

"use strict";

var create_float_array_xyz = function(x, y, z) {
    var nx = x.length || 1;
    var ny = y.length || 1;
    var nz = z.length || 1;
    var n = Math.max(nx, ny, nz);
    x = (nx == 1) ? repeat_float(x, n) : x;
    y = (ny == 1) ? repeat_float(y, n) : y;
    z = (nz == 1) ? repeat_float(z, n) : z;
    var xyz = new Float32Array(n * 3)
    for (var i=0, i3=0; i<n; i++, i3+=3) {
        xyz[i3] = x[i];
        xyz[i3+1] = y[i];
        xyz[i3+2] = z[i];
    };
    return xyz;
};

var gen_field_arrays = function(fps) {
    return {x: linspace(fps.ox, fps.fx, fps.nx),
            y: linspace(fps.oy, fps.fy, fps.ny),
            z: linspace(fps.oz, fps.fz, fps.nz)}
};

//var xrange = function(orig, delta, num) {
//    console.log(orig, delta, num);
//    var end = orig + (num - 1) * delta;
//    return linspace(orig, end, num);
//};

var repeat_float = function(value, n) {
    var array = new Float32Array(n);
    for (var i=0; i<n; i++) {
        array[i] = value
    };
    return array;
};

var repeat_obj = function(value, n) {
    var obj = [];
    for (var i=0; i<n; i++) {
        obj.push(value);
    };
    return obj;
};

var mapper = function(indices, map) {
    var n = indices.length;
    var mapped = [];
    for (var i=0; i<n; i++) {
        mapped.push(map[indices[i]])
    };
    return mapped;
};

var linspace = function(min, max, n) {
    var n1 = n - 1;
    var step = (max - min) / n1;
    var array = [min];
    for (var i=0; i<n1; i++) {
        min += step;
        array.push(min);
    };
    return new Float32Array(array);
};

var Sphere = function(x, y, z) {
    return (x * x + y * y + z * z);
}

var Ellipsoid = function(x, y, z, a, b, c) {
    a = (a === undefined) ? 2.0 : a;
    b = (b === undefined) ? 1.5 : b;
    c = (c === undefined) ? 1.0 : c;
    return 2 * ((x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c));
};

var Torus = function(x, y, z, c) {
    c = (c === undefined) ? 2.5 : c;
    return  2 * (Math.pow(c - Math.sqrt(x*x + y*y), 2) + z*z);
}

var compute_field = function(xs, ys, zs, n, func) {
    var i = 0;
    var vals = new Float32Array(n);
    for (var x of xs) {
        for (var y of ys) {
            for (var z of zs) {
                vals[i] = func(x, y, z);
                i += 1;
            };
        };
    };
    return vals
};

var factorial2 = function(n) {
    if (n < -1) { return 0;
    } else if (n < 2) { return 1;
    } else {
        var prod = 1;
        while (n > 0) {
            prod *= n;
            n -= 2;
        };
        return prod;
    };
};

var normalize_gaussian = function(alpha, L) {
    var prefac = Math.pow((2 / Math.PI), 0.75);
    var numer = Math.pow(2, L) * Math.pow(alpha, ((L + 1.5) / 2));
    var denom = Math.pow(factorial2(2 * L - 1), 0.5);
    return prefac * numer / denom;
};


// the following field attributes are used:
// field.nx, field.ny, field.nz, field.x, field.y, field.z, field.values
var scalar_field = function(dims, funcvals) {
    /*"""
    scalar_field
    ==============
    Args:
        dims: {"ox": ox, "nx": nx, "dxi": dxi, "dxj": dxj, "dxk": dxk,
               "oy": oy, "ny": ny, "dyi": dyi, "dyj": dyj, "dyk": dyk,
               "oz": oz, "nz": nz, "dzi": dzi, "dzj": dzj, "dzk": dzk}

    Note:
        The dimensions argument can alternatively be
        {"x": xarray, "y": yarray, "z": zarray}
        if they have already been constructed but in this
        case the arrays should form cubic discrete points
    */
    var x = dims["x"] || linspace(dims.ox, dims.fxi, dims.nx);
    var y = dims["y"] || linspace(dims.oy, dims.fyj, dims.ny);
    var z = dims["z"] || linspace(dims.oz, dims.fzk, dims.nz);
    var nx = x.length;
    var ny = y.length;
    var nz = z.length;
    var n = nx * ny * nz;
    if (typeof funcvals === "function") {
        var values = compute_field(x, y, z, n, funcvals);
   } else {
        var values = new Float32Array(funcvals);
    };
    return { "x": x,   "y": y,   "z": z,
            "nx": nx, "ny": ny, "nz": nz,
            "values": values}
};


var Gaussian = function(dims, which) {
    var alpha = 0.01;
    var lmap = {"s": 0, "p": 1, "d": 2, "f": 3};
    var norm = normalize_gaussian(alpha, lmap[which[0]]);
    var xmap = {"x": 0, "y": 0, "z": 0};
    if (lmap[which[0]] == 1) {
        xmap[which[1]] = 1
    } else if (lmap[which[0]] > 1) {
        xmap["x"] = which[1];
        xmap["y"] = which[2];
        xmap["z"] = which[3];
    };
    var func = function(x, y, z) {
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        return (norm * Math.pow(x, xmap["x"]) * Math.pow(y, xmap["y"])
                     * Math.pow(z, xmap["z"]) * Math.exp(-alpha * r2));
    };
    return scalar_field(dims, func);
};


var Hydrogenic = function(dims, which) {
    var Z = 1;
    var z32 = Math.pow(Z, 3/2);
    var sqpi = Math.sqrt(Math.PI);
    var sq2pi = Math.sqrt(2 * Math.PI);
    var sq3pi = Math.sqrt(3 * Math.PI);
    var rnorm = 1 / Math.sqrt(2430);
    var func = function(x, y, z) {
        var norm, prefac, ret;
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var sigma = Z * r;
        var rbody = z32 * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        switch (which) {
            case "1s":
                ret = 1 / sqpi * z32 * Math.exp(-sigma);
                break;
            case "2s":
                norm = 1 / (4 * sq2pi) * z32;
                ret = norm * (2 - sigma) * Math.exp(-sigma / 2);
                break;
            case "2pz":
                norm = 1 / (4 * sq2pi) * z32;
                ret = norm * Z * z * Math.exp(-sigma / 2);
                break;
            case "2py":
                norm = 1 / (4 * sq2pi) * z32;
                ret = norm * Z * y * Math.exp(-sigma / 2);
                break;
            case "2px":
                norm = 1 / (4 * sq2pi) * z32;
                ret = norm * Z * x * Math.exp(-sigma / 2);
                break;
            case "3s":
                norm = 1 / (81 * sq3pi) * z32;
                prefac = (27 - 18 * sigma + 2 * Math.pow(sigma, 2));
                ret = norm * prefac * Math.exp(-sigma / 3);
                break;
            case "3pz":
                norm = Math.sqrt(2) / (81 * sqpi) * z32;
                prefac = Z * (6 - sigma);
                ret = norm * prefac * z * Math.exp(-sigma / 3);
                break;
            case "3py":
                norm = Math.sqrt(2) / (81 * sqpi) * z32;
                prefac = Z * (6 - sigma);
                ret = norm * prefac * y * Math.exp(-sigma / 3);
                break;
            case "3px":
                norm = Math.sqrt(2) / (81 * sqpi) * z32;
                prefac = Z * (6 - sigma);
                ret = norm * prefac * x * Math.exp(-sigma / 3);
                break;
            case "3d0":
                var ynorm = 1 / 4 * Math.sqrt(5 / Math.PI);
                var ybody = (-x2 -y2 + 2 * z2) / r2;
                ret = ynorm * ybody * rnorm * rbody;
                break;
            case "3d+1":
                var ynorm = 1 / 2 * Math.sqrt(15 / Math.PI);
                var ybody = z * x / r2;
                ret = ynorm * ybody * rnorm * rbody;
                break;
            case "3d+2":
                var ynorm = 1 / 4 * Math.sqrt(15 / Math.PI);
                var ybody = (x2 - y2) / r2;
                ret = ynorm * ybody * rnorm * rbody;
                break;
            case "3d-1":
                var ynorm = 1 / 2 * Math.sqrt(15 / Math.PI);
                var ybody = y * z / r2;
                ret = ynorm * ybody * rnorm * rbody;
                break;
            case "3d-2":
                rbody = z32 * Math.pow(sigma, 2) * Math.exp(-sigma / 3);
                var ynorm = 1 / 4 * Math.sqrt(15 / Math.PI);
                var ybody = x * y / r2;
                ret = ynorm * ybody * rnorm * rbody;
                break;
        };
        return ret;
    };
    return scalar_field(dims, func);
};


var SolidHarmonic = function(dimensions, l, m) {
    var sh = solid_harmonics[l][m];
    var func = function(x, y, z) {
        var r2 = x * x + y * y + z * z;
        return sh(x, y, z) * Math.exp(-Math.sqrt(r2));
    };
    return scalar_field(dimensions, func);
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
var r2 = function(r) { return Math.pow(r, 2) };
var r4 = function(r) { return Math.pow(r, 4) };
var r6 = function(r) { return Math.pow(r, 6) };


/*
Real cartesian solid harmonic type functions
These were computed using the python side:
exatomic.algorithms.basis.solid_harmonics
*/
var solid_harmonics = {
    0: { "0": function(x, y, z) {return 1},},
    1: {"-1": function(x, y, z) {return y},
         "0": function(x, y, z) {return z},
         "1": function(x, y , z) {return x}},
    2: {"-2": function(x, y, z) {return t3*x*y},
        "-1": function(x, y, z) {return t3*y*z},
         "0": function(x, y, z) {return -x*x/2 - y*y/2 + z*z},
         "1": function(x, y, z) {return t3*x*z},
         "2": function(x, y, z) {return t3/2*(x*x - y*y)}},
    3: {"-3": function(x, y, z) {return t10/4*y*(3*x*x - y*y)},
        "-2": function(x, y, z) {return t15*x*y*z},
        "-1": function(x, y, z) {return t6/4*y*(-x*x - y*y + 4*z*z)},
         "0": function(x, y, z) {return z/2*(-3*x*x - 3*y*y + 2*z*z)},
         "1": function(x, y, z) {return t6/4*x*(-x*x -y*y + 4*z*z)},
         "2": function(x, y, z) {return t15/2*z*(x*x - y*y)},
         "3": function(x, y, z) {return t10/4*x*(x*x - 3*y*y)}},
    4: {"-4": function(x, y, z) {return t35/2*x*y*(x*x -y*y)},
        "-3": function(x, y, z) {return t70/4*y*z*(3*x*x - y*y)},
        "-2": function(x, y, z) {return t5/2*x*y*(-x*x - y*y +6*z*z)},
        "-1": function(x, y, z) {return t10/4*y*z*(-3*x*x - 3*y*y + 4*z*z)},
         "0": function(x, y, z) {return 3*r4(x)/8 + 3*x*x*y*y/4 - 3*x*x*z*z + 3*r4(y)/8 - 3*y*y*z*z + r4(z)},
         "1": function(x, y, z) {return t10/4*x*z*(-3*x*x - 3*y*y + 4*z*z)},
         "2": function(x, y, z) {return t5/4*(x*x - y*y)*(-x*x - y*y + 6*z*z)},
         "3": function(x, y, z) {return t70/4*x*z*(x*x - 3*y*y)},
         "4": function(x, y, z) {return t35/8*(r4(x) - 6*x*x*y*y + r4(y))}},
    5: {"-5": function(x, y, z) {
            return 3/16*t14*y*(5*r4(x) - 10*x*x*y*y + r4(y))},
        "-4": function(x, y, z) {
            return 3/2*t35*x*y*z*(x*x - y*y)},
        "-3": function(x, y, z) {
            return t70/16*y*(3*x*x - y*y)*(-x*x - y*y + 8*z*z)},
        "-2": function(x, y, z) {
            return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z)},
        "-1": function(x, y, z) {
            return t15/8*y*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z))},
         "0": function(x, y, z) {
            return z/8*(15*r4(x) + 30*x*x*y*y - 40*x*x*z*z + 15*r4(y) - 40*y*y*z*z + 8*r4(z))},
         "1": function(x, y, z) {
            return t15/8*x*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z))},
         "2": function(x, y, z) {
            return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z)},
         "3": function(x, y, z) {
            return t70/16*x*(9*z*z*(x*x - 3*y*y) + (-x*x + 3*y*y)*(x*x + y*y + z*z))},
         "4": function(x, y, z) {
            return 3/8*t35*z*(r4(x) - 6*x*x*y*y + r4(y))},
         "5": function(x, y, z) {
            return 3/16*t14*x*(r4(x) - 10*x*x*y*y + 5*r4(y))}},
    6: {"-6": function(x, y, z) {
            return t462/16*x*y*(3*r4(x) -10*x*x*y*y + 3*r4(y))},
        "-5": function(x, y, z) {
            return 3*t154*y*z*(5*r4(x) - 10*x*x*y*y + r4(y))/16},
        "-4": function(x, y, z) {
            return 3*t7*x*y*(x*x - y*y)*(-x*x - y*y + 10*z*z)/4},
        "-3": function(x, y, z) {
            return t210/16*y*z*(3*x*x - y*y)*(-3*x*x - 3*y*y + 8*z*z)},
        "-2": function(x, y, z) {
            return t210*x*y*(r4(x) + 2*x*x*y*y - 16*x*x*z*z + r4(y) - 16*y*y*z*z + 16*r4(z))/16},
        "-1": function(x, y, z) {
            return t21*y*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8},
         "0": function(x, y, z) {
            return -5*r6(x)/16 - 15*r4(x)*y*y/16 + 45*r4(x)*z*z/8 - 15*x*x*r4(y)/16 + 45*x*x*y*y*z*z/4 - 15*x*x*r4(z)/2 - 5*r6(y)/16 + 45*r4(y)*z*z/8 - 15*y*y*r4(z)/2 + r6(z)},
         "1": function(x, y, z) {
            return t21*x*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8},
         "2": function(x, y, z) {
            return -t210/32*(x*x - y*y)*(11*z*z*(x*x + y*y - 2*z*z) - (x*x + y*y - 6*z*z)*(x*x + y*y + z*z))},
         "3": function(x, y, z) {
            return t210/16*x*z*(-x*x + 3*y*y)*(3*x*x + 3*y*y - 8*z*z)},
         "4": function(x, y, z) {
            return 3*t7/16*(-r6(x) + 5*r4(x)*y*y + 10*r4(x)*z*z + 5*x*x*r4(y) - 60*x*x*y*y*z*z - r6(y) + 10*r4(y)*z*z)},
         "5": function(x, y, z) {
            return 3/16*t154*x*z*(r4(x) - 10*x*x*y*y + 5*r4(y))},
         "6": function(x, y, z) {
            return t462/32*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y))}},
    7: {"-7": function(x, y, z) {
            return t429/32*y*(7*r6(x) - 35*r4(x)*y*y + 21*x*x*r4(y) - r6(y))},
        "-6": function(x, y, z) {
            return t6006/16*x*y*z*(3*r4(x) - 10*x*x*y*y + 3*r4(y))},
        "-5": function(x, y, z) {
            return t231/32*y*(-x*x - y*y + 12*z*z)*(-x*x*(-x*x + 3*y*y) + 4*x*x*(x*x - y*y) - y*y*(3*x*x - y*y))},
        "-4": function(x, y, z) {
            return t231/4*x*y*z*(x*x - y*y)*(-3*x*x - 3*y*y + 10*z*z)},
        "-3": function(x, y, z) {
            return t21/32*y*(3*x*x - y*y)*(z*z*(-39*x*x - 39*y*y + 104*z*z) - 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z))},
        "-2": function(x, y, z) {
            return t42/16*x*y*z*(15*r4(x) + 30*x*x*y*y - 80*x*x*z*z + 15*r4(y) - 80*y*y*z*z + 48*r4(z))},
        "-1": function(x, y, z) {
            return t7/32*y*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z))},
         "0": function(x, y, z) {
            return z/16*(-35*r6(x) - 105*r4(x)*y*y + 210*r4(x)*z*z - 105*x*x*r4(y) + 420*x*x*y*y*z*z - 168*x*x*r4(z) - 35*r6(y) + 210*r4(y)*z*z - 168*y*y*r4(z) + 16*r6(z))},
         "1": function(x, y, z) {
            return t7/32*x*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z))},
         "2": function(x, y, z) {
            return t42/32*z*(15*r6(x) + 15*r4(x)*y*y - 80*r4(x)*z*z - 15*x*x*r4(y) + 48*x*x*r4(z) - 15*r6(y) + 80*r4(y)*z*z - 48*y*y*r4(z))},
         "3": function(x, y, z) {
            return t21/32*x*(-x*x + 3*y*y)*(z*z*(39*x*x + 39*y*y - 104*z*z) + 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z))},
         "4": function(x, y, z) {
            return t231/16*z*(-3*r6(x) + 15*r4(x)*y*y + 10*r4(x)*z*z + 15*x*x*r4(y) - 60*x*x*y*y*z*z - 3*r6(y) + 10*r4(y)*z*z)},
         "5": function(x, y, z) {
            return t231/32*x*(-r6(x) + 9*r4(x)*y*y + 12*r4(x)*z*z + 5*x*x*r4(y) - 120*x*x*y*y*z*z - 5*r6(y) + 60*r4(y)*z*z)},
         "6": function(x, y, z) {
            return t6006/32*z*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y))},
         "7": function(x, y, z) {
            return t429/32*x*(r6(x) - 21*r4(x)*y*y + 35*x*x*r4(y) -7*r6(y))}}};


module.exports = {
    create_float_array_xyz: create_float_array_xyz,
    normalize_gaussian: normalize_gaussian,
    gen_field_arrays: gen_field_arrays,
    compute_field: compute_field,
    repeat_float: repeat_float,
    repeat_obj: repeat_obj,
    factorial2: factorial2,
    linspace: linspace,
    mapper: mapper,
    scalar_field: scalar_field,
    SolidHarmonic: SolidHarmonic,
    Hydrogenic: Hydrogenic,
    Gaussian: Gaussian,
    Ellipsoid: Ellipsoid,
    Sphere: Sphere,
    Torus: Torus
};
