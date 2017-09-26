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

var sphere = function(x, y, z) {
    return (x * x + y * y + z * z);
}

var ellipsoid = function(x, y, z, a, b, c) {
    a = (a === undefined) ? 2.0 : a;
    b = (b === undefined) ? 1.5 : b;
    c = (c === undefined) ? 1.0 : c;
    return 2 * ((x*x)/(a*a) + (y*y)/(b*b) + (z*z)/(c*c));
};

var torus = function(x, y, z, c) {
    c = (c === undefined) ? 2.5 : c;
    return  2 * (Math.pow(c - Math.sqrt(x*x + y*y), 2) + z*z);
}

var compute_field = function(xs, ys, zs, n, func) {
    var values = new Float32Array(n);
    var dv = (xs[1]-xs[0]) * (ys[1]-ys[0]) * (zs[1]-zs[0]);
    var i = 0;
    for (var x of xs) {
        for (var y of ys) {
            for (var z of zs) {
                var tmp = func(x, y, z);
                values[i] = tmp;
                i += 1;
            };
        };
    };
    return values
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

var contour = function(data, dims, orig, scale, val, ncont, contlims, axis) {

    var compute_contours = function(num_contours, cont_limits) {
        // Determine the spacing for contour lines given the limits
        // (exponents of base ten) and the number of contour lines
        var clevels = [];
        var d = (cont_limits[1] - cont_limits[0]) / num_contours;
        var tmp1 = [];
        var tmp2 = [];
        for(var i = 0; i < num_contours; i++) {
            tmp1.push(-Math.pow(10, -i * d));
            tmp2.push(Math.pow(10, -i * d));
        };
        tmp2.reverse()
        for(var i = 0; i < tmp1.length; i++) {
            clevels.push(tmp1[i]);
        };
        for(var i = 0; i < tmp2.length; i++) {
            clevels.push(tmp2[i]);
        };
        console.log(clevels);
        return clevels;
    };

    var get_square = function(data, dims, orig, scale, val, axis) {
        // Given a 1D array data that is ordered by x (outer increment)
        // then y (middle increment) and z (inner increment), determine
        // the appropriate z index given the z value to plot the contour
        // over. Then select that z-axis of the cube data and convert
        // it to a 2D array for processing in marching squares.
        var x = [];
        var y = [];
        var dat = [];
        // Default to the "z" axis
        var xidx = 0;
        var yidx = 1;
        var plidx = 2;

        if (axis == "x") {
            xidx = 0;
            yidx = 2;
            plidx = 1;
        } else if (axis == "y") {
            xidx = 1;
            yidx = 2;
            plidx = 0;
        };

        for(var i = 0; i < dims[xidx]; i++) {
            x.push(orig[xidx] + i * scale[xidx]);
        };
        for(var j = 0; j < dims[yidx]; j++) {
            y.push(orig[yidx] + j * scale[yidx]);
        };

        var idx = 0;
        var cur = 0;
        var first = orig[plidx];
        var amax = orig[plidx] + dims[plidx] * scale[plidx];

        if (val < first) {
            idx = 0;
        } else if (val > amax) {
            idx = dims[plidx] - 1;
        } else {
            for(var k = 0; k < dims[plidx]; k++) {
                cur = orig[plidx] + k * scale[plidx];
                if (Math.abs(val - cur) < Math.abs(val - first)) {
                    first = cur;
                    idx = k;
                };
            };
        };

        if (axis == "z") {
            for(var nx = 0; nx < dims[0]; nx++) {
                var tmp = [];
                for(var ny = 0; ny < dims[1]; ny++) {
                    tmp.push(data[(dims[0] * nx + ny) * dims[1] + idx]);
                };
                dat.push(tmp);
            };
        } else if (axis == "x") {
            for(var nx = 0; nx < dims[0]; nx++) {
                var tmp = [];
                for(var nz = 0; nz < dims[2]; nz++) {
                    tmp.push(data[(dims[0] * nx + idx) * dims[1] + nz]);
                };
                dat.push(tmp);
            };
        } else if (axis == "y") {
            for(var ny = 0; ny < dims[1]; ny++) {
                var tmp = [];
                for(var nz = 0; nz < dims[2]; nz++) {
                    tmp.push(data[(dims[0] * idx + ny) * dims[1] + nz]);
                };
                dat.push(tmp);
            };
        };
        return {"dat": dat, "x": x, "y": y};
    };

    // Interpolate a value along the side of a square
    // by the relative heights at the square of interest
    var sect = function(p1, p2, rh, w) {
        return (rh[p2] * w[p1] - rh[p1] * w[p2]) / (rh[p2] - rh[p1]);
    };

    // Get the contour values given the limits and number of lines
    var z = compute_contours(ncont, contlims);
    var nc = z.length;

    // Get square of interest and x, y data
    var dat = get_square(data, dims, orig, scale, val, axis);
    var sq = dat.dat;
    var x = dat.x;
    var y = dat.y;

    // Relative indices of adjacent square vertices
    var im = [0, 1, 1, 0];
    var jm = [0, 0, 1, 1];
    // Indexes case values for case value switch
    var castab = [[[0, 0, 8], [0, 2, 5], [7, 6, 9]],
                  [[0, 3, 4], [1, 3, 1], [4, 3, 0]],
                  [[9, 6, 7], [5, 2, 0], [8, 0, 0]]];

    // Probably don't need as many empty arrays as number of contours
    // but need at least two for positive and negative contours.
    // exa.threejs.app.add_contours currently expects this behavior, however.
    var cverts = [];
    for(var i = 0; i < nc; i++) {
      cverts.push([]);
    };

    // Vars to hold the relative heights and x, y values of
    // the square of interest.
    var h = new Array(5);
    var xh = new Array(5);
    var yh = new Array(5);
    var sh = new Array(5);

    // m indexes the triangles in the square of interest
    var m1;
    var m2;
    var m3;
    // case value tells us whether or not to interpolate
    // or which x and y values to use to build the contour
    var cv;
    // Place holders for the x and y values of the square of interest
    var x1 = 0;
    var x2 = 0;
    var y1 = 0;
    var y2 = 0;
    // Find out whether or not the x, y range is relevant for
    // the given contour level
    var tmp1 = 0;
    var tmp2 = 0;
    var dmin;
    var dmax;

    // Starting at the end of the y array
    for(var j = dims[1] - 2; j >= 0; j--) {
        // But at the beginning of the x array
        for(var i = 0; i <= dims[0] - 2; i++) {
            // The smallest and largest values in square of interest
            tmp1 = Math.min(sq[i][j], sq[i][j + 1]);
            tmp2 = Math.min(sq[i + 1][j], sq[i + 1][j + 1]);
            dmin = Math.min(tmp1, tmp2);
            tmp1 = Math.max(sq[i][j], sq[i][j + 1]);
            tmp2 = Math.max(sq[i + 1][j], sq[i + 1][j + 1]);
            dmax = Math.max(tmp1, tmp2);
            // If outside all contour bounds, move along
            if (dmax < z[0] || dmin > z[nc - 1]) {
                continue;
            };
            // For each contour
            for(var c = 0; c < nc; c++) {
                // If outside individual contour bound, move along
                if (z[c] < dmin || z[c] > dmax) {
                    continue;
                };
                // The box is considered as follows:
                //
                //  v4  +------------------+ v3
                //      |        m=3       |
                //      |                  |
                //      |  m=2    X   m=2  |  center is v0
                //      |                  |
                //      |        m=1       |
                //  v1  +------------------+ v2
                //
                // m indexes the triangle
                // For each vertex in the square (5 of them)
                for(var m = 4; m >= 0; m--) {
                    if (m > 0) {
                        h[m] = sq[i + im[m-1]][j + jm[m-1]] - z[c];
                        xh[m] = x[i + im[m-1]];
                        yh[m] = y[j + jm[m-1]];
                    } else {
                        h[0] = 0.25 * (h[1] + h[2] + h[3] + h[4]);
                        xh[0] = 0.5 * (x[i] + x[i + 1]);
                        yh[0] = 0.5 * (y[j] + y[j + 1]);
                    };
                    if (h[m] > 0.0) {
                        sh[m] = 1;
                    } else if (h[m] < 0.0) {
                        sh[m] = -1;
                    } else {
                        sh[m] = 0;
                    };
                };
                // Now loop over the triangles in the square to find the case
                for(var m = 1; m <= 4; m++) {
                    m1 = m;
                    m2 = 0;
                    if (m != 4) {
                      m3 = m + 1;
                    } else {
                      m3 = 1;
                    };
                    if ((cv = castab[sh[m1] + 1][sh[m2] + 1][sh[m3] + 1]) == 0) {
                        continue;
                    };
                    // Assign x and y appropriately given the case
                    switch (cv) {
                        case 1: // Vertices 1 and 2
                            x1 = xh[m1];
                            y1 = yh[m1];
                            x2 = xh[m2];
                            y2 = yh[m2];
                            break;
                        case 2: // Vertices 2 and 3
                            x1 = xh[m2];
                            y1 = yh[m2];
                            x2 = xh[m3];
                            y2 = yh[m3];
                            break;
                        case 3: // Vertices 3 and 1
                            x1 = xh[m3];
                            y1 = yh[m3];
                            x2 = xh[m1];
                            y2 = yh[m1];
                            break;
                        case 4: // Vertex 1 and side 2-3
                            x1 = xh[m1];
                            y1 = yh[m1];
                            x2 = sect(m2, m3, h, xh);
                            y2 = sect(m2, m3, h, yh);
                            break;
                        case 5: // Vertex 2 and side 3-1
                            x1 = xh[m2];
                            y1 = yh[m2];
                            x2 = sect(m3, m1, h, xh);
                            y2 = sect(m3, m1, h, yh);
                            break;
                        case 6: // Vertex 3 and side 1-2
                            x1 = xh[m3];
                            y1 = yh[m3];
                            x2 = sect(m1, m2, h, xh);
                            y2 = sect(m1, m2, h, yh);
                            break;
                        case 7: // Sides 1-2 and 2-3
                            x1 = sect(m1, m2, h, xh);
                            y1 = sect(m1, m2, h, yh);
                            x2 = sect(m2, m3, h, xh);
                            y2 = sect(m2, m3, h, yh);
                            break;
                        case 8: // Sides 2-3 and 3-1
                            x1 = sect(m2, m3, h, xh);
                            y1 = sect(m2, m3, h, yh);
                            x2 = sect(m3, m1, h, xh);
                            y2 = sect(m3, m1, h, yh);
                            break;
                        case 9: // Sides 3-1 and 1-2
                            x1 = sect(m3, m1, h, xh);
                            y1 = sect(m3, m1, h, yh);
                            x2 = sect(m1, m2, h, xh);
                            y2 = sect(m1, m2, h, yh);
                            break;
                        default:
                            break;
                    };
                    // Push the x and y positions
                    cverts[c].push([x1, y1, val]);
                    cverts[c].push([x2, y2, val]);
                };
            };
        };
    };
    // Default behavior of algorithm
    if (axis == "z") {
        return cverts;
    // cverts is populated assuming x, y correspond to x, y
    // dimensions in the cube file, but to avoid a double nested
    // if else inside the tight O(N^2) loop, just move around the
    // elements accordingly afterwards according to the axis of
    // interest
    } else if (axis == "x") {
        var rear = [];
        for(var c = 0; c < nc; c++) {
            rear.push([]);
            var maxi = cverts[c].length;
            for(var v = 0; v < maxi; v++) {
                rear[c].push([cverts[c][v][0], val, cverts[c][v][1]]);
            };
        };
    } else if (axis == "y") {
        var rear = [];
        for(var c = 0; c < nc; c++) {
            rear.push([]);
            var maxi = cverts[c].length;
            for(var v = 0; v < maxi; v++) {
                rear[c].push([val, cverts[c][v][0], cverts[c][v][1]]);
            };
        };
    };
    return rear;
};

// in jupyter-exatomic-three.js marching_cubes
// the following field attributes are used:
// field.nx, field.ny, field.nz, field.x, field.y, field.z, field.values
var scalar_field = function(dims, func_or_values) {
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
    console.log(dims.hasOwnProperty("x"),
                dims.hasOwnProperty("y"),
                dims.hasOwnProperty("z"));
    var x = dims.hasOwnProperty("x") ? dims.x : linspace(dims.ox, dims.fxi, dims.nx);
    var nx = x.length;
    var y = dims.hasOwnProperty("y") ? dims.y : linspace(dims.oy, dims.fyj, dims.ny);
    var ny = y.length;
    var z = dims.hasOwnProperty("z") ? dims.z : linspace(dims.oz, dims.fzk, dims.nz);
    var nz = z.length;
    var n = nx * ny * nz;
    if (typeof func_or_values === "function") {
        var values = compute_field(x, y, z, n, func_or_values);
    } else {
        var values = new Float32Array(func_or_values);
    };
    return { "x": x,   "y": y,   "z": z, 
            "nx": nx, "ny": ny, "nz": nz,
            "values": values}
};


var Gaussian = function(dimensions, which) {
//    var alpha = 0.01;
//    var lmap = {"s": 0, "p": 1, "d": 2, "f": 3};
//    var norm = normalize_gaussian(alpha, lmap[which[0]]);
//    var xmap = {"x": 0, "y": 0, "z": 0};
//    if (lmap[which[0]] == 1) {
//        xmap[which[1]] = 1
//    } else if (lmap[which[0]] > 1) {
//        xmap["x"] = which[1];
//        xmap["y"] = which[2];
//        xmap["z"] = which[3];
//    };
//    console.log('dimensions and which');
//    console.log(dimensions);
//    console.log(which);
//    console.log('l and norm');
//    console.log(lmap, which[0]);
//    console.log(lmap[which[0]]);
//    console.log(lmap[which[0]], norm, xmap);
//    var func = function(x, y, z) {
//        var x2 = x*x;
//        var y2 = y*y;
//        var z2 = z*z;
//        var r2 = x2 + y2 + z2;
//        //console.log(xmap);
//        return (norm * Math.pow(x, xmap["x"]) * Math.pow(y, xmap["y"]),
//                       Math.pow(z, xmap["z"]) * Math.exp(-alpha * r2));
//    };
    return scalar_field(dimensions, primitives[which]);
};


var primitives = {
    "s": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 0);
        return norm * Math.exp(-alpha * r2);
    },
    "px": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 1);
        return norm * x * Math.exp(-alpha * r2);
    },

    "py": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 1);
        return norm * y * Math.exp(-alpha * r2);
    },

    "pz": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 1);
        return norm * z * Math.exp(-alpha * r2);
    },

    "d200": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * x2 * Math.exp(-alpha * r2);
    },

    "d110": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * x * y * Math.exp(-alpha * r2);
    },

    "d101": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * x * z * Math.exp(-alpha * r2);
    },


    "d020": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * y2 * Math.exp(-alpha * r2);
    },

    "d011": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * y * z * Math.exp(-alpha * r2);
    },

    "d002": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 2);
        return norm * z2 * Math.exp(-alpha * r2);
    },

    "f300": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*x*x * Math.exp(-alpha * r2);
    },

    "f210": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*x*y * Math.exp(-alpha * r2);
    },

    "f201": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*x*z * Math.exp(-alpha * r2);
    },

    "f120": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*y*y * Math.exp(-alpha * r2);
    },

    "f102": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*z*z * Math.exp(-alpha * r2);
    },

    "f111": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * x*y*z * Math.exp(-alpha * r2);
    },

    "f030": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * y*y*y * Math.exp(-alpha * r2);
    },

    "f021": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * y*y*z * Math.exp(-alpha * r2);
    },

    "f012": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * y*z*z * Math.exp(-alpha * r2);
    },

    "f003": function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var alpha = 0.01;
        var norm = normalize_gaussian(alpha, 3);
        return norm * z*z*z * Math.exp(-alpha * r2);
    },

};

var Hydrogenic = function(dimensions, which) {
    return scalar_field(dimensions, hydrogen[which]);
};

var SolidHarmonic = function(dimensions, l, m) {
    var sh = solid_harmonics[l][m];
    var func = function(x, y, z) {
        var r2 = x*x + y*y + z*z;
        return sh(x, y, z) * Math.exp(-Math.sqrt(r2));
    };
    return scalar_field(dimensions, func);
};

/*
Hydrogenic atomic orbital type functions
*/
var hydrogen = {
    "1s": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        return 1 / Math.sqrt(Math.PI) * Math.pow(Z, 3/2) * Math.exp(-sigma);
    },

    "2s": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = 1 / (4 * Math.sqrt(2 * Math.PI)) * Math.pow(Z, 3/2)
        return norm * (2 - sigma) * Math.exp(-sigma / 2);
    },

    "2pz": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = 1 / (4 * Math.sqrt(2 * Math.PI)) * Math.pow(Z, 3/2)
        return norm * Z * z * Math.exp(-sigma / 2);
    },

    "2px": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = 1 / (4 * Math.sqrt(2 * Math.PI)) * Math.pow(Z, 3/2)
        return norm * Z * x * Math.exp(-sigma / 2);
    },

    "2py": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = 1 / (4 * Math.sqrt(2 * Math.PI)) * Math.pow(Z, 3/2)
        return norm * Z * y * Math.exp(-sigma / 2);
    },

    "3s": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = 1 / (81 * Math.sqrt(3 * Math.PI)) * Math.pow(Z, 3/2)
        var prefac = (27 - 18 * sigma + 2 * Math.pow(sigma, 2))
        return norm * prefac * Math.exp(-sigma / 3);
    },

    "3pz": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = Math.sqrt(2) / (81 * Math.sqrt(Math.PI)) * Math.pow(Z, 3/2)
        var prefac = Z * (6 - sigma)
        return norm * prefac * z * Math.exp(-sigma / 3);
    },

    "3py": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = Math.sqrt(2) / (81 * Math.sqrt(Math.PI)) * Math.pow(Z, 3/2)
        var prefac = Z * (6 - sigma)
        return norm * prefac * y * Math.exp(-sigma / 3);
    },

    "3px": function(x, y, z) {
        var r = Math.sqrt(x * x + y * y + z * z);
        var Z = 1;
        var sigma = Z * r
        var norm = Math.sqrt(2) / (81 * Math.sqrt(Math.PI)) * Math.pow(Z, 3/2)
        var prefac = Z * (6 - sigma)
        return norm * prefac * x * Math.exp(-sigma / 3);
    },

    "3d0": function(x, y, z) {
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var Z = 1;
        var sigma = Z * r;
        var rnorm = 1 / Math.sqrt(2430);
        var rbody = Math.pow(Z, 3 / 2) * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        var ynorm = 1 / 4 * Math.sqrt(5 / Math.PI);
        var ybody = (-x2 -y2 + 2 * z2) / r2;
        return ynorm * ybody * rnorm * rbody;
    },

    "3d+1": function(x, y, z) {
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var Z = 1;
        var sigma = Z * r;
        var rnorm = 1 / Math.sqrt(2430);
        var rbody = Math.pow(Z, 3 / 2) * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        var ynorm = 1 / 2 * Math.sqrt(15 / Math.PI);
        var ybody = z * x / r2;
        return ynorm * ybody * rnorm * rbody;
    },

    "3d-1": function(x, y, z) {
        var sigma = Z * r;
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var Z = 1;
        var sigma = Z * r;
        var rnorm = 1 / Math.sqrt(2430);
        var rbody = Math.pow(Z, 3 / 2) * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        var ynorm = 1 / 2 * Math.sqrt(15 / Math.PI);
        var ybody = y * z / r2;
        return rnorm * rbody * ynorm * ybody;
    },

    "3d+2": function(x, y, z) {
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var Z = 1;
        var sigma = Z * r;
        var rnorm = 1 / Math.sqrt(2430);
        var rbody = Math.pow(Z, 3 / 2) * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        var ynorm = 1 / 4 * Math.sqrt(15 / Math.PI);
        var ybody = (x2 - y2) / r2;
        return rnorm * rbody * ynorm * ybody;
    },

    "3d-2": function(x, y, z) {
        var x2 = x * x;
        var y2 = y * y;
        var z2 = z * z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        var Z = 1;
        var sigma = Z * r;
        var rnorm = 1 / Math.sqrt(2430);
        //var rbody = Math.pow(Z, 3 / 2) * Math.pow(2 / 3 * sigma, 2) * Math.exp(-sigma / 3);
        var rbody = Math.pow(Z, 3 / 2) * Math.pow(sigma, 2) * Math.exp(-sigma / 3);
        var ynorm = 1 / 4 * Math.sqrt(15 / Math.PI);
        //var ynorm = 1 / 2 * Math.sqrt(15 / Math.PI);
        var ybody = x * y / r2;
        return rnorm * rbody * ynorm * ybody;
    },
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
var r4 = function(r) {return r*r*r*r};
var r6 = function(r) {return r*r*r*r*r*r};


/*
Real cartesian solid harmonic type functions
These were computed using the python side:
exatomic.algorithms.basis.solid_harmonics
*/
var solid_harmonics = {
    "0": {"0": function(x, y, z) {return 1},},

    "1": {"-1": function(x, y, z) {return y},
          "0": function(x, y, z) {return z},
          "1": function(x, y , z) {return x}},

    "2": {
        "-2": function(x, y, z) {
            return t3*x*y;
        },
        "-1": function(x, y, z) {
            return t3*y*z;
        },
        "0": function(x, y, z) {
            return -x*x/2 - y*y/2 + z*z;
        },
        "1": function(x, y, z) {
            return t3*x*z;
        },
        "2": function(x, y, z) {
            return t3/2*(x*x - y*y);
        }
    },

    "3": {
        "-3": function(x, y, z) {
            return t10/4*y*(3*x*x - y*y);
        },
        "-2": function(x, y, z) {
            return t15*x*y*z;
        },
        "-1": function(x, y, z) {
            return t6/4*y*(-x*x - y*y + 4*z*z);
        },
        "0": function(x, y, z) {
            return z/2*(-3*x*x - 3*y*y + 2*z*z);
        },
        "1": function(x, y, z) {
            return t6/4*x*(-x*x -y*y + 4*z*z);
        },
        "2": function(x, y, z) {
            return t15/2*z*(x*x - y*y);
        },
        "3": function(x, y, z) {
            return t10/4*x*(x*x - 3*y*y);
        }
    },

    "4": {
        "-4": function(x, y, z) {
            return t35/2*x*y*(x*x -y*y);
        },
        "-3": function(x, y, z) {
            return t70/4*y*z*(3*x*x - y*y);
        },
        "-2": function(x, y, z) {
            return t5/2*x*y*(-x*x - y*y +6*z*z);
        },
        "-1": function(x, y, z) {
            return t10/4*y*z*(-3*x*x - 3*y*y + 4*z*z);
        },
        "0": function(x, y, z) {
            return 3*r4(x)/8 + 3*x*x*y*y/4 - 3*x*x*z*z + 3*r4(y)/8 - 3*y*y*z*z + r4(z);
        },
        "1": function(x, y, z) {
            return t10/4*x*z*(-3*x*x - 3*y*y + 4*z*z);
        },
        "2": function(x, y, z) {
            return t5/4*(x*x - y*y)*(-x*x - y*y + 6*z*z);
        },
        "3": function(x, y, z) {
            return t70/4*x*z*(x*x - 3*y*y);
        },
        "4": function(x, y, z) {
            return t35/8*(r4(x) - 6*x*x*y*y + r4(y));
        }
    },

    "5": {
        "-5": function(x, y, z) {
            return 3/16*t14*y*(5*r4(x) - 10*x*x*y*y + r4(y));
        },
        "-4": function(x, y, z) {
            return 3/2*t35*x*y*z*(x*x - y*y);
        },
        "-3": function(x, y, z) {
            return t70/16*y*(3*x*x - y*y)*(-x*x - y*y + 8*z*z);
        },
        "-2": function(x, y, z) {
            return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z);
        },
        "-1": function(x, y, z) {
            return t15/8*y*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z));
        },
        "0": function(x, y, z) {
            return z/8*(15*r4(x) + 30*x*x*y*y - 40*x*x*z*z + 15*r4(y) - 40*y*y*z*z + 8*r4(z));
        },
        "1": function(x, y, z) {
            return t15/8*x*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z));
        },
        "2": function(x, y, z) {
            return t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z);
        },
        "3": function(x, y, z) {
            return t70/16*x*(9*z*z*(x*x - 3*y*y) + (-x*x + 3*y*y)*(x*x + y*y + z*z));
        },
        "4": function(x, y, z) {
            return 3/8*t35*z*(r4(x) - 6*x*x*y*y + r4(y));
        },
        "5": function(x, y, z) {
            return 3/16*t14*x*(r4(x) - 10*x*x*y*y + 5*r4(y));
        }
    },

    "6": {
        "-6": function(x, y, z) {
            return t462/16*x*y*(3*r4(x) -10*x*x*y*y + 3*r4(y));
        },
        "-5": function(x, y, z) {
            return 3*t154*y*z*(5*r4(x) - 10*x*x*y*y + r4(y))/16;
        },
        "-4": function(x, y, z) {
            return 3*t7*x*y*(x*x - y*y)*(-x*x - y*y + 10*z*z)/4;
        },
        "-3": function(x, y, z) {
            return t210/16*y*z*(3*x*x - y*y)*(-3*x*x - 3*y*y + 8*z*z);
        },
        "-2": function(x, y, z) {
            return t210*x*y*(r4(x) + 2*x*x*y*y - 16*x*x*z*z + r4(y) - 16*y*y*z*z + 16*r4(z))/16;
        },
        "-1": function(x, y, z) {
            return t21*y*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8;
        },
        "0": function(x, y, z) {
            return -5*r6(x)/16 - 15*r4(x)*y*y/16 + 45*r4(x)*z*z/8 - 15*x*x*r4(y)/16 + 45*x*x*y*y*z*z/4 - 15*x*x*r4(z)/2 - 5*r6(y)/16 + 45*r4(y)*z*z/8 - 15*y*y*r4(z)/2 + r6(z);
        },
        "1": function(x, y, z) {
            return t21*x*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8;
        },
        "2": function(x, y, z) {
            return -t210/32*(x*x - y*y)*(11*z*z*(x*x + y*y - 2*z*z) - (x*x + y*y - 6*z*z)*(x*x + y*y + z*z));
        },
        "3": function(x, y, z) {
            return t210/16*x*z*(-x*x + 3*y*y)*(3*x*x + 3*y*y - 8*z*z);
        },
        "4": function(x, y, z) {
            return 3*t7/16*(-r6(x) + 5*r4(x)*y*y + 10*r4(x)*z*z + 5*x*x*r4(y) - 60*x*x*y*y*z*z - r6(y) + 10*r4(y)*z*z);
        },
        "5": function(x, y, z) {
            return 3/16*t154*x*z*(r4(x) - 10*x*x*y*y + 5*r4(y));
        },
        "6": function(x, y, z) {
            return t462/32*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y));
        }
    },

    7: {
        "-7": function(x, y, z) {
            return t429/32*y*(7*r6(x) - 35*r4(x)*y*y + 21*x*x*r4(y) - r6(y));
        },
        "-6": function(x, y, z) {
            return t6006/16*x*y*z*(3*r4(x) - 10*x*x*y*y + 3*r4(y));
        },
        "-5": function(x, y, z) {
            return t231/32*y*(-x*x - y*y + 12*z*z)*(-x*x*(-x*x + 3*y*y) + 4*x*x*(x*x - y*y) - y*y*(3*x*x - y*y));
        },
        "-4": function(x, y, z) {
            return t231/4*x*y*z*(x*x - y*y)*(-3*x*x - 3*y*y + 10*z*z);
        },
        "-3": function(x, y, z) {
            return t21/32*y*(3*x*x - y*y)*(z*z*(-39*x*x - 39*y*y + 104*z*z) - 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z));
        },
        "-2": function(x, y, z) {
            return t42/16*x*y*z*(15*r4(x) + 30*x*x*y*y - 80*x*x*z*z + 15*r4(y) - 80*y*y*z*z + 48*r4(z));
        },
        "-1": function(x, y, z) {
            return t7/32*y*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z));
        },
        "0": function(x, y, z) {
            return z/16*(-35*r6(x) - 105*r4(x)*y*y + 210*r4(x)*z*z - 105*x*x*r4(y) + 420*x*x*y*y*z*z - 168*x*x*r4(z) - 35*r6(y) + 210*r4(y)*z*z - 168*y*y*r4(z) + 16*r6(z));
        },
        "1": function(x, y, z) {
            return t7/32*x*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z));
        },
        "2": function(x, y, z) {
            return t42/32*z*(15*r6(x) + 15*r4(x)*y*y - 80*r4(x)*z*z - 15*x*x*r4(y) + 48*x*x*r4(z) - 15*r6(y) + 80*r4(y)*z*z - 48*y*y*r4(z));
        },
        "3": function(x, y, z) {
            return t21/32*x*(-x*x + 3*y*y)*(z*z*(39*x*x + 39*y*y - 104*z*z) + 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z));
        },
        "4": function(x, y, z) {
            return t231/16*z*(-3*r6(x) + 15*r4(x)*y*y + 10*r4(x)*z*z + 15*x*x*r4(y) - 60*x*x*y*y*z*z - 3*r6(y) + 10*r4(y)*z*z);
        },
        "5": function(x, y, z) {
            return t231/32*x*(-r6(x) + 9*r4(x)*y*y + 12*r4(x)*z*z + 5*x*x*r4(y) - 120*x*x*y*y*z*z - 5*r6(y) + 60*r4(y)*z*z);
        },
        "6": function(x, y, z) {
            return t6006/32*z*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y));
        },
        "7": function(x, y, z) {
            return t429/32*x*(r6(x) - 21*r4(x)*y*y + 35*x*x*r4(y) -7*r6(y));
        }
    }
};


module.exports = {
    create_float_array_xyz: create_float_array_xyz,
    gen_field_arrays: gen_field_arrays,
    compute_field: compute_field,
    repeat_float: repeat_float,
    repeat_obj: repeat_obj,
    factorial2: factorial2,
    linspace: linspace,
    //xrange: xrange,
    mapper: mapper,
    ellipsoid: ellipsoid,
    sphere: sphere,
    torus: torus,
    normalize_gaussian: normalize_gaussian,
    scalar_field: scalar_field,
    contour: contour,
    Gaussian: Gaussian,
    Hydrogenic: Hydrogenic,
    SolidHarmonic: SolidHarmonic 
};
