'use strict';

var Contour = function(data, dims, orig, scale, val, ncont, contlims, axis) {

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
        // Default to the 'z' axis
        var xidx = 0;
        var yidx = 1;
        var plidx = 2;

        if (axis == 'x') {
            xidx = 0;
            yidx = 2;
            plidx = 1;
        } else if (axis == 'y') {
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

        console.log(idx);

        if (axis == 'z') {
            for(var nx = 0; nx < dims[0]; nx++) {
                var tmp = [];
                for(var ny = 0; ny < dims[1]; ny++) {
                    tmp.push(data[(dims[0] * nx + ny) * dims[1] + idx]);
                };
                dat.push(tmp);
            };
        } else if (axis == 'x') {
            for(var nx = 0; nx < dims[0]; nx++) {
                var tmp = [];
                for(var nz = 0; nz < dims[2]; nz++) {
                    tmp.push(data[(dims[0] * nx + idx) * dims[1] + nz]);
                };
                dat.push(tmp);
            };
        } else if (axis == 'y') {
            for(var ny = 0; ny < dims[1]; ny++) {
                var tmp = [];
                for(var nz = 0; nz < dims[2]; nz++) {
                    tmp.push(data[(dims[0] * idx + ny) * dims[1] + nz]);
                };
                dat.push(tmp);
            };
        };
        return {'dat': dat, 'x': x, 'y': y};
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
    console.log(dat);
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
      console.log(Math.min(...sq[i]));
      console.log(Math.max(...sq[i]));
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
    if (axis == 'z') {
        return cverts;
    // cverts is populated assuming x, y correspond to x, y
    // dimensions in the cube file, but to avoid a double nested
    // if else inside the tight O(N^2) loop, just move around the
    // elements accordingly afterwards according to the axis of
    // interest
    } else if (axis == 'x') {
        var rear = [];
        for(var c = 0; c < nc; c++) {
            rear.push([]);
            var maxi = cverts[c].length;
            for(var v = 0; v < maxi; v++) {
                rear[c].push([cverts[c][v][0], val, cverts[c][v][1]]);
            };
        };
    } else if (axis == 'y') {
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
