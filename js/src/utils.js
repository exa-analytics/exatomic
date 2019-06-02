// Copright (c) 2015-2018, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
================
utils.js
================
Helper JS functions.
*/
"use strict";


let repeat_float = (value, n) => {
    let array = new Float32Array(n)
    for (let i=0; i<n; i++) {
        array[i] = value
    }
    return array
}


let create_float_array_xyz = (x, y, z) => {
    let nx = x.length || 1
    let ny = y.length || 1
    let nz = z.length || 1
    let n = Math.max(nx, ny, nz)
    x = (nx === 1) ? repeat_float(x, n) : x
    y = (ny === 1) ? repeat_float(y, n) : y
    z = (nz === 1) ? repeat_float(z, n) : z
    let xyz = new Float32Array(n * 3)
    for (let i=0, i3=0; i<n; i++, i3+=3) {
        xyz[i3  ] = x[i]
        xyz[i3+1] = y[i]
        xyz[i3+2] = z[i]
    }
    return xyz
}


let linspace = (min, max, n) => {
    let step = (max - min) / (n - 1)
    let array = new Float32Array(n)
    for (let i=0; i<n; i++) {
        array[i] = min + i * step
    }
    return array
    // let array = [min]
    // for (let i=0; i<n1; i++) {
    //     min += step
    //     array.push(min)
    // }
    // return new Float32Array(array)
}


var gen_field_arrays = function(fps) {
    return {x: linspace(fps.ox, fps.fx, fps.nx),
            y: linspace(fps.oy, fps.fy, fps.ny),
            z: linspace(fps.oz, fps.fz, fps.nz)};
};


var xrange = function(orig, delta, num) {
    console.log(orig, delta, num);
    var end = orig + (num - 1) * delta;
    return linspace(orig, end, num);
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
        mapped.push(map[indices[i]]);
    }
    return mapped;
};


var jsonparse = function(string) {
    return new Promise(function(resolve, reject) {
        try { resolve(JSON.parse(string)); }
        catch(e) { reject(e); }
    });
};


var logerror = function(e) {console.log(e.message);};


var fparse = function(obj, key) {
    jsonparse(obj.model.get(key))
    .then(function(p) {obj[key] = p}).catch(logerror);
};


var resolv = function(obj, key) {
    return Promise.resolve(obj["init_" + key]())
    .then(function(p) {obj[key] = p}).catch(logerror);
};


var mesolv = function(obj, key) {
    return Promise.resolve(obj.model.get(key))
    .then(function(p) {obj[key] = p}).catch(logerror);
};


let Sphere = (x, y, z) => {
    return (x*x + y*y + z*z)
}

let Ellipsoid = (x, y, z) => {
    return 2 * ((x*x)/4 + (y*y)/2.25 + (z*z))
}

let Torus = (x, y, z) => {
    return  2 * (Math.pow(2.5 - Math.sqrt(x*x + y*y), 2) + z*z)
}

let compute_field = (xs, ys, zs, n, func) => {
    let i = 0
    let vals = new Float32Array(n)
    for (let x of xs) {
        for (let y of ys) {
            for (let z of zs) {
                vals[i] = func(x, y, z)
                i += 1
            }
        }
    }
    return vals
}

let factorial2 = (n) => {
    if (n < -1) {
        return 0
    } else if (n < 2) {
        return 1
    } else {
        let prod = 1
        while (n > 0) {
            prod *= n
            n -= 2
        }
        return prod
    }
}


let scalar_field = (dims, funcvals) => {
    /*"""
    scalar_field
    ==============
    Args:
        dims: {"ox": ox, "nx": nx, "fx": fx,
               "oy": oy, "ny": ny, "fy": fy,
               "oz": oz, "nz": nz, "fz": fz}
        funcvals: function of (x, y, z) to be
                  evaluated or already evaluated
                  field values

    Note:
        The dimensions argument can alternatively be
        {"x": xarray, "y": yarray, "z": zarray}
        if they have already been constructed but in this
        case the arrays should form cubic discrete points
    */
    let nx = dims.nx
    let ny = dims.ny
    let nz = dims.nz
    let x = dims.x || linspace(dims.ox, dims.fx, nx)
    let y = dims.y || linspace(dims.oy, dims.fy, ny)
    let z = dims.z || linspace(dims.oz, dims.fz, nz)
    let n = nx * ny * nz
    let values
    if (typeof funcvals === "function") {
        values = compute_field(x, y, z, n, funcvals)
    } else {
        values = new Float32Array(funcvals)
    }
    return {
        "x": x, "y": y, "z": z,
        "nx": nx, "ny": ny, "nz": nz,
        "values": values
    }
}


let Gaussian = (which) => {
    let alpha = 1.0
    let lmap = {"s": 0, "p": 1, "d": 2, "f": 3}
    let xmap = {"x": 0, "y": 0, "z": 0}
    let L = lmap[which[0]]
    let c = which[1]
    let prefac = Math.pow((2 / Math.PI), 0.75)
    let numer = Math.pow(2, L) * Math.pow(alpha, ((L + 1.5) / 2))
    let denom = Math.pow(factorial2(2 * L - 1), 0.5)
    let norm = prefac * numer / denom
    if (L === 1) {
        xmap[c] = 1
    } else if (L > 1) {
        xmap["x"] = which[1]
        xmap["y"] = which[2]
        xmap["z"] = which[3]
    }
    let func = (x, y, z) => {
        let x2 = x*x
        let y2 = y*y
        let z2 = z*z
        let r2 = x2 + y2 + z2
        return (norm * Math.pow(x, xmap["x"])
                     * Math.pow(y, xmap["y"])
                     * Math.pow(z, xmap["z"])
                     * Math.exp(-alpha * r2))
    }
    return func
}


let Hydrogenic = (which) => {
    let Z = 1
    let z32 = Math.pow(Z, 3/2)
    let sqpi = Math.sqrt(Math.PI)
    let sq2pi = Math.sqrt(2 * Math.PI)
    let sq3pi = Math.sqrt(3 * Math.PI)
    let rnorm = 1 / Math.sqrt(2430)
    let func = (x, y, z) => {
        let norm, prefac, ret, ynorm, ybody
        let x2 = x * x
        let y2 = y * y
        let z2 = z * z
        let r2 = x2 + y2 + z2
        let r = Math.sqrt(r2)
        let sigma = Z * r
        let rbody = (z32 * Math.pow(2 / 3 * sigma, 2)
                         * Math.exp(-sigma / 3))
        switch (which) {
            case "1s":
                ret = 1 / sqpi * z32 * Math.exp(-sigma)
                break
            case "2s":
                norm = 1 / (4 * sq2pi) * z32
                ret = norm * (2 - sigma) * Math.exp(-sigma / 2)
                break
            case "2pz":
                norm = 1 / (4 * sq2pi) * z32
                ret = norm * Z * z * Math.exp(-sigma / 2)
                break
            case "2py":
                norm = 1 / (4 * sq2pi) * z32
                ret = norm * Z * y * Math.exp(-sigma / 2)
                break
            case "2px":
                norm = 1 / (4 * sq2pi) * z32
                ret = norm * Z * x * Math.exp(-sigma / 2)
                break
            case "3s":
                norm = 1 / (81 * sq3pi) * z32
                prefac = (27 - 18 * sigma + 2 * Math.pow(sigma, 2))
                ret = norm * prefac * Math.exp(-sigma / 3)
                break
            case "3pz":
                norm = Math.sqrt(2) / (81 * sqpi) * z32
                prefac = Z * (6 - sigma)
                ret = norm * prefac * z * Math.exp(-sigma / 3)
                break
            case "3py":
                norm = Math.sqrt(2) / (81 * sqpi) * z32
                prefac = Z * (6 - sigma)
                ret = norm * prefac * y * Math.exp(-sigma / 3)
                break
            case "3px":
                norm = Math.sqrt(2) / (81 * sqpi) * z32
                prefac = Z * (6 - sigma)
                ret = norm * prefac * x * Math.exp(-sigma / 3)
                break
            case "3d0":
                ynorm = 1 / 4 * Math.sqrt(5 / Math.PI)
                ybody = (-x2 -y2 + 2 * z2) / r2
                ret = ynorm * ybody * rnorm * rbody
                break
            case "3d+1":
                ynorm = 1 / 2 * Math.sqrt(15 / Math.PI)
                ybody = z * x / r2
                ret = ynorm * ybody * rnorm * rbody
                break
            case "3d+2":
                ynorm = 1 / 4 * Math.sqrt(15 / Math.PI)
                ybody = (x2 - y2) / r2
                ret = ynorm * ybody * rnorm * rbody
                break
            case "3d-1":
                ynorm = 1 / 2 * Math.sqrt(15 / Math.PI)
                ybody = y * z / r2
                ret = ynorm * ybody * rnorm * rbody
                break
            case "3d-2":
                rbody = z32 * Math.pow(sigma, 2) * Math.exp(-sigma / 3)
                ynorm = 1 / 4 * Math.sqrt(15 / Math.PI)
                ybody = x * y / r2
                ret = ynorm * ybody * rnorm * rbody
                break
        }
        return ret
    }
    return func
}


// Some numerical constants for convenience in the formulas below
let t2 = Math.sqrt(2)
let t3 = Math.sqrt(3)
let t5 = Math.sqrt(5)
let t6 = Math.sqrt(6)
let t7 = Math.sqrt(7)
let t10 = Math.sqrt(10)
let t14 = Math.sqrt(14)
let t15 = Math.sqrt(15)
let t21 = Math.sqrt(21)
let t30 = Math.sqrt(30)
let t35 = Math.sqrt(35)
let t42 = Math.sqrt(42)
let t70 = Math.sqrt(70)
let t105 = Math.sqrt(105)
let t154 = Math.sqrt(154)
let t210 = Math.sqrt(210)
let t231 = Math.sqrt(231)
let t429 = Math.sqrt(429)
let t462 = Math.sqrt(462)
let t6006 = Math.sqrt(6006)
let r2 = (r) => Math.pow(r, 2)
let r4 = (r) => Math.pow(r, 4)
let r6 = (r) => Math.pow(r, 6)

/*
Real cartesian solid harmonic type functions
These were computed using the python side:
exatomic.algorithms.basis.solid_harmonics
*/
let Harmonic = (l, m) => {
    let sh
    if ((l == 0) && (m == 0)) {
        sh = (x, y, z) => { return 1 }
    } else if ((l == 1) && (m == -1)) {
        sh = (x, y, z) => { return y }
    } else if ((l == 1) && (m == 0)) {
        sh = (x, y, z) => { return z }
    } else if ((l == 1) && (m == 1)) {
        sh = (x, y, z) => { return x }
    } else if ((l == 2) && (m == -2)) {
        sh = (x, y, z) => { return (t3*x*y) }
    } else if ((l == 2) && (m == -1)) {
        sh = (x, y, z) => { return (t3*y*z) }
    } else if ((l == 2) && (m == 0)) {
        sh = (x, y, z) => { return (-x*x/2 - y*y/2 + z*z) }
    } else if ((l == 2) && (m == 1)) {
        sh = (x, y, z) => { return (t3*x*z) }
    } else if ((l == 2) && (m == 2)) {
        sh = (x, y, z) => { return (t3/2*(x*x - y*y)) }
    } else if ((l == 3) && (m == -3)) {
        sh = (x, y, z) => { return (t10/4*y*(3*x*x - y*y)) }
    } else if ((l == 3) && (m == -2)) {
        sh = (x, y, z) => { return (t15*x*y*z) }
    } else if ((l == 3) && (m == -1)) {
        sh = (x, y, z) => { return (t6/4*y*(-x*x - y*y + 4*z*z)) }
    } else if ((l == 3) && (m == 0)) {
        sh = (x, y, z) => { return (z/2*(-3*x*x - 3*y*y + 2*z*z)) }
    } else if ((l == 3) && (m == 1)) {
        sh = (x, y, z) => { return (t6/4*x*(-x*x -y*y + 4*z*z)) }
    } else if ((l == 3) && (m == 2)) {
        sh = (x, y, z) => { return (t15/2*z*(x*x - y*y)) }
    } else if ((l == 3) && (m == 3)) {
        sh = (x, y, z) => { return (t10/4*x*(x*x - 3*y*y)) } 
    } else if ((l == 4) && (m == -4)) {
        sh = (x, y, z) => { return (t35/2*x*y*(x*x -y*y)) }
    } else if ((l == 4) && (m == -3)) {
        sh = (x, y, z) => { return (t70/4*y*z*(3*x*x - y*y)) }
    } else if ((l == 4) && (m == -2)) {
        sh = (x, y, z) => { return (t5/2*x*y*(-x*x - y*y +6*z*z)) }
    } else if ((l == 4) && (m == -1)) {
        sh = (x, y, z) => { return (t10/4*y*z*(-3*x*x - 3*y*y + 4*z*z)) }
    } else if ((l == 4) && (m == 0)) {
        sh = (x, y, z) => { return (3*r4(x)/8 + 3*x*x*y*y/4 - 3*x*x*z*z + 3*r4(y)/8 - 3*y*y*z*z + r4(z)) }
    } else if ((l == 4) && (m == 1)) {
        sh = (x, y, z) => { return (t10/4*x*z*(-3*x*x - 3*y*y + 4*z*z)) }
    } else if ((l == 4) && (m == 2)) {
        sh = (x, y, z) => { return (t5/4*(x*x - y*y)*(-x*x - y*y + 6*z*z)) }
    } else if ((l == 4) && (m == 3)) {
        sh = (x, y, z) => { return (t70/4*x*z*(x*x - 3*y*y)) }
    } else if ((l == 4) && (m == 4)) {
        sh = (x, y, z) => { return (t35/8*(r4(x) - 6*x*x*y*y + r4(y))) }
    } else if ((l == 5) && (m == -5)) {
        sh = (x, y, z) => { return (3/16*t14*y*(5*r4(x) - 10*x*x*y*y + r4(y))) }
    } else if ((l == 5) && (m == -4)) {
        sh = (x, y, z) => { return (3/2*t35*x*y*z*(x*x - y*y)) }
    } else if ((l == 5) && (m == -3)) {
        sh = (x, y, z) => { return (t70/16*y*(3*x*x - y*y)*(-x*x - y*y + 8*z*z)) }
    } else if ((l == 5) && (m == -2)) {
        sh = (x, y, z) => { return (t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z)) }
    } else if ((l == 5) && (m == -1)) {
        sh = (x, y, z) => { return (t15/8*y*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z))) }
    } else if ((l == 5) && (m == 0)) {
        sh = (x, y, z) => { return (z/8*(15*r4(x) + 30*x*x*y*y - 40*x*x*z*z + 15*r4(y) - 40*y*y*z*z + 8*r4(z))) }
    } else if ((l == 5) && (m == 1)) {
        sh = (x, y, z) => { return (t15/8*x*(r4(x) + 2*x*x*y*y - 12*x*x*z*z + r4(y) - 12*y*y*z*z + 8*r4(z))) }
    } else if ((l == 5) && (m == 2)) {
        sh = (x, y, z) => { return (t105/4*z*(x*x - y*y)*(-x*x - y*y + 2*z*z)) }
    } else if ((l == 5) && (m == 3)) {
        sh = (x, y, z) => { return (t70/16*x*(9*z*z*(x*x - 3*y*y) + (-x*x + 3*y*y)*(x*x + y*y + z*z))) }
    } else if ((l == 5) && (m == 4)) {
        sh = (x, y, z) => { return (3/8*t35*z*(r4(x) - 6*x*x*y*y + r4(y))) }
    } else if ((l == 5) && (m == 5)) {
        sh = (x, y, z) => { return (3/16*t14*x*(r4(x) - 10*x*x*y*y + 5*r4(y))) }
    } else if ((l == 6) && (m == -6)) {
        sh = (x, y, z) => { return (t462/16*x*y*(3*r4(x) -10*x*x*y*y + 3*r4(y))) }
    } else if ((l == 6) && (m == -5)) {
        sh = (x, y, z) => { return (3*t154*y*z*(5*r4(x) - 10*x*x*y*y + r4(y))/16) }
    } else if ((l == 6) && (m == -4)) {
        sh = (x, y, z) => { return (3*t7*x*y*(x*x - y*y)*(-x*x - y*y + 10*z*z)/4) }
    } else if ((l == 6) && (m == -3)) {
        sh = (x, y, z) => { return (t210/16*y*z*(3*x*x - y*y)*(-3*x*x - 3*y*y + 8*z*z)) }
    } else if ((l == 6) && (m == -2)) {
        sh = (x, y, z) => { return (t210*x*y*(r4(x) + 2*x*x*y*y - 16*x*x*z*z + r4(y) - 16*y*y*z*z + 16*r4(z))/16) }
    } else if ((l == 6) && (m == -1)) {
        sh = (x, y, z) => { return (t21*y*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8) }
    } else if ((l == 6) && (m == 0)) {
        sh = (x, y, z) => { return (-5*r6(x)/16 - 15*r4(x)*y*y/16 + 45*r4(x)*z*z/8 - 15*x*x*r4(y)/16 + 45*x*x*y*y*z*z/4 - 15*x*x*r4(z)/2 - 5*r6(y)/16 + 45*r4(y)*z*z/8 - 15*y*y*r4(z)/2 + r6(z)) }
    } else if ((l == 6) && (m == 1)) {
        sh = (x, y, z) => { return (t21*x*z*(5*r4(x) + 10*x*x*y*y - 20*x*x*z*z + 5*r4(y) - 20*y*y*z*z + 8*r4(z))/8) }
    } else if ((l == 6) && (m == 2)) {
        sh = (x, y, z) => { return (-t210/32*(x*x - y*y)*(11*z*z*(x*x + y*y - 2*z*z) - (x*x + y*y - 6*z*z)*(x*x + y*y + z*z))) }
    } else if ((l == 6) && (m == 3)) {
        sh = (x, y, z) => { return (t210/16*x*z*(-x*x + 3*y*y)*(3*x*x + 3*y*y - 8*z*z)) }
    } else if ((l == 6) && (m == 4)) {
        sh = (x, y, z) => { return (3*t7/16*(-r6(x) + 5*r4(x)*y*y + 10*r4(x)*z*z + 5*x*x*r4(y) - 60*x*x*y*y*z*z - r6(y) + 10*r4(y)*z*z)) }
    } else if ((l == 6) && (m == 5)) {
        sh = (x, y, z) => { return (3/16*t154*x*z*(r4(x) - 10*x*x*y*y + 5*r4(y))) }
    } else if ((l == 6) && (m == 6)) {
        sh = (x, y, z) => { return (t462/32*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y))) }
    } else if ((l == 7) && (m == -7)) {
        sh = (x, y, z) => { return (t429/32*y*(7*r6(x) - 35*r4(x)*y*y + 21*x*x*r4(y) - r6(y))) }
    } else if ((l == 7) && (m == -6)) {
        sh = (x, y, z) => { return (t6006/16*x*y*z*(3*r4(x) - 10*x*x*y*y + 3*r4(y))) }
    } else if ((l == 7) && (m == -5)) {
        sh = (x, y, z) => { return (t231/32*y*(-x*x - y*y + 12*z*z)*(-x*x*(-x*x + 3*y*y) + 4*x*x*(x*x - y*y) - y*y*(3*x*x - y*y))) }
    } else if ((l == 7) && (m == -4)) {
        sh = (x, y, z) => { return (t231/4*x*y*z*(x*x - y*y)*(-3*x*x - 3*y*y + 10*z*z)) }
    } else if ((l == 7) && (m == -3)) {
        sh = (x, y, z) => { return (t21/32*y*(3*x*x - y*y)*(z*z*(-39*x*x - 39*y*y + 104*z*z) - 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z))) }
    } else if ((l == 7) && (m == -2)) {
        sh = (x, y, z) => { return (t42/16*x*y*z*(15*r4(x) + 30*x*x*y*y - 80*x*x*z*z + 15*r4(y) - 80*y*y*z*z + 48*r4(z))) }
    } else if ((l == 7) && (m == -1)) {
        sh = (x, y, z) => { return (t7/32*y*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z))) }
    } else if ((l == 7) && (m == 0)) {
        sh = (x, y, z) => { return (z/16*(-35*r6(x) - 105*r4(x)*y*y + 210*r4(x)*z*z - 105*x*x*r4(y) + 420*x*x*y*y*z*z - 168*x*x*r4(z) - 35*r6(y) + 210*r4(y)*z*z - 168*y*y*r4(z) + 16*r6(z))) }
    } else if ((l == 7) && (m == 1)) {
        sh = (x, y, z) => { return (t7/32*x*(-5*r6(x) - 15*r4(x)*y*y + 120*r4(x)*z*z - 15*x*x*r4(y) + 240*x*x*y*y*z*z - 240*x*x*r4(z) - 5*r6(y) + 120*r4(y)*z*z - 240*y*y*r4(z) + 64*r6(z))) }
    } else if ((l == 7) && (m == 2)) {
        sh = (x, y, z) => { return (t42/32*z*(15*r6(x) + 15*r4(x)*y*y - 80*r4(x)*z*z - 15*x*x*r4(y) + 48*x*x*r4(z) - 15*r6(y) + 80*r4(y)*z*z - 48*y*y*r4(z))) }
    } else if ((l == 7) && (m == 3)) {
        sh = (x, y, z) => { return (t21/32*x*(-x*x + 3*y*y)*(z*z*(39*x*x + 39*y*y - 104*z*z) + 3*(-x*x - y*y + 8*z*z)*(x*x + y*y + z*z))) }
    } else if ((l == 7) && (m == 4)) {
        sh = (x, y, z) => { return (t231/16*z*(-3*r6(x) + 15*r4(x)*y*y + 10*r4(x)*z*z + 15*x*x*r4(y) - 60*x*x*y*y*z*z - 3*r6(y) + 10*r4(y)*z*z)) }
    } else if ((l == 7) && (m == 5)) {
        sh = (x, y, z) => { return (t231/32*x*(-r6(x) + 9*r4(x)*y*y + 12*r4(x)*z*z + 5*x*x*r4(y) - 120*x*x*y*y*z*z - 5*r6(y) + 60*r4(y)*z*z)) }
    } else if ((l == 7) && (m == 6)) {
        sh = (x, y, z) => { return (t6006/32*z*(r6(x) - 15*r4(x)*y*y + 15*x*x*r4(y) - r6(y))) }
    } else if ((l == 7) && (m == 7)) {
        sh = (x, y, z) => { return (t429/32*x*(r6(x) - 21*r4(x)*y*y + 305*x*x*r4(y) -7*r6(y))) }
    } else {
        console.log("not implemented")
    }
    let func = (x, y, z) => {
        let r2 = x * x + y * y + z * z
        return sh(x, y, z) * Math.exp(-Math.sqrt(r2));
    }
    return func
}


module.exports = {
    create_float_array_xyz: create_float_array_xyz,
    //normalize_gaussian: normalize_gaussian,
    gen_field_arrays: gen_field_arrays,
    compute_field: compute_field,
    repeat_float: repeat_float,
    repeat_obj: repeat_obj,
    factorial2: factorial2,
    linspace: linspace,
    mapper: mapper,
    scalar_field: scalar_field,
    Hydrogenic: Hydrogenic,
    Gaussian: Gaussian,
    Harmonic: Harmonic,
    Ellipsoid: Ellipsoid,
    Sphere: Sphere,
    Torus: Torus,
    jsonparse: jsonparse,
    fparse: fparse,
    mesolv: mesolv,
    resolv: resolv
};
