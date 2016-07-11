// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
gaussian.js
################
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/exatomic/field': {
            exports: 'field'
        },
        'nbextensions/exa/exatomic/harmonics': {
            exports: 'sh'
        },
        'nbextensions/exa/num': {
            exports: 'num'
        }
    },
});


define([
    'nbextensions/exa/exatomic/field',
    'nbextensions/exa/exatomic/harmonics',
    'nbextensions/exa/num'
], function(AtomicField, sh, num) {
    class GaussianOrbital extends AtomicField {
        /*"""
        GaussianOrbital
        ========================
        The necessary machinery required to generate GaussianOrbitals
        is laid out in the functions order_gtf_basis and construct_mos.
        This is merely a thin wrapper around the AtomicField class.
        */
        constructor(dims, func) {
            super(dims, func)
        };
    };

    var construct_mos = function(bfns, coefs, dims) {
        /*"""
        construct_mos
        =======================
        From a list of properly formatted strings representing basis functions,
        take appropriate linear combinations of them weighted by MO coefficients
        to generate the full string representation of the MOs.
        */

        // Compute the normalization factor for each basis function independently
        var norms = [];
        var nbfns = bfns.length;
        for (var bf = 0; bf < nbfns; bf++) {
            var func = new Function('x,y,z', 'return '.concat(bfns[bf]));
            norms.push(num.compute_field(dims.x, dims.y, dims.z, dims.n, func)['norm']);
        };
        var mos = [];
        for (var i = 0; i < nbfns; i++) {
            var mo = 'return ';
            for (var j = 0; j < nbfns; j++) {
                if (j == nbfns - 1) {
                    mo = mo.concat(coefs[j][i].toFixed(12), ' * ', norms[j], ' * (', bfns[j], ')');
                } else {
                    mo = mo.concat(coefs[j][i].toFixed(12), ' * ', norms[j], ' * (', bfns[j], ') + ');
                };
            };
            mos.push(new Function('x,y,z', mo));
        };
        return mos;
    };

    var order_gtf_basis = function(xs, ys, zs, sets, nbfns, d, l, alpha, pl, pm, pn, sgto) {
      /*"""
      order_gtf_basis
      =====================
      Unpack the nested data structures to generate strings representing
      basis functions. This necessarily assumes some knowledge of the way
      a computational code orders basis functions, therefore it will not work
      for all cases and a different function may be needed.
      */
      /*
        console.log('inside order_gtf_basis');
        console.log('xs');
        console.log(xs);
        console.log('ys');
        console.log(ys);
        console.log('zs');
        console.log(zs);
        console.log('sets');
        console.log(sets);
        console.log('nbfns');
        console.log(nbfns);
        console.log('d');
        console.log(d);
        console.log('l');
        console.log(l);
        console.log('pl');
        console.log(pl);
        console.log('pm');
        console.log(pm);
        console.log('pn');
        console.log(pn);
        console.log('alpha');
        console.log(alpha);
        console.log('sgto');
        console.log(sgto);
        */
        var bfns = [];
        var nat = xs.length;
        for (var atom = 0; atom < nat; atom++) {
            //console.log('atom');
            //console.log(atom);
            // Here we skip accounting for the xa**l * ya**m * za**n
            // pre-factor because we are only using s basis functions for the demo
            // but looks something like (defined a bit below this comment):
            // var preface = xa**pl[idx] * ya**pm[idx] * za**pn[idx]
            var set = sets[atom];
            var xa = ''.concat('x - ', xs[atom].toFixed(6));
            var ya = ''.concat('y - ', ys[atom].toFixed(6));
            var za = ''.concat('z - ', zs[atom].toFixed(6));
            var x2 = ''.concat('Math.pow(', xa, ', 2)');
            var y2 = ''.concat('Math.pow(', ya, ', 2)');
            var z2 = ''.concat('Math.pow(', za, ', 2)');
            var r2 = ''.concat('(', x2, ' + ', y2, ' + ', z2, ')');
            var ds = d[set];
            var ls = l[set];
            var alphas = alpha[set];
            var nshells = alphas.length;
            for (var ns = 0; ns < nshells; ns++) {
                //console.log('nshell');
                //console.log(ns);
                var bstr = '';
                var exps = alphas[ns];
                var cs = ds[ns];
                var ang = ls[ns][0];
                var nprim = exps.length;
                // This doesn't account for degeneracy due to m_l
                // Assuming that it would have to be a loop outside
                // iteration over the 'primitives' as they all share
                // exponents and coefficients
                //var mls = sgto[angs[prim]];
                //var mldegen = mls.length;
                //for (var ml = 0; ml < mldegen; ml++) {
                //    iterate over primitives in here.
                // }
                //var degen = sgto[ang];
                //for (var car = 0; car < degen; car++) {
                for (var prim = 0; prim < nprim; prim++) {
                    //console.log('prim = ', prim);
                    if (prim == nprim - 1) {
                        bstr = bstr.concat(cs[prim].toFixed(8), ' * ', 'Math.exp(-', exps[prim], ' * ', r2, ')');
                    } else {
                        bstr = bstr.concat(cs[prim].toFixed(8), ' * ', 'Math.exp(-', exps[prim], ' * ', r2, ') + ');
                    };
                };
                bfns.push(bstr);
            };
        };
        return bfns
    };

    return {
        GaussianOrbital: GaussianOrbital,
        order_gtf_basis: order_gtf_basis,
        construct_mos: construct_mos
    };

});
