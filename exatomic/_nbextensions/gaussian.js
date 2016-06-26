/*"""
==================
gaussian.js
==================
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/field': {
            exports: 'field'
        },
        'nbextensions/exa/exatomic/harmonics': {
            exports: 'sh'
        },
    },
});


define([
    'nbextensions/exa/field',
    'nbextensions/exa/exatomic/harmonics'
], function(field, sh) {
    class GaussianTypeFunction extends field.ScalarField {
        /*"""
        GaussianTypeFunction
        ========================
        */
        constructor(dimensions, field) {
            super(dimensions, field);
        };
    };

    class SphericalGaussian extends GaussianTypeFunction {
        /*"""
        SphericaGaussian
        ====================
        */
        constructor(momatrix, basis_set) {
            console.log(momatrix);
            console.log(basis_set);
            var p = solid_harmonics(4);
            console.log(p);
        };
    };

    // Some constants
    var pi = Math.PI;

    var spherical_gtf = function(x, y, z, alpha, l, m) {
        /*"""
        partial_spherical_gtf
        ==========================
        Full spherical Gaussian type function is:

        .. math::

            f(\mathbf{r}; p, \alpha, l, \mathbf{r}_{A}) = N\left(\alpha, p\right)p\left(\mathbf{r}-\mathbf{r}_{A}\right)
        */
        var arg = 2*l - 1;
        //if (Math.mod(arg, 2) == 0) {

        //}
        //var N = Math.sqrt(Math.pow(2*alpha/pi, 1.5)*(4*alpha*l))
        var r2 = x*x + y*y + z*z
        return Math.exp(-alpha*r2);
    };


    return {SphericalGaussian: SphericalGaussian};
});
