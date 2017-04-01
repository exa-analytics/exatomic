// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
===============
gtf.js
===============
*/
'use strict';
var num = require("jupyter-exawidgets").num;
var field = require("jupyter-exawidgets").field;

/*
require.config({
    shim: {
        "nbextensions/exa/num": {exports: 'num'},
        "nbextensions/exa/field": {exports: 'field'}
    },
});
*/
/*
define([
    "nbextensions/exa/num",
    "nbextensions/exa/field"
],
*/
//function(num, field) {
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
        return num.normalize_gaussian(25000, 0) * Math.exp(-r2);
    },
    'px': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 1) * x * Math.exp(-r2);
    },

    'py': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 1) * y * Math.exp(-r2);
    },

    'pz': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 1) * z * Math.exp(-r2);
    },

    'd200': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 2) * x2 * Math.exp(-r2);
    },

    'd110': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        var r = Math.sqrt(r2);
        return num.normalize_gaussian(25000, 2) * x * y * Math.exp(-r2);
    },

    'd101': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 2) * x * z * Math.exp(-r2);
    },


    'd020': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 2) * y2 * Math.exp(-r2);
    },

    'd011': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 2) * y * z * Math.exp(-r2);
    },

    'd002': function(x, y, z) {
        var x2 = x*x;
        var y2 = y*y;
        var z2 = z*z;
        var r2 = x2 + y2 + z2;
        return num.normalize_gaussian(25000, 2) * z2 * Math.exp(-r2);
    },

};

module.exports = {
    "GTF": GTF
}
