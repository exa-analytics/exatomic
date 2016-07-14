// Copyright (c) 2015-2016, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
field.js
###############
*/
'use strict';

require.config({
    shim: {
        'nbextensions/exa/field': {
            exports: 'field'
        },
        'nbextensions/exa/num': {
            exports: 'num'
        }
    },
});


define([
    'nbextensions/exa/field',
    'nbextensions/exa/num'
], function(field, num) {
    class AtomicField extends field.ScalarField {
        /*"""
        AtomicField
        =============
        JS repr. of .cube file field values and dimensions.
        */
        constructor(dimensions, values) {
            super(dimensions, values);
        };

    };

    return AtomicField;

});
