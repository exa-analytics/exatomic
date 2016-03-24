/*"""
===========================
Universe Application
===========================
This module defines the JavaScript application that is loaded when a user
requests the HTML representation of a universe data container.
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/lib/dat.gui.min': {
            exports: 'dat'
        },

        'nbextensions/exa/three.app': {
            exports: 'app3D'
        },
    },
});


define([
    'nbextensions/exa/lib/dat.gui.min',
    'nbextensions/exa/three.app'
], function(dat, app3D) {
    var AtomicApp = function() {
        /*"""
        AtomicApp
        ============
        A class like object that handles communication between the Universe
        container (Python backend) and the exa three.app application frontend
        (JavaScript).
        */
    };

    return {'AtomicApp': AtomicApp};
});
