/*"""
=========================================================
Universe View
=========================================================
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'container'
        },

        'nbextensions/exa/atomic/app': {
            exports: 'app'
        }
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/container',
    'nbextensions/exa/atomic/app',
], function(widget, container, app){
    var UniverseView = container.ContainerView.extend({
        /*"""
        UniverseView
        ====================
        Frontend representation of an atomic universe instance.
        */
        render: function() {
            /*"""
            render
            --------------
            Main entry point for the universe container frontend.
            */
            console.log('Initializing universe...');
            var self = this;
        },
    });

    return {'UniverseView': UniverseView};
});
