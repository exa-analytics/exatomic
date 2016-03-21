/*"""
IPython Widget for Visualization of the atomic Universe
`````````````````````````````````````````````````````````
This extension is allows for visualization of the atomc Universe within a
Jupyter notebook environment. Visualization utilizes the three.js library.
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/container': {
            exports: 'Container'
        },
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/container'
], function(widget, Container){
    var ContainerView = Container['ContainerView'];


    var UniverseView = ContainerView.extend({
        /*"""
        UniverseView
        ``````````````````
        The UniverseView object is responsible for creating the necessary HTML
        elements required for displaying the 3D representation of the Universe
        container. It is also responsible for handling messages passed between
        the JavaScript frontend and Python backend. Drawing and rendering
        atoms, bonds, volumes, etc is handled by the atomic threejs application,
        AtomicThreeJS (in threejs.js).
        */
        init: function() {
            /*"""
            Render
            ````````````
            This is the main entry point for UniverseView. For more information
            see the documentation of Backbone.js. Here we create the widget
            three.js viewport and gui controls.
            */
            var self = this;    // Alias the instance of the view for future ref.
            console.log('universe');
            console.log(self);
            console.log(this.value_changed);
        },
    });

    return {'UniverseView': UniverseView};
});
