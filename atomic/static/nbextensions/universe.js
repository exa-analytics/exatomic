/*"""
IPython Widget View of the Universe Container
````````````````````````````````````````````````
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/atomic/threejs': {
            exports: 'AtomicThreeJS'
        },
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/atomic/threejs'
], function(widget, AtomicThreeJS){
    var UniverseView = widget.DOMWidgetView.extend({
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
        render: function() {
            /*"""
            Render
            ````````````
            This is the main entry point for UniverseView. For more information
            see the documentation of Backbone.js.
            */
            var self = this;    // Alias the instance of the view for future ref.
            this.width = this.model.get('width');
            this.height = this.model.get('height');
            this.container = $('<div/>').width(this.width).height(this.height).resizable({
                aspectRatio: false,
                resize: function(event, ui) {
                    self.width = ui.size.width;
                    self.height = ui.size.height;
                    self.model.set('width', self.width);
                    self.model.set('height', self.height);
                    self.canvas.width(self.width - 300);
                    self.canvas.height(self.height);
                    self.threejsapp.resize();
                },
                stop: function(event, ui) {
                    self.threejsapp.render();
                }
            });
            this.canvas = $('<canvas/>').width(this.width - 300).height(this.height);
            this.button = $('<button/>').width(64).height(16).click(function(event) {
                self.callme();
            });
            this.threejsapp = new AtomicThreeJS(this.canvas);
            this.container.append(this.button);
            this.container.append(this.canvas);
            this.setElement(this.container);
            this.threejsapp.render();
        },

        callme: function() {
            console.log('called callme');
            console.log(this.model);
            var got = this.model.get('_atom_x');
            console.log(got[0][1]);
            this.send({'call': 'update'});
        },
    });

    return {'UniverseView': UniverseView};
});
