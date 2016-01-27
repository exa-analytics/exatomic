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
        'nbextensions/exa/atomic/gui': {
            exports: 'GUI'
        },
    },
});


define([
    'widgets/js/widget',
    'nbextensions/exa/atomic/threejs',
    'nbextensions/exa/atomic/gui'
], function(widget, AtomicThreeJS, GUI){
    var UniverseController = function(view) {
    };
    
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
            this.frames = this.model.get('_framelist');
            console.log(this.frames);
            this.sidebar_width = 300;
            this.width = this.model.get('width');
            this.height = this.model.get('height');
            console.log(this.height);
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
            this.canvas = $('<canvas/>').width(this.width - this.sidebar_width).height(this.height);
            this.canvas.css('position', 'absolute');
            this.canvas.css('top', 0);
            this.canvas.css('left', this.sidebar_width);
            this.button = $('<button/>').width(64).height(16).click(function(event) {
                self.callme();
            });
            this.threejsapp = new AtomicThreeJS(this.canvas);
            this.threejsgui = new GUI(this.sidebar_width, this.frames, this);
            this.gui_style = document.createElement('style');
            this.gui_style.innerHTML = this.threejsgui.style_str;
            this.container.append(this.canvas);
            this.container.append(this.threejsgui.domElement);
            this.container.append(this.button);
            this.container.append(this.gui_style);
            this.setElement(this.container);

            // By default draw the system
            this.x = this.get('_atom_x');
            this.y = this.get('_atom_y');
            this.z = this.get('_atom_z');
            this.r = this.get('_atom_radius');
            this.c = this.get('_atom_color');
            this.filled = true;
            console.log(this.x);
            //this.threejsapp.add_points(this.x[0], y[0], z[0], r[0], c[0], filled);
            this.update_points(this.frames[0]);
            var cam = this.get('_camera');
            this.threejsapp.camera.position.x = cam[0];
            this.threejsapp.camera.position.y = cam[1];
            this.threejsapp.camera.position.z = cam[2];
            var cen = this.get('_center');
            var center = new THREE.Vector3(cen[0], cen[1], cen[2]);
            this.threejsapp.camera.lookAt(center);
            this.threejsapp.controls.target = center;
            this.threejsapp.render();
            this.on('displayed', function () {
                self.threejsapp.animate();
                self.threejsapp.controls.handleResize();
            });
            this.$el.height(this.height);
        },

        update_points: function(frame) {
            var x = this.x[frame];
            var y = this.y[frame];
            var z = this.z[frame];
            var r = this.r[frame];
            var c = this.c[frame];
            this.threejsapp.add_points(x, y, z, r, c, this.filled);
            this.threejsapp.render();
        },

        get: function(name) {
            /*"""
            Get
            ````````````
            Custom getter for Python objects stored as json strings
            */
            return JSON.parse(this.model.get(name));
        },

        callme: function() {
            console.log('called callme');
            console.log(this.model);
            var x = this.model.get('_atom_x');
            console.log(x);
            var y = this.model.get('_atom_y');
            console.log(y);
            this.send({'call': 'update'});
        },
    });

    return {'UniverseView': UniverseView};
});
