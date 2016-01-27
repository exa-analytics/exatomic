/*"""
datgui
````````````````````````````````
This code depended
*/
'use strict';


require.config({
    shim: {
        'nbextensions/exa/atomic/lib/dat.gui.min': {
            exports: 'dat'
        },
    }
})


define([
    'nbextensions/exa/atomic/lib/dat.gui.min'
], function(dat) {
    var GUI = function(view) {
        /*"""
        GUI Application
        ``````````````````
        */
        var self = this;
        this.view = view;

        this.gui = new dat.GUI({autoPlace: false, width: width});
        this.gui_dom = $(this.gui.domElement);
        this.gui_dom.css('position', 'absolute');
        this.gui_dom.css('top', 0);
        this.gui_dom.css('left', 0);
        this.domElement = this.gui.domElement;

        this.folder01 = this.gui.addFolder('animation');
        this.folder02 = this.gui.addFolder('atoms');
        this.folder03 = this.gui.addFolder('bonds');
        this.folder04 = this.gui.addFolder('cell');
        this.folder05 = this.gui.addFolder('surfaces');
        this.folder06 = this.gui.addFolder('volumes');

        this.fps = 15;
        this.frames = frames;
        this.cidx = 0;
        this.current_frame = this.frames[this.cidx];
        this.nframes = this.frames.length;

        this.playing = false;
        this.folder01_data = {
            'pause': function() {
                this.playing = false;
                clearInterval(this._play_callback_id);
            },
            'play': function() {
                var _self = this;
                if (this.playing == true) {
                    this.pause();
                } else {
                    this.playing = true;
                    this._play_callback_id = setInterval(function() {
                        if (self.cidx < this.nframes) {
                            self.cidx += 1;
                            self.current_frame = self.frames[self.cidx];
                        } else {
                            _self.pause();
                        };
                    }, 1000 / this.fps);
                };
            },
            'index': self.cidx,
            'fps': self.fps
        };

        this.folder01_listeners = {};
        this.folder01_listeners['play'] = this.folder01.add(this.folder01_data, 'play');
        this.folder01_listeners['index'] = this.folder01.add(
            this.folder01_data, 'index'
        ).min(0).max(this.nframes - 1).step(1);
        this.folder01_listeners['fps'] = this.folder01.add(this.folder01_data, 'fps').min(1).max(60).step(1);

        this.folder01_listeners['index'].onChange(function(index) {
            self.cidx = index;
            self.current_frame = self.frames[self.cidx];
            self.view.update_points(self.current_frame);
        });
    };

    GUI.prototype.style_str = ".dg {\
        color: black;\
        font: 400 13px Verdana, Arial, sans-serif;\
        text-shadow: white 0 0 0;\
    }\
    .hue-field {\
        width: 10;\
    }\
    .dg .c .slider {\
        background: silver\
    }\
    .dg .c .slider:hover {\
        background: silver\
    }\
    .dg .c input[type=text] {\
        background: white;\
        border-color: lightgrey;\
        border-radius: 2px;\
        border-style: solid;\
        border-width: 1.1px;\
        color: black\
    }\
    .dg .c input[type=text]:active {\
        background: white;\
        color: black;\
        outline-color: lightblue;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .c input[type=text]:focus {\
        background: white;\
        color: black;\
        outline-color: lightblue;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .c input[type=text]:hover {\
        background: white;\
        color: black;\
        outline-color: lightblue;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg .closed li.title {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: #777;\
        text-shadow: 1px 0 0 white\
    }\
    .dg .cr.boolean:hover {\
        background: white;\
        border-bottom: 1px solid silver;\
        border-right: 1px solid silver\
    }\
    .dg .cr.function:hover {\
        background: white;\
        border-bottom: 1px solid silver;\
        border-right: 1px solid silver\
    }\
    .dg li.cr {\
        background: #fafafa;\
        border-bottom: 1px solid silver;\
        border-right: 1px solid silver\
    }\
    .dg li.cr:hover {\
        background: white;\
        border-bottom: 1px solid silver;\
        border-right: 1px solid silver\
    }\
    .dg li.title, .dg closed {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: black;\
        text-shadow: 1px 0 0 white\
    }\
    .dg li.title:hover {\
        outline-color: lightblue;\
        outline-style: solid;\
        outline-width: 1.5px\
    }\
    .dg.main .close-button {\
        background: -moz-linear-gradient(center top, #ededed 34%, #dfdfdf 71%);\
        background: -ms-linear-gradient(top, #ededed 34%, #dfdfdf 71%);\
        background: -webkit-gradient(linear, left top, left bottom, color-stop(34%, #ededed),\
                     color-stop(71%, #dfdfdf));\
        background-color: #ededed;\
        border: 1px solid #dcdcdc;\
        border-radius: 2px;\
        box-shadow: inset 1px 0 9px 0 white;\
        color: black;\
        height: 27px;\
        line-height: 27px;\
        text-align: center;\
        text-shadow: 1px 0 0 white\
    }\
    .dg.main .close-button:hover {\
        outline-color: lightblue;\
        outline-style: solid;\
        outline-width: 1.5px\
    }";

    return GUI;
})
