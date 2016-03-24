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
        'nbextensions/exa/threejs': {
            deps: ['nbextensions/exa/lib/three.min'],
            exports: 'three'
        },

        'nbextensions/exa/lib/dat.gui.min': {
            exports: 'dat'
        },

        'nbextensions/exa/lib/three.min': {
            exports: 'THREE'
        },
    },
});


define([
    'nbextensions/exa/threejs',
    'nbextensions/exa/lib/dat.gui.min',
    'nbextensions/exa/lib/three.min'
], function(three, dat, THREE) {
    var ThreeJS = three.ThreeJS;
    var GUI = dat.GUI;

    GUI.prototype._init = function(view) {
        this.view = view;
        this.f1 = this.addFolder('animation');
        this.f2 = this.addFolder('atoms');
        this.f3 = this.addFolder('bonds');
        this.f4 = this.addFolder('cell');
        this.f5 = this.addFolder('surfaces');
        this.f6 = this.addFolder('volumes');
        console.log(this.view);
    };

    var style_str = ".dg {\
            color: black;\
            font: 400 13px Verdana, Arial, sans-serif;\
            text-shadow: white 0 0 0;\
        }\
        .hue-field {\
            width: 10;\
        }\
        .dg .c .slider {\
            background: white\
        }\
        .dg .c .slider:hover {\
            background: white\
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
            outline-color: lightgrey;\
            outline-style: solid;\
            outline-width: 1.5px\
        }\
        .dg .c input[type=text]:focus {\
            background: white;\
            color: black;\
            outline-color: lightgrey;\
            outline-style: solid;\
            outline-width: 1.5px\
        }\
        .dg .c input[type=text]:hover {\
            background: white;\
            color: black;\
            outline-color: lightgrey;\
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
            border-bottom: 1px solid white;\
            border-right: 1px solid white\
        }\
        .dg .cr.function:hover {\
            background: white;\
            border-bottom: 1px solid white;\
            border-right: 1px solid white\
        }\
        .dg li.cr {\
            background: #fafafa;\
            border-bottom: 1px solid white;\
            border-right: 1px solid white\
        }\
        .dg li.cr:hover {\
            background: white;\
            border-bottom: 1px solid white;\
            border-right: 1px solid white\
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
            outline-color: lightgrey;\
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
            outline-color: lightgrey;\
            outline-style: solid;\
            outline-width: 1.5px\
        }";

    return {
        'ThreeJS': ThreeJS,
        'GUI': GUI,
        'gui_style': style_str
    };
});
