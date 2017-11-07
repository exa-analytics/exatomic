// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
jupyter-exatomic-editor
#################################
*/
declare function require(name: string);

var base  = require("./jupyter-exatomic-base.js"); 


export class EditorSceneModel extends base.ExatomicSceneModel {
    defaults(): any {
        return {...super.defaults(), ...{
            '_view_name': "EditorSceneView",
            '_model_name': "EditorSceneModel"
        }};
    }
}

export class EditorSceneView extends base.ExatomicSceneView {
    init(): any {
        super.init();
        // Is it a problem, reassigning this?
        this.three_promises = this.three_promises.then(this.worker.bind(this));
    }

    worker(): any {
        console.log("in worker....");
        console.log(this);
        console.log(this.app3d);
        console.log(this.app3d.scene);
    }
}
