// Copyright (c) 2015-2017, Exa Analytics Development Team
// Distributed under the terms of the Apache License 2.0
/*"""
jupyter-exatomic-editor
#################################
*/
declare function require(name: string);

var base = require("./exatomic-base.js"); 
var three = require("three");


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
        var that = this;
        this.el.addEventListener('click', that.clicked);
        console.log("added listener");
    }

    add_sphere(x, y): any {
        let geometry = new three.SphereGeometry(2, 32, 32);
        let material = new three.MeshPhongMaterial(color="blue");
        let sphere = new three.Mesh(geometry, material);
        sphere.position = new three.Vector3(x, y, 0);
        console.log(sphere);
        this.app3d.meshes['editor'] = [sphere];
        this.app3d.add_meshes('editor');
    }

    del_sphere(x, y): any {
        console.log("Good luck deleting....");
    }

    clicked(e): any {
        console.log("clicked!");
        switch (e.button) {
            case 0: 
                this.add_sphere(e.clientX, e.clientY);
            case 2:
                this.del_sphere(e.clientX, e.clientY);
        }
    }
}
